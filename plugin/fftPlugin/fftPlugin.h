/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_FFT_PLUGIN_H
#define TRT_FFT_PLUGIN_H
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "common/cufftWrapper.h"
#include "common/plugin.h"

#include <array>
#include <compare>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

using pluginInternal::cufftHandle;

//! Deleter that destroys an owned cuFFT plan handle.
struct CufftHandleDeleter
{
    void operator()(cufftHandle* handle) const noexcept;
};
using CufftHandlePtr = std::unique_ptr<cufftHandle, CufftHandleDeleter>;

//! Key identifying a cached cuFFT plan. A plan depends only on the transformed
//! signal dimensions, the batch size, and the element type; the plugin's
//! direction/onesided/ndims are fixed per instance.
struct FFTPlanKey
{
    std::array<int64_t, 3> signalDims;
    int64_t batchSize;
    DataType dtype;

    auto operator<=>(FFTPlanKey const& other) const = default;
};

//! Cached cuFFT plan plus its queried workspace size.
struct FFTPlanContext
{
    CufftHandlePtr handle;
    size_t workspaceSize{0};
};

//! IPluginV3 wrapping cuFFT. Transforms the trailing mNdims dimensions of the
//! input, batched over the leading dimensions. Complex values use the interleaved
//! [..., 2] (real, imaginary) layout shared by ONNX DFT and torch.view_as_real.
//! cuFFT's unnormalized convention is used for both forward and inverse.
class FFTPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime
{
public:
    FFTPlugin() = delete;

    FFTPlugin(bool inverse, bool onesided, int32_t ndims);

    ~FFTPlugin() override = default;

    // IPluginV3 Methods
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;

    FFTPlugin* clone() noexcept override;
    // end IPluginV3 Methods

    // IPluginV3OneCore Methods
    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    char const* getPluginNamespace() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;
    // end IPluginV3OneCore Methods

    // IPluginV3OneBuild Methods
    int32_t getNbOutputs() const noexcept override;

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    // end IPluginV3OneBuild Methods

    // IPluginV3OneRuntime Methods
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;

    PluginFieldCollection const* getFieldsToSerialize() noexcept override;
    // end IPluginV3OneRuntime Methods

private:
    //! Derive the (signal dims, batch size) used to build a cuFFT plan from the
    //! concrete input/output descriptors for the current shape.
    void computePlanShape(PluginTensorDesc const& in, PluginTensorDesc const& out, std::array<int64_t, 3>& signalDims,
        int64_t& batchSize) const;

    //! Build or the cuFFT plan or fetch from cache for the given key and return the cached
    //! entry. The cache is never evicted, so the returned reference stays valid.
    FFTPlanContext const& ensurePlan(FFTPlanKey const& key) const;

    bool mInverse;
    bool mOnesided;
    int32_t mNdims;

    // INT32 mirrors of the bool attributes, kept alive for getFieldsToSerialize.
    int32_t mInverseField{0};
    int32_t mOnesidedField{0};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
    std::string mNamespace;

    //! Guards \c mPlanCache, \c mCurrentPlan, and \c mCurrentWorkspaceSize.
    //! TensorRT may call \c getWorkspaceSize and \c onShapeChange concurrently.
    //! Entries in \c mPlanCache are never removed for the lifetime of this \c FFTPlugin
    //! instance; because \c std::map nodes have stable addresses, a \c cufftHandle*
    //! extracted from the map remains valid after the mutex is released.
    mutable std::mutex mCacheMutex;
    //! \see mCacheMutex
    mutable std::map<FFTPlanKey, FFTPlanContext> mPlanCache;
    //! cuFFT plan for the most recently configured shape. \see mCacheMutex
    mutable cufftHandle* mCurrentPlan{nullptr};
    //! Workspace size for \c mCurrentPlan. \see mCacheMutex
    mutable size_t mCurrentWorkspaceSize{0};

    // Device input 0 is the signal; output 0 is the transform result. For the
    // onesided inverse (C2R), an optional INT64 shape input at index 1 carries the
    // original signal length so odd N can be reconstructed.
    static constexpr int32_t kINPUT_TENSOR_IDX = 0;
    static constexpr int32_t kFFT_LENGTH_INPUT_IDX = 1;
    static constexpr int32_t kOUTPUT_TENSOR_IDX = 0;
};

class FFTPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    FFTPluginCreator();

    ~FFTPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    char const* getPluginNamespace() const noexcept override;

    void setPluginNamespace(char const* libNamespace) noexcept;

    IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_FFT_PLUGIN_H
