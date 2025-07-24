/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef TRT_INSTANCE_NORMALIZATION_PLUGIN_H
#define TRT_INSTANCE_NORMALIZATION_PLUGIN_H
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "common/plugin.h"
#include "common/serialize.hpp"
#include "instanceNormalizationPlugin/instanceNormFwd.h"
#include <cuda_fp16.h>
#include <iostream>
#include <string>
#include <vector>

typedef uint16_t half_type;

namespace
{
constexpr char const* gInstancePluginFullNameV3{"InstanceNormalization_TRT, version:3"};
} // namespace

namespace nvinfer1
{
namespace plugin
{
class InstanceNormalizationV3Plugin : public IPluginV3,
                                      public IPluginV3OneCore,
                                      public IPluginV3OneBuild,
                                      public IPluginV3OneRuntime
{

public:
    InstanceNormalizationV3Plugin(float epsilon, nvinfer1::Weights const& scale, nvinfer1::Weights const& bias,
        int32_t relu = 0, float alpha = 0.F);
    InstanceNormalizationV3Plugin(float epsilon, std::vector<float> const& scale, std::vector<float> const& bias,
        int32_t relu = 0, float alpha = 0.F);
    InstanceNormalizationV3Plugin(void const* serialData, size_t serialLength);

    InstanceNormalizationV3Plugin() = delete;

    InstanceNormalizationV3Plugin(InstanceNormalizationV3Plugin const&) = default;

    ~InstanceNormalizationV3Plugin() override;

    int32_t getNbOutputs() const noexcept override;

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;

    InstanceNormalizationV3Plugin* clone() noexcept override;

    char const* getPluginName() const noexcept override;

    char const* getPluginNamespace() const noexcept override;

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // DynamicExt plugin supportsFormat update.
    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    char const* getPluginVersion() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    int32_t initializeContext();

protected:
    void exitContext();

private:
    float mEpsilon{};
    float mAlpha{};
    int32_t mRelu{};
    int32_t mNchan{};
    std::vector<float> mHostScale;
    std::vector<float> mHostBias;
    float* mDeviceScale{nullptr};
    float* mDeviceBias{nullptr};
    nvinfer1::pluginInternal::cudnnHandle_t mCudnnHandle{nullptr};
    nvinfer1::pluginInternal::CudnnWrapper& mCudnnWrapper
        = nvinfer1::pluginInternal::getCudnnWrapper(gInstancePluginFullNameV3);

    nvinfer1::pluginInternal::cudnnTensorDescriptor_t mXDescriptor{nullptr};
    nvinfer1::pluginInternal::cudnnTensorDescriptor_t mYDescriptor{nullptr};
    nvinfer1::pluginInternal::cudnnTensorDescriptor_t mBDescriptor{nullptr};
    std::string mPluginNamespace;
    bool mInitialized{false};
    int32_t mCudaDriverVersion{-1};
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;

    // NDHWC implementation
    instance_norm_impl::InstanceNormFwdContext mContext;
};

class InstanceNormalizationV3PluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    InstanceNormalizationV3PluginCreator();

    ~InstanceNormalizationV3PluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;

    void setPluginNamespace(char const* libNamespace) noexcept;

    char const* getPluginNamespace() const noexcept override;

private:
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_INSTANCE_NORMALIZATION_PLUGIN_H
