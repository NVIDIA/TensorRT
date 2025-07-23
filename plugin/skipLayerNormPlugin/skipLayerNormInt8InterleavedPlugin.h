/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef TRT_SKIP_LAYER_NORM_INTERLEAVED_PLUGIN_H
#define TRT_SKIP_LAYER_NORM_INTERLEAVED_PLUGIN_H
#include "NvInferPlugin.h"
#include <cuda.h>

#include "common/bertCommon.h"
#include <memory>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
namespace bert
{

int32_t launch_small_hface(cudaStream_t stream, int32_t const ld, int32_t const total, int8_t const* input,
    int8_t const* skip, half const* beta, half const* gamma, int8_t* output, float const dqScaleIn,
    float const dqScaleSkip, float const qScale);

int32_t launch_large_hface(cudaStream_t stream, int32_t const ld, int32_t const total, int8_t const* input,
    int8_t const* skip, half const* beta, half const* gamma, int8_t* output, float const dqScaleIn,
    float const dqScaleSkip, float const qScale);

int32_t launch_small_mtron(cudaStream_t stream, int32_t const ld, int32_t const total, int8_t const* input,
    int8_t const* skip, half const* beta, half const* gamma, int8_t* output, int8_t* preln, float const dqScaleIn,
    float const dqScaleSkip, float const qScale, float const qSkipScale);

int32_t launch_large_mtron(cudaStream_t stream, int32_t const ld, int32_t const total, int8_t const* input,
    int8_t const* skip, half const* beta, half const* gamma, int8_t* output, int8_t* preln, float const dqScaleIn,
    float const dqScaleSkip, float const qScale, float const qSkipScale);

class SkipLayerNormInterleavedPluginBase : public IPluginV3,
                                           public IPluginV3OneCore,
                                           public IPluginV3OneBuild,
                                           public IPluginV3OneRuntime
{
public:
    SkipLayerNormInterleavedPluginBase(
        std::string const& name, nvinfer1::Weights const& beta, nvinfer1::Weights const& gamma);

    // It doesn't make sense to make SkipLayerNormInterleavedPlugin without
    // arguments, so we delete default constructor.
    SkipLayerNormInterleavedPluginBase() = delete;

    ~SkipLayerNormInterleavedPluginBase() override;

    // IPluginV3 Methods
    // NOTE: since this is itself is an abstract class, the rest of virtual methods defined in its children classes
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;
    // end of IPluginV3 Methods

    // IPluginV3OneCore Methods
    char const* getPluginName() const noexcept override;

    char const* getPluginNamespace() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;
    // end of IPluginV3OneCore Methods

    // IPluginV3Build Methods
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
    // end IPluginV3Build Methods

    // IPluginV3Runtime Methods
    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;

    PluginFieldCollection const* getFieldsToSerialize() noexcept override;
    // end IPluginV3Runtime Methods

protected:
    // metadata fields
    std::string const& mLayerName;
    std::string mNamespace;
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;

    // members that participate in ser/deserialization
    bert::WeightsWithOwnership mGamma;
    bert::WeightsWithOwnership mBeta;

    // device-side
    bert::cuda_unique_ptr<void> mGammaDev;
    bert::cuda_unique_ptr<void> mBetaDev;

    // derived members
    size_t mLd{}; // leading dim
    size_t mParamWordsize{};
    bool mParamsOnDevice{};
};

class SkipLayerNormInterleavedPluginHFace : public SkipLayerNormInterleavedPluginBase
{
public:
    SkipLayerNormInterleavedPluginHFace(
        std::string const& name, nvinfer1::Weights const& beta, nvinfer1::Weights const& gamma);

    // It doesn't make sense to make SkipLayerNormInterleavedPlugin without
    // arguments, so we delete default constructor.
    SkipLayerNormInterleavedPluginHFace() = delete;

    ~SkipLayerNormInterleavedPluginHFace() override;

    // IPluginV3Runtime overrides
    IPluginV3* clone() noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV3OneCore override
    char const* getPluginVersion() const noexcept override;

    // IPluginV3OneBuild override
    int32_t getNbOutputs() const noexcept override;
};

class SkipLayerNormInterleavedPluginMTron : public SkipLayerNormInterleavedPluginBase
{
public:
    SkipLayerNormInterleavedPluginMTron(
        std::string const& name, nvinfer1::Weights const& beta, nvinfer1::Weights const& gamma);

    // It doesn't make sense to make SkipLayerNormInterleavedPlugin without
    // arguments, so we delete default constructor.
    SkipLayerNormInterleavedPluginMTron() = delete;

    ~SkipLayerNormInterleavedPluginMTron() override;

    // IPluginV3Runtime overrides
    IPluginV3* clone() noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV3OneCore override
    char const* getPluginVersion() const noexcept override;

    // IPluginV3OneBuild override
    int32_t getNbOutputs() const noexcept override;
};

class SkipLayerNormInterleavedPluginBaseCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    SkipLayerNormInterleavedPluginBaseCreator();
    ~SkipLayerNormInterleavedPluginBaseCreator() override = default;

    char const* getPluginName() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

    char const* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

class SkipLayerNormInterleavedPluginHFaceCreator : public SkipLayerNormInterleavedPluginBaseCreator
{
public:
    SkipLayerNormInterleavedPluginHFaceCreator();

    ~SkipLayerNormInterleavedPluginHFaceCreator() override = default;

    char const* getPluginVersion() const noexcept override;

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;
};

class SkipLayerNormInterleavedPluginMTronCreator : public SkipLayerNormInterleavedPluginBaseCreator
{
public:
    SkipLayerNormInterleavedPluginMTronCreator();

    ~SkipLayerNormInterleavedPluginMTronCreator() override = default;

    char const* getPluginVersion() const noexcept override;

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;
};

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_SKIP_LAYER_NORM_INTERLEAVED_PLUGIN_H
