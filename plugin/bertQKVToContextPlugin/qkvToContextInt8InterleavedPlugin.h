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

#ifndef TRT_QKV_TO_CONTEXT_INTERLEAVED_PLUGIN_H
#define TRT_QKV_TO_CONTEXT_INTERLEAVED_PLUGIN_H

#include "NvInferPlugin.h"
#include "fused_multihead_attention_v2/fused_multihead_attention_v2.h"
#include <cuda.h>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
namespace bert
{
static constexpr int32_t kSM_XAVIER = 72;
static constexpr int32_t kSM_TURING = 75;
static constexpr int32_t kSM_AMPERE_100 = 80;
static constexpr int32_t kSM_AMPERE_10X = 86;
static constexpr int32_t kSM_AMPERE_10B = 87;
static constexpr int32_t kSM_ADA_10X = 89;
static constexpr int32_t kSM_HOPPER_100 = 90;
static constexpr int32_t kSM_BLACKWELL_100 = 100;
static constexpr int32_t kSM_BLACKWELL_120 = 120;

class QKVToContextInterleavedPlugin : public IPluginV3,
                                      public IPluginV3OneCore,
                                      public IPluginV3OneBuild,
                                      public IPluginV3OneRuntime
{
public:
    QKVToContextInterleavedPlugin(std::string const& name, int32_t hiddenSize, int32_t numHeads, float dqProbs,
        bool useInt8ScaleMax, bool useExplicitInt8, float qkvScale, float ctxScale);

    // It doesn't make sense to make QKVToContextInterleavedPlugin without arguments, so we
    // delete default constructor.
    QKVToContextInterleavedPlugin() = delete;

    ~QKVToContextInterleavedPlugin() override;

    // IPluginV3 Methods
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;
    // end of IPluginV3 Methods

    // IPluginV3OneCore Methods
    char const* getPluginName() const noexcept override;

    char const* getPluginNamespace() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

    char const* getPluginVersion() const noexcept override;
    // end of IPluginV3OneCore Methods

    // IPluginV3Build Methods
    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getNbOutputs() const noexcept override;

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;
    // end IPluginV3Build Methods

    // IPluginV3Runtime Methods
    IPluginV3* clone() noexcept override;

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;

    PluginFieldCollection const* getFieldsToSerialize() noexcept override;
    // end IPluginV3Runtime Methods

protected:
private:
    std::string const& mLayerName;
    std::string mNamespace;

    int32_t mSM{};
    int32_t mHeadSize{};
    int32_t mHiddenSize{};
    int32_t mNumHeads{};
    int32_t mUseInt8ScaleMax{1};
    int32_t mUseExplicitInt8{};

    FusedMultiHeadAttentionXMMAKernelV2 const* mXmmaKernel;

    float mDqProbs{};
    float mQkvScale{};
    float mCtxScale{};

    // IPluginV3 serialization related
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class QKVToContextInterleavedPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    QKVToContextInterleavedPluginCreator();
    ~QKVToContextInterleavedPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;

    void setPluginNamespace(char const* libNamespace) noexcept;

    char const* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_QKV_TO_CONTEXT_INTERLEAVED_PLUGIN_H
