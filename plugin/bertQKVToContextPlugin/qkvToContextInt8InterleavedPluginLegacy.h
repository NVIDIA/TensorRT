/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_QKV_TO_CONTEXT_INTERLEAVED_PLUGIN_LEGACY_H
#define TRT_QKV_TO_CONTEXT_INTERLEAVED_PLUGIN_LEGACY_H

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

class QKVToContextInterleavedPluginLegacy : public nvinfer1::IPluginV2DynamicExt
{
public:
    QKVToContextInterleavedPluginLegacy(std::string const& name, int32_t hiddenSize, int32_t numHeads, float dqProbs,
        bool useInt8ScaleMax, bool useExplicitInt8, float qkvScale, float ctxScale);

    QKVToContextInterleavedPluginLegacy(std::string const& name, void const* data, size_t length);

    // It doesn't make sense to make QKVToContextInterleavedPluginLegacy without arguments, so we
    // delete default constructor.
    QKVToContextInterleavedPluginLegacy() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    // IPluginV2 Methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

protected:
    void createMHARunner() noexcept;

private:
    std::string const& mLayerName;
    std::string mNamespace;

    int32_t mS{};
    int32_t mB{};
    int32_t mSM{};
    int32_t mHeadSize{};
    int32_t mHiddenSize{};
    int32_t mNumHeads{};

    FusedMultiHeadAttentionXMMAKernelV2 const* mXmmaKernel;

    float mDqProbs{};
    bool mUseInt8ScaleMax{true};

    bool mUseExplicitInt8{};
    float mQkvScale{};
    float mCtxScale{};
};

class QKVToContextInterleavedPluginLegacyCreator : public nvinfer1::IPluginCreator
{
public:
    QKVToContextInterleavedPluginLegacyCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_QKV_TO_CONTEXT_INTERLEAVED_PLUGIN_LEGACY_H
