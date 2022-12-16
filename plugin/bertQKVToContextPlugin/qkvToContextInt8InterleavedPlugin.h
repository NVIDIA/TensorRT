/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cublas_v2.h"
#include "fused_multihead_attention_v2/include/fused_multihead_attention_v2.h"
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

class QKVToContextInterleavedPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    QKVToContextInterleavedPlugin(std::string const& name, int32_t const hiddenSize, int32_t const numHeads,
        float const dqProbs, bool const useInt8ScaleMax);

    QKVToContextInterleavedPlugin(std::string const& name, void const* data, size_t length);

    // It doesn't make sense to make QKVToContextInterleavedPlugin without arguments, so we
    // delete default constructor.
    QKVToContextInterleavedPlugin() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
        noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

protected:
    void createMHARunner() noexcept;
    int getSMVersion() const noexcept;

private:
    std::string const& mLayerName;
    std::string mNamespace;

    int mS;
    int mB;
    int mSM;
    int mHeadSize;
    int mHiddenSize;
    int mNumHeads;

    const FusedMultiHeadAttentionXMMAKernelV2* mXmmaKernel;

    float mDqProbs;
    bool mUseInt8ScaleMax{true};
};

class QKVToContextInterleavedPluginCreator : public nvinfer1::IPluginCreator
{
public:
    QKVToContextInterleavedPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_QKV_TO_CONTEXT_INTERLEAVED_PLUGIN_H
