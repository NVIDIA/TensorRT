/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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
#include "fused_multihead_attention_v2.h"
#include <cuda.h>
#include <string>
#include <vector>

namespace bert
{
static constexpr int32_t kSM_XAVIER = 72;
static constexpr int32_t kSM_TURING = 75;
static constexpr int32_t kSM_AMPERE = 80;

class QKVToContextInterleavedPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    QKVToContextInterleavedPlugin(
        const std::string name, const int hiddenSize, const int numHeads, const float dqProbs);

    QKVToContextInterleavedPlugin(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make QKVToContextInterleavedPlugin without arguments, so we
    // delete default constructor.
    QKVToContextInterleavedPlugin() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const override;
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    // IPluginV2 Methods
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    int getNbOutputs() const override;
    int initialize() override;
    void terminate() override;
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void destroy() override;
    void setPluginNamespace(const char* pluginNamespace) override;
    const char* getPluginNamespace() const override;

protected:
    void createMHARunner();
    int getSMVersion() const;

private:
    const std::string mLayerName;
    std::string mNamespace;

    int mS;
    int mB;
    int mSM;
    int mHeadSize;
    int mHiddenSize;
    int mNumHeads;

    const FusedMultiHeadAttentionXMMAKernelV2* mXmmaKernel;

    float mDqProbs;

protected:
    // To prevent compiler warnings.
    using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::configurePlugin;
    using nvinfer1::IPluginV2DynamicExt::enqueue;
    using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
    using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
    using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::supportsFormat;
};

class QKVToContextInterleavedPluginCreator : public nvinfer1::IPluginCreator
{
public:
    QKVToContextInterleavedPluginCreator();

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const nvinfer1::PluginFieldCollection* getFieldNames() override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace bert
#endif // TRT_QKV_TO_CONTEXT_INTERLEAVED_PLUGIN_H
