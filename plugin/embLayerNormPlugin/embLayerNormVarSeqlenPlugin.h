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

#ifndef TRT_EMB_LAYER_NORM_PLUGIN_H
#define TRT_EMB_LAYER_NORM_PLUGIN_H

#include <cuda.h>

#include "NvInferPlugin.h"
#include "NvInferRuntime.h"

#include "bertCommon.h"
#include <string>
#include <vector>
namespace bert
{

template <typename T>
int embSkipLayerNormVarSeqlen(cudaStream_t stream, int ld, int B, int S, const uint32_t* cuSeqlens, const int* inputIds,
    const int* token_ids, const T* beta, const T* gamma, const T* wordEmb, const T* posEmb, const T* tokEmb, T* output);

template <typename T>
int embSkipLayerNorm2(cudaStream_t stream, int ld, int B, int S, const int* inputIds, const int* tokenIds,
    const int* cuSeqlens, const float* beta, const float* gamma, const T* wordEmb, const T* posEmb, const T* tokEmb,
    T* output);

void cuSeqlensToPackedMask(const uint32_t S, const uint32_t B, const uint32_t warps_m, const uint32_t warps_n,
    const uint32_t warps_k, const int* cuSeqlens, uint32_t* inputMaskX, cudaStream_t stream);

class EmbLayerNormVarSeqlenPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    EmbLayerNormVarSeqlenPlugin(const std::string& name, const nvinfer1::DataType type, const nvinfer1::Weights& beta,
        const nvinfer1::Weights& gamma, const nvinfer1::Weights& word_emb, const nvinfer1::Weights& pos_emb,
        const nvinfer1::Weights& tok_emb);

    EmbLayerNormVarSeqlenPlugin(const std::string& name, const void* data, size_t length);

    // It doesn't make sense to make EmbLayerNormVarSeqlenPlugin without arguments, so we
    // delete default constructor.
    EmbLayerNormVarSeqlenPlugin() = delete;

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

private:
    const std::string mLayerName;
    std::string mNamespace;

    bert::cuda_unique_ptr<float> mGammaDev;
    bert::cuda_unique_ptr<float> mBetaDev;
    bert::cuda_unique_ptr<void> mWordEmbDev;
    bert::cuda_unique_ptr<void> mTokEmbDev;
    bert::cuda_unique_ptr<void> mPosEmbDev;
    size_t mLd; // leading dim = hidden size
    size_t mWordVocabSize;
    size_t mPosVocabSize;
    size_t mTokVocabSize;
    bert::WeightsWithOwnership mBeta;
    bert::WeightsWithOwnership mGamma;
    bert::WeightsWithOwnership mWordEmb;
    bert::WeightsWithOwnership mTokEmb;
    bert::WeightsWithOwnership mPosEmb;
    nvinfer1::DataType mType;

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

class EmbLayerNormVarSeqlenPluginCreator : public nvinfer1::IPluginCreator
{
public:
    EmbLayerNormVarSeqlenPluginCreator();

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
#endif // TRT_EMB_LAYER_NORM_PLUGIN_H
