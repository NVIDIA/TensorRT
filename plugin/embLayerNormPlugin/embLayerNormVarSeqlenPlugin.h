/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
int32_t embSkipLayerNormVarSeqlenHFace(cudaStream_t stream, int32_t ld, int32_t B, int32_t S, const uint32_t* cuSeqlens,
    const int32_t* inputIds, const int32_t* token_ids, const T* beta, const T* gamma, const T* wordEmb, const T* posEmb,
    const T* tokEmb, T* output);

template <typename T>
int32_t embSkipLayerNormHFace(cudaStream_t stream, int32_t ld, int32_t B, int32_t S, const int32_t* inputIds,
    const int32_t* tokenIds, const int32_t* cuSeqlens, const float* beta, const float* gamma, const T* wordEmb,
    const T* posEmb, const T* tokEmb, T* output);

template <typename T>
int32_t embSkipLayerNormVarSeqlenMTron(cudaStream_t stream, int32_t ld, int32_t B, int32_t S, const uint32_t* cuSeqlens,
    const int32_t* inputIds, const int32_t* token_ids, const T* beta, const T* gamma, const T* wordEmb, const T* posEmb,
    const T* tokEmb, T* output, T* skip);

template <typename T>
int32_t embSkipLayerNormMTron(cudaStream_t stream, int32_t ld, int32_t B, int32_t S, const int32_t* inputIds,
    const int32_t* tokenIds, const int32_t* cuSeqlens, const float* beta, const float* gamma, const T* wordEmb,
    const T* posEmb, const T* tokEmb, T* output, T* skip);

void cuSeqlensToPackedMask(const uint32_t S, const uint32_t B, const uint32_t warps_m, const uint32_t warps_n,
    const uint32_t warps_k, const int32_t* cuSeqlens, uint32_t* inputMaskX, cudaStream_t stream);

class EmbLayerNormVarSeqlenPluginBase : public nvinfer1::IPluginV2DynamicExt
{
public:
    EmbLayerNormVarSeqlenPluginBase(const std::string& name, const nvinfer1::DataType type,
        const nvinfer1::Weights& beta, const nvinfer1::Weights& gamma, const nvinfer1::Weights& word_emb,
        const nvinfer1::Weights& pos_emb, const nvinfer1::Weights& tok_emb);

    EmbLayerNormVarSeqlenPluginBase(const std::string& name, const void* data, size_t length);

    // It doesn't make sense to make EmbLayerNormVarSeqlenPlugin without arguments, so we
    // delete default constructor.
    EmbLayerNormVarSeqlenPluginBase() = delete;

    // IPluginV2DynamicExt Methods
    bool supportsFormatCombination(
        int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const
        noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    const char* getPluginNamespace() const noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;

protected:
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
};

class EmbLayerNormVarSeqlenPluginHFace : public EmbLayerNormVarSeqlenPluginBase
{
public:
    EmbLayerNormVarSeqlenPluginHFace(const std::string& name, const nvinfer1::DataType type, const nvinfer1::Weights& beta,
        const nvinfer1::Weights& gamma, const nvinfer1::Weights& word_emb, const nvinfer1::Weights& pos_emb,
        const nvinfer1::Weights& tok_emb);

    EmbLayerNormVarSeqlenPluginHFace(const std::string& name, const void* data, size_t length);

    // It doesn't make sense to make EmbLayerNormVarSeqlenPlugin without arguments, so we
    // delete default constructor.
    EmbLayerNormVarSeqlenPluginHFace() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override;
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2 Methods
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;
    const char* getPluginVersion() const noexcept override;
};

class EmbLayerNormVarSeqlenPluginMTron : public EmbLayerNormVarSeqlenPluginBase
{
public:
    EmbLayerNormVarSeqlenPluginMTron(const std::string& name, const nvinfer1::DataType type, const nvinfer1::Weights& beta,
        const nvinfer1::Weights& gamma, const nvinfer1::Weights& word_emb, const nvinfer1::Weights& pos_emb,
        const nvinfer1::Weights& tok_emb);

    EmbLayerNormVarSeqlenPluginMTron(const std::string& name, const void* data, size_t length);

    // It doesn't make sense to make EmbLayerNormVarSeqlenPlugin without arguments, so we
    // delete default constructor.
    EmbLayerNormVarSeqlenPluginMTron() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override;
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2 Methods
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;
    const char* getPluginVersion() const noexcept override;
};

class EmbLayerNormVarSeqlenPluginBaseCreator : public nvinfer1::IPluginCreator
{
public:
    EmbLayerNormVarSeqlenPluginBaseCreator();

    const char* getPluginName() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

protected:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

class EmbLayerNormVarSeqlenPluginHFaceCreator : public EmbLayerNormVarSeqlenPluginBaseCreator
{
public:
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    const char* getPluginVersion() const noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;
};

class EmbLayerNormVarSeqlenPluginMTronCreator : public EmbLayerNormVarSeqlenPluginBaseCreator
{
public:
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    const char* getPluginVersion() const noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;
};

} // namespace bert
#endif // TRT_EMB_LAYER_NORM_PLUGIN_H
