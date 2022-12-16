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

#include <cuda.h>
#if CUDA_VERSION >= 10010

#ifndef TRT_EMB_LAYER_NORM_PLUGIN_H
#define TRT_EMB_LAYER_NORM_PLUGIN_H

#include "NvInferPlugin.h"
#include "NvInferRuntime.h"

#include "common/bertCommon.h"
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
namespace bert
{

int32_t computeMaskIdx(cudaStream_t stream, int32_t const S, int32_t const B, int32_t const* mask, int* maskIdx);

template <typename T>
int32_t embSkipLayerNorm(cudaStream_t stream, int32_t ld, int32_t B, int32_t S, int32_t const* inputIds,
    int32_t const* token_ids, float const* beta, float const* gamma, T const* wordEmb, T const* posEmb, T const* tokEmb,
    int32_t const wordSize, int32_t const tokSize, T* output);

cudaError_t convertMask(uint32_t const S, uint32_t const B, uint32_t const warps_m, uint32_t const warps_n,
    uint32_t const warps_k, int32_t const* inputMaskSB, uint32_t* inputMaskX, cudaStream_t stream);

class EmbLayerNormPluginDynamic : public nvinfer1::IPluginV2DynamicExt
{
public:
    EmbLayerNormPluginDynamic(std::string const& name, nvinfer1::DataType const type, nvinfer1::DataType const mhaType,
        nvinfer1::Weights const& beta, nvinfer1::Weights const& gamma, nvinfer1::Weights const& word_emb,
        nvinfer1::Weights const& pos_emb, nvinfer1::Weights const& tok_emb, bool const useFullMask);

    EmbLayerNormPluginDynamic(std::string const& name, const void* data, size_t length);

    // It doesn't make sense to make EmbLayerNormPluginDynamic without arguments, so we
    // delete default constructor.
    EmbLayerNormPluginDynamic() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

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

private:
    std::string const mLayerName;
    std::string mNamespace;

    bert::cuda_unique_ptr<float> mGammaDev;
    bert::cuda_unique_ptr<float> mBetaDev;
    bert::cuda_unique_ptr<void> mWordEmbDev;
    bert::cuda_unique_ptr<void> mTokEmbDev;
    bert::cuda_unique_ptr<void> mPosEmbDev;
    size_t mLd; // leading dim = hidden size
    size_t mS;  // sequence length
    size_t mWordVocabSize;
    size_t mPosVocabSize;
    size_t mTokVocabSize;
    bert::WeightsWithOwnership mBeta;
    bert::WeightsWithOwnership mGamma;
    bert::WeightsWithOwnership mWordEmb;
    bert::WeightsWithOwnership mTokEmb;
    bert::WeightsWithOwnership mPosEmb;
    nvinfer1::DataType mType;
    bool mUseFullMask;
    nvinfer1::DataType mMhaType;
    int32_t mSM;

    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2::enqueue;
    using IPluginV2Ext::configurePlugin;
};

class EmbLayerNormPluginDynamicCreator : public nvinfer1::IPluginCreator
{
public:
    EmbLayerNormPluginDynamicCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};
} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_EMB_LAYER_NORM_PLUGIN_H

#endif // CUDA_VERSION >= 10010
