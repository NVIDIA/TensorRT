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

int32_t computeMaskIdx(cudaStream_t stream, int32_t const S, int32_t const B, int32_t const* mask, int32_t* maskIdx);

template <typename T>
int32_t embSkipLayerNorm(cudaStream_t stream, int32_t ld, int32_t B, int32_t S, int32_t const* inputIds,
    int32_t const* token_ids, float const* beta, float const* gamma, T const* wordEmb, T const* posEmb, T const* tokEmb,
    int32_t const wordSize, int32_t const tokSize, T* output);

cudaError_t convertMask(uint32_t const S, uint32_t const B, uint32_t const warps_m, uint32_t const warps_n,
    uint32_t const warps_k, int32_t const* inputMaskSB, uint32_t* inputMaskX, cudaStream_t stream);

class EmbLayerNormPluginDynamic : public IPluginV3,
                                  public IPluginV3OneCore,
                                  public IPluginV3OneBuild,
                                  public IPluginV3OneRuntime
{
public:
    EmbLayerNormPluginDynamic(std::string const& name, nvinfer1::DataType const type, nvinfer1::DataType const mhaType,
        nvinfer1::Weights const& beta, nvinfer1::Weights const& gamma, nvinfer1::Weights const& word_emb,
        nvinfer1::Weights const& pos_emb, nvinfer1::Weights const& tok_emb, bool const useFullMask);

    // It doesn't make sense to make EmbLayerNormPluginDynamic without arguments, so we
    // delete default constructor.
    EmbLayerNormPluginDynamic() = delete;

    ~EmbLayerNormPluginDynamic() override;

    // IPluginV3 Methods
    // NOTE: since this is itself is an abstract class, the rest of virtual methods defined in its children classes
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

private:
    // metadata fields
    std::string const mLayerName;
    std::string mNamespace;

    // device-side
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

    // members that partcipate in ser/deserialization
    bert::WeightsWithOwnership mBeta;
    bert::WeightsWithOwnership mGamma;
    bert::WeightsWithOwnership mWordEmb;
    bert::WeightsWithOwnership mTokEmb;
    bert::WeightsWithOwnership mPosEmb;
    nvinfer1::DataType mType;
    int32_t mOutputFp16;
    int32_t mUseFullMask;
    nvinfer1::DataType mMhaType;
    int32_t mSM;

    // IPluginV3 serialization related
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class EmbLayerNormPluginDynamicCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    EmbLayerNormPluginDynamicCreator();
    ~EmbLayerNormPluginDynamicCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

    char const* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};
} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_EMB_LAYER_NORM_PLUGIN_H

#endif // CUDA_VERSION >= 10010
