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
#ifndef TRT_EMB_LAYER_NORM_VARSEQ_PLUGIN_H
#define TRT_EMB_LAYER_NORM_VARSEQ_PLUGIN_H

#include <cuda.h>

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

template <typename T>
int32_t embSkipLayerNormHFace(cudaStream_t stream, int32_t ld, int32_t B, int32_t S, int32_t const* inputIds,
    int32_t const* tokenIds, int32_t const* cuSeqlens, float const* beta, float const* gamma, T const* wordEmb,
    T const* posEmb, T const* tokEmb, int32_t const wordSize, int32_t const tokSize, T* output);

template <typename T>
int32_t embSkipLayerNormMTron(cudaStream_t stream, int32_t ld, int32_t B, int32_t S, int32_t const* inputIds,
    int32_t const* tokenIds, int32_t const* cuSeqlens, float const* beta, float const* gamma, T const* wordEmb,
    T const* posEmb, T const* tokEmb, int32_t const wordSize, int32_t const tokSize, T* output, T* skip);

class EmbLayerNormVarSeqlenPluginBase : public IPluginV3,
                                        public IPluginV3OneCore,
                                        public IPluginV3OneBuild,
                                        public IPluginV3OneRuntime
{
public:
    EmbLayerNormVarSeqlenPluginBase(std::string const& name, DataType type, Weights const& beta, Weights const& gamma,
        Weights const& word_emb, Weights const& pos_emb, Weights const& tok_emb, DataType maskType);

    // It doesn't make sense to make EmbLayerNormVarSeqlenPlugin without arguments, so we
    // delete default constructor.
    EmbLayerNormVarSeqlenPluginBase() = delete;

    ~EmbLayerNormVarSeqlenPluginBase() override;

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

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getNbOutputs() const noexcept override;
    // end IPluginV3Build Methods

    // IPluginV3Runtime Methods

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;

    PluginFieldCollection const* getFieldsToSerialize() noexcept override;
    // end IPluginV3Runtime Methods

protected:
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
    size_t mWordVocabSize;
    size_t mPosVocabSize;
    size_t mTokVocabSize;

    // members that partcipate in ser/deserialization
    bert::WeightsWithOwnership mBeta;
    bert::WeightsWithOwnership mGamma;
    bert::WeightsWithOwnership mWordEmb;
    bert::WeightsWithOwnership mTokEmb;
    bert::WeightsWithOwnership mPosEmb;
    DataType mType{};
    DataType mMaskType{};

    // IPluginV3 serialization related
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class EmbLayerNormVarSeqlenPluginHFace : public EmbLayerNormVarSeqlenPluginBase
{
public:
    EmbLayerNormVarSeqlenPluginHFace(std::string const& name, nvinfer1::DataType const type,
        nvinfer1::Weights const& beta, nvinfer1::Weights const& gamma, nvinfer1::Weights const& word_emb,
        nvinfer1::Weights const& pos_emb, nvinfer1::Weights const& tok_emb);

    // It doesn't make sense to make EmbLayerNormVarSeqlenPlugin without arguments, so we
    // delete default constructor.
    EmbLayerNormVarSeqlenPluginHFace() = delete;

    ~EmbLayerNormVarSeqlenPluginHFace() override;

    // IPluginV3Runtime overrides
    IPluginV3* clone() noexcept override;

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV3OneCore override
    char const* getPluginVersion() const noexcept override;

    // IPluginV3OneBuild override
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;
};

class EmbLayerNormVarSeqlenPluginMTron : public EmbLayerNormVarSeqlenPluginBase
{
public:
    EmbLayerNormVarSeqlenPluginMTron(std::string const& name, nvinfer1::DataType const type,
        nvinfer1::Weights const& beta, nvinfer1::Weights const& gamma, nvinfer1::Weights const& word_emb,
        nvinfer1::Weights const& pos_emb, nvinfer1::Weights const& tok_emb);

    // It doesn't make sense to make EmbLayerNormVarSeqlenPlugin without arguments, so we
    // delete default constructor.
    EmbLayerNormVarSeqlenPluginMTron() = delete;

    ~EmbLayerNormVarSeqlenPluginMTron() override;

    // IPluginV3Runtime overrides
    IPluginV3* clone() noexcept override;

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV3OneCore override
    char const* getPluginVersion() const noexcept override;

    // IPluginV3OneBuild override
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;
};

class EmbLayerNormVarSeqlenPluginBaseCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    EmbLayerNormVarSeqlenPluginBaseCreator();

    char const* getPluginName() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    void setPluginNamespace(char const* libNamespace) noexcept;

    char const* getPluginNamespace() const noexcept override;

protected:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

class EmbLayerNormVarSeqlenPluginHFaceCreator : public EmbLayerNormVarSeqlenPluginBaseCreator
{
public:
    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;
    char const* getPluginVersion() const noexcept override;
};

class EmbLayerNormVarSeqlenPluginMTronCreator : public EmbLayerNormVarSeqlenPluginBaseCreator
{
public:
    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;
    char const* getPluginVersion() const noexcept override;
};

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_EMB_LAYER_NORM_VARSEQ_PLUGIN_H
