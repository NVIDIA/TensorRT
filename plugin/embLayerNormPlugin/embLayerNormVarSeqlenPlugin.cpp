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

#include <cstring>
#include <cuda.h>
#include <set>
#include <vector>

#include "NvInfer.h"
#include "common/serialize.hpp"
#include "embLayerNormVarSeqlenPlugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace nvinfer1::plugin::bert;

namespace
{
constexpr char const* kEMB_LAYER_NORM_VAR_SEQLEN_VERSION_HFACE{"4"};
constexpr char const* kEMB_LAYER_NORM_VAR_SEQLEN_VERSION_MTRON{"5"};
constexpr char const* kEMB_LAYER_NORM_VAR_SEQLEN_NAME{"CustomEmbLayerNormPluginDynamic"};

void checkConfigurationInputs(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    // Validate input arguments
    PLUGIN_ASSERT(nbInputs == 4);
    PLUGIN_ASSERT(nbOutputs == 2);

    PLUGIN_ASSERT(inputs[0].dims.nbDims == 1);
    PLUGIN_ASSERT(inputs[1].dims.nbDims == 1);

    PLUGIN_ASSERT(inputs[1].dims.d[0] == inputs[0].dims.d[0]);

    PLUGIN_ASSERT(inputs[2].dims.nbDims == 1);

    PLUGIN_ASSERT(outputs[0].dims.nbDims == 4);
    PLUGIN_ASSERT(static_cast<size_t>(outputs[0].dims.d[0]) == static_cast<size_t>(inputs[0].dims.d[0]));
    PLUGIN_ASSERT(outputs[0].dims.d[2] == 1);
    PLUGIN_ASSERT(outputs[0].dims.d[3] == 1);

    PLUGIN_ASSERT(inputs[0].type == DataType::kINT32);
    PLUGIN_ASSERT(inputs[1].type == DataType::kINT32);
    PLUGIN_ASSERT(inputs[2].type == DataType::kINT32);
}

bool initializeFields(char const* name, PluginFieldCollection const* fc, Weights& beta, Weights& gamma,
    Weights& word_emb, Weights& pos_emb, Weights& tok_emb)
{
    bool output_fp16 = false;
    std::set<std::string> const requiredAttributes{
        "bert_embeddings_layernorm_beta",
        "bert_embeddings_layernorm_gamma",
        "bert_embeddings_word_embeddings",
        "bert_embeddings_token_type_embeddings",
        "bert_embeddings_position_embeddings",
    };
    plugin::validateRequiredAttributesExist(requiredAttributes, fc);

    for (int32_t i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);
        if (field_name.compare("bert_embeddings_layernorm_beta") == 0)
        {
            BERT_DEBUG_MSG("Building bert_embeddings_layernorm_beta...");
            beta.values = fc->fields[i].data;
            beta.count = fc->fields[i].length;
            beta.type = fieldTypeToDataType(fc->fields[i].type);
        }

        else if (field_name.compare("bert_embeddings_layernorm_gamma") == 0)
        {
            BERT_DEBUG_MSG("Building bert_embeddings_layernorm_gamma...");
            gamma.values = fc->fields[i].data;
            gamma.count = fc->fields[i].length;
            gamma.type = fieldTypeToDataType(fc->fields[i].type);
        }

        else if (field_name.compare("bert_embeddings_word_embeddings") == 0)
        {
            BERT_DEBUG_MSG("Building bert_embeddings_word_embeddings...");
            word_emb.values = fc->fields[i].data;
            word_emb.count = fc->fields[i].length;
            word_emb.type = fieldTypeToDataType(fc->fields[i].type);
        }

        else if (field_name.compare("bert_embeddings_token_type_embeddings") == 0)
        {
            BERT_DEBUG_MSG("Building bert_embeddings_token_type_embeddings...");
            tok_emb.values = fc->fields[i].data;
            tok_emb.count = fc->fields[i].length;
            tok_emb.type = fieldTypeToDataType(fc->fields[i].type);
        }

        else if (field_name.compare("bert_embeddings_position_embeddings") == 0)
        {
            BERT_DEBUG_MSG("Building bert_embeddings_position_embeddings...");
            pos_emb.values = fc->fields[i].data;
            pos_emb.count = fc->fields[i].length;
            pos_emb.type = fieldTypeToDataType(fc->fields[i].type);
        }
        else if (field_name.compare("output_fp16") == 0)
        {
            BERT_DEBUG_MSG("Building output_fp16...");
            PLUGIN_VALIDATE(fc->fields[i].type == PluginFieldType::kINT32);
            output_fp16 = static_cast<int32_t const*>(fc->fields[i].data)[0] != 0;
        }
    }
    return output_fp16;
}

} // namespace

REGISTER_TENSORRT_PLUGIN(EmbLayerNormVarSeqlenPluginHFaceCreator);
REGISTER_TENSORRT_PLUGIN(EmbLayerNormVarSeqlenPluginMTronCreator);

EmbLayerNormVarSeqlenPluginBase::EmbLayerNormVarSeqlenPluginBase(std::string const& name, DataType type,
    Weights const& beta, Weights const& gamma, Weights const& wordEmb, Weights const& posEmb, Weights const& tokEmb,
    DataType maskType)
    : mLayerName(name)
    , mLd(beta.count)
    , mType(type)
    , mMaskType(maskType)
{
    // Assuming Weights.count is the number of elements and not bytes
    PLUGIN_VALIDATE(beta.count == gamma.count);
    PLUGIN_VALIDATE(mLd > 0U);
    PLUGIN_VALIDATE(wordEmb.count % mLd == 0);
    PLUGIN_VALIDATE(posEmb.count % mLd == 0);
    PLUGIN_VALIDATE(tokEmb.count % mLd == 0);
    mWordVocabSize = wordEmb.count / mLd;
    mPosVocabSize = posEmb.count / mLd;
    mTokVocabSize = tokEmb.count / mLd;

    mBeta.convertAndCopy(beta, nvinfer1::DataType::kFLOAT);
    mGamma.convertAndCopy(gamma, nvinfer1::DataType::kFLOAT);

    mWordEmb.convertAndCopy(wordEmb, mType);
    mTokEmb.convertAndCopy(tokEmb, mType);
    mPosEmb.convertAndCopy(posEmb, mType);

    copyToDevice(mGamma, sizeof(float) * mGamma.count, mGammaDev);
    copyToDevice(mBeta, sizeof(float) * mBeta.count, mBetaDev);

    copyToDevice(mWordEmb, getWeightsSize(mWordEmb, mType), mWordEmbDev);
    copyToDevice(mPosEmb, getWeightsSize(mPosEmb, mType), mPosEmbDev);
    copyToDevice(mTokEmb, getWeightsSize(mTokEmb, mType), mTokEmbDev);
}

EmbLayerNormVarSeqlenPluginHFace::EmbLayerNormVarSeqlenPluginHFace(std::string const& name, DataType const type,
    Weights const& beta, Weights const& gamma, Weights const& wordEmb, Weights const& posEmb, Weights const& tokEmb)
    : EmbLayerNormVarSeqlenPluginBase(name, type, beta, gamma, wordEmb, posEmb, tokEmb, DataType::kINT32)
{
    BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginHFace creation");
}

EmbLayerNormVarSeqlenPluginMTron::EmbLayerNormVarSeqlenPluginMTron(std::string const& name, DataType const type,
    Weights const& beta, Weights const& gamma, Weights const& wordEmb, Weights const& posEmb, Weights const& tokEmb)
    : EmbLayerNormVarSeqlenPluginBase(name, type, beta, gamma, wordEmb, posEmb, tokEmb, type)
{
    BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginMTron creation");
}

EmbLayerNormVarSeqlenPluginBase::~EmbLayerNormVarSeqlenPluginBase()
{
    try
    {
        // This gets called when the network containing plugin is destroyed
        mGammaDev.reset(nullptr);
        mBetaDev.reset(nullptr);
        mWordEmbDev.reset(nullptr);
        mPosEmbDev.reset(nullptr);
        mTokEmbDev.reset(nullptr);
        // delete this; (TRT will delete this plugin object)
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

EmbLayerNormVarSeqlenPluginHFace::~EmbLayerNormVarSeqlenPluginHFace()
{
    BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginHFace destruction");
}

EmbLayerNormVarSeqlenPluginMTron::~EmbLayerNormVarSeqlenPluginMTron()
{
    BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginMTron destruction");
}

//////
// IPluginV3 method definitions:
// - getCapabilityInterface() (Base)
// - clone() (HFace, MTron)
//////
IPluginCapability* EmbLayerNormVarSeqlenPluginBase::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV3* EmbLayerNormVarSeqlenPluginHFace::clone() noexcept
{
    try
    {
        BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginHFace clone");

        auto p = new EmbLayerNormVarSeqlenPluginHFace(mLayerName, mType, mBeta, mGamma, mWordEmb, mPosEmb, mTokEmb);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV3* EmbLayerNormVarSeqlenPluginMTron::clone() noexcept
{
    try
    {
        BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginMTron clone");

        auto p = new EmbLayerNormVarSeqlenPluginMTron(mLayerName, mType, mBeta, mGamma, mWordEmb, mPosEmb, mTokEmb);
        p->setPluginNamespace(mNamespace.c_str());

        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

// End IPluginV3 method definitions

//////
// IPluginV3OneRuntime method definitions:
// - getFieldsToSerialize() (Base)
// - onShapeChange() (Base)
// - attachToContext() (Base)
// - enqueue() (HFace, MTron)
/////

PluginFieldCollection const* EmbLayerNormVarSeqlenPluginBase::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    bool output_fp16 = mType == DataType::kHALF;
    mDataToSerialize.emplace_back("output_fp16", &output_fp16, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("bert_embeddings_layernorm_beta", static_cast<float const*>(mBeta.values),
        PluginFieldType::kFLOAT32, mBeta.count);
    mDataToSerialize.emplace_back("bert_embeddings_layernorm_gamma", static_cast<float const*>(mGamma.values),
        PluginFieldType::kFLOAT32, mGamma.count);
    if (output_fp16)
    {
        mDataToSerialize.emplace_back("bert_embeddings_word_embeddings", static_cast<half const*>(mWordEmb.values),
            PluginFieldType::kFLOAT16, mWordEmb.count);
        mDataToSerialize.emplace_back("bert_embeddings_token_type_embeddings", static_cast<half const*>(mTokEmb.values),
            PluginFieldType::kFLOAT16, mTokEmb.count);
        mDataToSerialize.emplace_back("bert_embeddings_position_embeddings", static_cast<half const*>(mPosEmb.values),
            PluginFieldType::kFLOAT16, mPosEmb.count);
    }
    else
    {
        mDataToSerialize.emplace_back("bert_embeddings_word_embeddings", static_cast<float const*>(mWordEmb.values),
            PluginFieldType::kFLOAT32, mWordEmb.count);
        mDataToSerialize.emplace_back("bert_embeddings_token_type_embeddings",
            static_cast<float const*>(mTokEmb.values), PluginFieldType::kFLOAT32, mTokEmb.count);
        mDataToSerialize.emplace_back("bert_embeddings_position_embeddings", static_cast<float const*>(mPosEmb.values),
            PluginFieldType::kFLOAT32, mPosEmb.count);
    }
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

int32_t EmbLayerNormVarSeqlenPluginHFace::onShapeChange(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    try
    {
        BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginHFace onShapeChange");
        checkConfigurationInputs(inputs, nbInputs, outputs, nbOutputs);

        // output 0 is the embedding
        PLUGIN_ASSERT(static_cast<size_t>(outputs[0].dims.d[1]) == static_cast<size_t>(mLd));
        PLUGIN_ASSERT(outputs[0].type == mType);
        // output 1 is the mask indices (empty for HFace variant)
        PLUGIN_ASSERT(outputs[1].dims.nbDims == 0);
        PLUGIN_ASSERT(outputs[1].type == mMaskType);
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

int32_t EmbLayerNormVarSeqlenPluginMTron::onShapeChange(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    try
    {
        // Validate input arguments
        BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginMTron onShapeChange");
        checkConfigurationInputs(inputs, nbInputs, outputs, nbOutputs);
        PLUGIN_ASSERT(static_cast<size_t>(outputs[0].dims.d[1]) == static_cast<size_t>(mLd));

        PLUGIN_ASSERT(outputs[1].dims.nbDims == 4);
        PLUGIN_ASSERT(static_cast<size_t>(outputs[1].dims.d[0]) == static_cast<size_t>(inputs[0].dims.d[0]));
        PLUGIN_ASSERT(static_cast<size_t>(outputs[1].dims.d[1]) == static_cast<size_t>(mLd));
        PLUGIN_ASSERT(outputs[1].dims.d[2] == 1);
        PLUGIN_ASSERT(outputs[1].dims.d[3] == 1);

        PLUGIN_ASSERT(outputs[0].type == mType);
        PLUGIN_ASSERT(outputs[1].type == mMaskType);
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

IPluginV3* EmbLayerNormVarSeqlenPluginBase::attachToContext(IPluginResourceContext* context) noexcept
{
    return clone();
}

int32_t EmbLayerNormVarSeqlenPluginHFace::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /* outputDesc */, void const* const* inputs, void* const* outputs, void* /* workspace */,
    cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr && inputs != nullptr && outputs != nullptr);

        int32_t const batchSize = inputDesc[2].dims.d[0] - 1;
        // read out the maximum sequence length from the dummy input
        int32_t const maxSeqlen = inputDesc[3].dims.d[0];

        // There are four versions of the kernel which are optimized for sequence lengths 384, 256, 192 and 128.
        // Find the closest sequence length bigger than the max seq length in this batch.
        int32_t S = 384;
        if (maxSeqlen <= 128)
        {
            S = 128;
        }
        else if (maxSeqlen <= 192)
        {
            S = 192;
        }
        else if (maxSeqlen <= 256)
        {
            S = 256;
        }

        // Our plugin outputs only one tensor
        auto const inputIds = static_cast<int32_t const*>(inputs[0]);
        auto const segmentIds = static_cast<int32_t const*>(inputs[1]);
        int32_t const* cuSeqlens = static_cast<int32_t const*>(inputs[2]);

        float const* beta = mBetaDev.get();
        float const* gamma = mGammaDev.get();
        if (mType == DataType::kFLOAT)
        {
            auto output = static_cast<float*>(outputs[0]);
            auto const wordEmb = static_cast<float const*>(mWordEmbDev.get());
            auto const tokEmb = static_cast<float const*>(mTokEmbDev.get());
            auto const posEmb = static_cast<float const*>(mPosEmbDev.get());

            return embSkipLayerNormHFace<float>(stream, static_cast<int32_t>(mLd), batchSize, S, inputIds, segmentIds,
                cuSeqlens, beta, gamma, wordEmb, posEmb, tokEmb, mWordVocabSize, mTokVocabSize, output);
        }
        if (mType == DataType::kHALF)
        {
            auto output = static_cast<half*>(outputs[0]);
            auto const wordEmb = static_cast<half const*>(mWordEmbDev.get());
            auto const tokEmb = static_cast<half const*>(mTokEmbDev.get());
            auto const posEmb = static_cast<half const*>(mPosEmbDev.get());

            return embSkipLayerNormHFace<half>(stream, static_cast<int32_t>(mLd), batchSize, S, inputIds, segmentIds,
                cuSeqlens, beta, gamma, wordEmb, posEmb, tokEmb, mWordVocabSize, mTokVocabSize, output);
        }
        else
        {
            gLogError << "Unsupported type error, expected [kHALF,kFLOAT], but received " << static_cast<int32_t>(mType)
                      << std::endl;

            return STATUS_NOT_SUPPORTED;
        }

        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

int32_t EmbLayerNormVarSeqlenPluginMTron::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /* outputDesc */, void const* const* inputs, void* const* outputs, void* /* workspace */,
    cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr && inputs != nullptr && outputs != nullptr);

        int32_t const batchSize = inputDesc[2].dims.d[0] - 1;
        // read out the maximum sequence length from the dummy input
        int32_t const maxSeqlen = inputDesc[3].dims.d[0];

        // There are four versions of the kernel which are optimized for sequence lengths 384, 256, 192 and 128.
        // Find the closest sequence length bigger than the max seq length in this batch.
        int32_t S = 384;
        if (maxSeqlen <= 128)
        {
            S = 128;
        }
        else if (maxSeqlen <= 192)
        {
            S = 192;
        }
        else if (maxSeqlen <= 256)
        {
            S = 256;
        }

        // Our plugin outputs only one tensor
        auto const inputIds = static_cast<int32_t const*>(inputs[0]);
        auto const segmentIds = static_cast<int32_t const*>(inputs[1]);
        int32_t const* cuSeqlens = static_cast<int32_t const*>(inputs[2]);

        float const* beta = mBetaDev.get();
        float const* gamma = mGammaDev.get();
        if (mType == DataType::kFLOAT)
        {
            auto output = static_cast<float*>(outputs[0]);
            auto skip = static_cast<float*>(outputs[1]);
            auto const wordEmb = static_cast<float const*>(mWordEmbDev.get());
            auto const tokEmb = static_cast<float const*>(mTokEmbDev.get());
            auto const posEmb = static_cast<float const*>(mPosEmbDev.get());

            return embSkipLayerNormMTron<float>(stream, static_cast<int32_t>(mLd), batchSize, S, inputIds, segmentIds,
                cuSeqlens, beta, gamma, wordEmb, posEmb, tokEmb, mWordVocabSize, mTokVocabSize, output, skip);
        }
        if (mType == DataType::kHALF)
        {
            auto output = static_cast<half*>(outputs[0]);
            auto skip = static_cast<half*>(outputs[1]);
            auto const wordEmb = static_cast<half const*>(mWordEmbDev.get());
            auto const tokEmb = static_cast<half const*>(mTokEmbDev.get());
            auto const posEmb = static_cast<half const*>(mPosEmbDev.get());

            return embSkipLayerNormMTron<half>(stream, static_cast<int32_t>(mLd), batchSize, S, inputIds, segmentIds,
                cuSeqlens, beta, gamma, wordEmb, posEmb, tokEmb, mWordVocabSize, mTokVocabSize, output, skip);
        }
        else
        {
            gLogError << "Unsupported type error, expected [kHALF,kFLOAT], but received " << static_cast<int32_t>(mType)
                      << std::endl;

            return STATUS_NOT_SUPPORTED;
        }

        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

// end IPluginV3OneRuntime method definitions

///////
// IPluginV3OneBuild method definitions
// - getNbOutputs() (Base)
// - supportsFormatCombination() (Base)
// - getOutputShapes (HFace, MTron)
// - getOutputDataTypes() (Base)
// - configurePlugin() (Base)
// - getWorkSpaceSize() (Base)
//////

int32_t EmbLayerNormVarSeqlenPluginBase::getNbOutputs() const noexcept
{
    return 2;
}

bool EmbLayerNormVarSeqlenPluginBase::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // The four inputs to this plugin input_ids, segment_ids, cu_seqlens and a dummy input with the
    // size of the max seq length in that order
    PLUGIN_ASSERT(nbInputs == 4);
    // The two outputs of the plugin are embedding and the mask
    PLUGIN_ASSERT(nbOutputs == 2);

    PluginTensorDesc const& desc = inOut[pos].desc;
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }
    if (pos == 0 || pos == 2) // input_ids and cu_seqlens
    {
        return desc.type == DataType::kINT32 && desc.dims.nbDims == 1;
    }

    PluginTensorDesc const& prev = inOut[pos - 1].desc;
    if (pos == 1) // segment ids: check it's the same as input_ids
    {
        return desc.type == DataType::kINT32 && desc.dims.nbDims == 1 && desc.dims.d[0] == prev.dims.d[0];
    }

    if (pos == 3)
    {
        return desc.dims.nbDims == 1;
    }

    // embedded sequence
    if (pos == nbInputs)
    {
        return desc.type == mType && desc.dims.nbDims == 4 && desc.dims.d[0] == inOut[0].desc.dims.d[0]
            && desc.dims.d[2] == 1 && desc.dims.d[3] == 1;
    }
    // mask
    return desc.type == mMaskType;
}

int32_t EmbLayerNormVarSeqlenPluginHFace::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* shapeInputs, int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);

        // Input should be input ids and token ids and cumulative seqlens
        // Output should be the embeddings tensor and mask indices
        PLUGIN_ASSERT(nbInputs == 4);
        PLUGIN_ASSERT(nbOutputs == 2);

        PLUGIN_ASSERT(inputs[0].nbDims == 1); // sum of all s
        PLUGIN_ASSERT(inputs[0].nbDims == inputs[1].nbDims);

        PLUGIN_ASSERT(inputs[2].nbDims == 1); // B+1

        // output 0 : embedded input
        outputs[0].nbDims = 4;
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = exprBuilder.constant(mLd);
        outputs[0].d[2] = exprBuilder.constant(1);
        outputs[0].d[3] = exprBuilder.constant(1);

        // Output 1 : maskIdx
        // Return empty tensor since this is dummy output, we do not delete it for backward compatibility.
        outputs[1].nbDims = 0;
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

int32_t EmbLayerNormVarSeqlenPluginMTron::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* shapeInputs, int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);
        // Input should be input ids and token ids and cumulative seqlens
        // Output should be the embeddings tensor and mask indices
        PLUGIN_ASSERT(nbInputs == 4);
        PLUGIN_ASSERT(nbOutputs == 2);

        PLUGIN_ASSERT(inputs[0].nbDims == 1); // sum of all s
        PLUGIN_ASSERT(inputs[0].nbDims == inputs[1].nbDims);
        PLUGIN_ASSERT(inputs[2].nbDims == 1); // B+1

        // Output 0 : embedded input
        outputs[0].nbDims = 4;
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = exprBuilder.constant(mLd);
        outputs[0].d[2] = exprBuilder.constant(1);
        outputs[0].d[3] = exprBuilder.constant(1);

        // Output 1 : maskIdx
        outputs[1].nbDims = 4;
        outputs[1].d[0] = inputs[0].d[0];
        outputs[1].d[1] = exprBuilder.constant(mLd);
        outputs[1].d[2] = exprBuilder.constant(1);
        outputs[1].d[3] = exprBuilder.constant(1);

        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

int32_t EmbLayerNormVarSeqlenPluginBase::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_ASSERT(mType == DataType::kHALF || mType == DataType::kFLOAT);
        outputTypes[0] = mType;
        outputTypes[1] = mMaskType;
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

int32_t EmbLayerNormVarSeqlenPluginBase::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    return pluginStatus_t::STATUS_SUCCESS;
}

size_t EmbLayerNormVarSeqlenPluginBase::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}
// End IPluginV3OneBuild method definitions

//////
// IPluginV3OneCore method definitions
// - getPluginVersion() (MTron, HFace)
// - getPluginName() (Base)
// - getPluginNamespace() (Base)
// - setPluginNamespace() (Base)
//////
char const* EmbLayerNormVarSeqlenPluginHFace::getPluginVersion() const noexcept
{
    return kEMB_LAYER_NORM_VAR_SEQLEN_VERSION_HFACE;
}

char const* EmbLayerNormVarSeqlenPluginMTron::getPluginVersion() const noexcept
{
    return kEMB_LAYER_NORM_VAR_SEQLEN_VERSION_MTRON;
}

char const* EmbLayerNormVarSeqlenPluginBase::getPluginName() const noexcept
{
    return kEMB_LAYER_NORM_VAR_SEQLEN_NAME;
}

char const* EmbLayerNormVarSeqlenPluginBase::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void EmbLayerNormVarSeqlenPluginBase::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}
// End IPluginV3OneCore method definitions

//////////////////////////// Plugin Creator member definitions /////////////////////////////

EmbLayerNormVarSeqlenPluginBaseCreator::EmbLayerNormVarSeqlenPluginBaseCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("output_fp16", nullptr, PluginFieldType::kINT32, 1));
    // the length of beta, gamma, word_emb, pos_emb, and tok_emb will only be known at the time of plugin creation
    // so we set it to 0 here
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_layernorm_beta", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_layernorm_gamma", nullptr, PluginFieldType::kFLOAT32, 0));
    // the embeddings datatype is determined by the output_fp16 attribute known at runtime
    // so we set it to kUNKNOWN here
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_word_embeddings", nullptr, PluginFieldType::kUNKNOWN, 0));
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_token_type_embeddings", nullptr, PluginFieldType::kUNKNOWN, 0));
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_position_embeddings", nullptr, PluginFieldType::kUNKNOWN, 0));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* EmbLayerNormVarSeqlenPluginBaseCreator::getPluginName() const noexcept
{
    return kEMB_LAYER_NORM_VAR_SEQLEN_NAME;
}

char const* EmbLayerNormVarSeqlenPluginHFaceCreator::getPluginVersion() const noexcept
{
    return kEMB_LAYER_NORM_VAR_SEQLEN_VERSION_HFACE;
}

char const* EmbLayerNormVarSeqlenPluginMTronCreator::getPluginVersion() const noexcept
{
    return kEMB_LAYER_NORM_VAR_SEQLEN_VERSION_MTRON;
}

PluginFieldCollection const* EmbLayerNormVarSeqlenPluginBaseCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV3* EmbLayerNormVarSeqlenPluginHFaceCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        BERT_DEBUG_MSG("EmbLayerNormVarSeqlenHFace createPlugin");

        Weights beta{};  // required attribute: validateRequiredAttributesExist() call in initializeFields() will verify
                         // existence
        Weights gamma{}; // required attribute: validateRequiredAttributesExist() call in initializeFields() will verify
                         // existence
        Weights word_emb{}; // required attribute: validateRequiredAttributesExist() call in initializeFields() will
                            // verify existence
        Weights pos_emb{};  // required attribute: validateRequiredAttributesExist() call in initializeFields() will
                            // verify existence
        Weights tok_emb{};  // required attribute: validateRequiredAttributesExist() call in initializeFields() will
                            // verify existence
        bool output_fp16 = initializeFields(name, fc, beta, gamma, word_emb, pos_emb, tok_emb);

        BERT_DEBUG_MSG("Building the Plugin...");
        EmbLayerNormVarSeqlenPluginHFace* p = new EmbLayerNormVarSeqlenPluginHFace(
            name, output_fp16 ? DataType::kHALF : DataType::kFLOAT, beta, gamma, word_emb, pos_emb, tok_emb);
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV3* EmbLayerNormVarSeqlenPluginMTronCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        BERT_DEBUG_MSG("EmbLayerNormVarSeqlenMTron createPlugin");

        Weights beta{};  // required attribute: validateRequiredAttributesExist() call in initializeFields() will verify
                         // existence
        Weights gamma{}; // required attribute: validateRequiredAttributesExist() call in initializeFields() will verify
                         // existence
        Weights word_emb{}; // required attribute: validateRequiredAttributesExist() call in initializeFields() will
                            // verify existence
        Weights pos_emb{};  // required attribute: validateRequiredAttributesExist() call in initializeFields() will
                            // verify existence
        Weights tok_emb{};  // required attribute: validateRequiredAttributesExist() call in initializeFields() will
                            // verify existence
        bool output_fp16 = initializeFields(name, fc, beta, gamma, word_emb, pos_emb, tok_emb);

        BERT_DEBUG_MSG("Building the Plugin...");
        EmbLayerNormVarSeqlenPluginMTron* p = new EmbLayerNormVarSeqlenPluginMTron(
            name, output_fp16 ? DataType::kHALF : DataType::kFLOAT, beta, gamma, word_emb, pos_emb, tok_emb);
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void EmbLayerNormVarSeqlenPluginBaseCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* EmbLayerNormVarSeqlenPluginBaseCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
