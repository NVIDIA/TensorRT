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

#include <cstring>
#include <set>
#include <vector>

#include "NvInfer.h"
#include "common/serialize.hpp"
#include "embLayerNormPlugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace nvinfer1::plugin::bert;


namespace
{
char const* EMB_LAYER_NORM_VERSION{"1"};
char const* EMB_LAYER_NORM_NAME{"CustomEmbLayerNormPluginDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection EmbLayerNormPluginDynamicCreator::mFC{};
std::vector<PluginField> EmbLayerNormPluginDynamicCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(EmbLayerNormPluginDynamicCreator);

EmbLayerNormPluginDynamic::EmbLayerNormPluginDynamic(std::string const& name, DataType const type,
    DataType const mhaType, Weights const& beta, Weights const& gamma, Weights const& wordEmb, Weights const& posEmb,
    Weights const& tokEmb, bool const useFullMask)
    : mLayerName(name)
    , mLd(beta.count)
    , mType(type)
    , mUseFullMask(useFullMask)
    , mMhaType(mhaType)
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
    mSM = getSMVersion();
    // mS is set during configure

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

EmbLayerNormPluginDynamic::EmbLayerNormPluginDynamic(std::string const& name, void const* data, size_t length)
    : mLayerName(name)
    , mGammaDev(nullptr)
    , mBetaDev(nullptr)
    , mWordEmbDev(nullptr)
    , mTokEmbDev(nullptr)
    , mPosEmbDev(nullptr)
{
    BERT_DEBUG_MSG("EmbLayerNormPluginDynamic deserialize.");

    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mMhaType);
    deserialize_value(&data, &length, &mLd);
    deserialize_value(&data, &length, &mS);
    deserialize_value(&data, &length, &mWordVocabSize);
    deserialize_value(&data, &length, &mPosVocabSize);
    deserialize_value(&data, &length, &mTokVocabSize);
    deserialize_value(&data, &length, &mUseFullMask);
    deserialize_value(&data, &length, &mSM);

    char const* d = static_cast<char const*>(data);
    mBeta.convertAndCopy(d, mLd, nvinfer1::DataType::kFLOAT);
    mGamma.convertAndCopy(d, mLd, nvinfer1::DataType::kFLOAT);
    mWordEmb.convertAndCopy(d, mLd * mWordVocabSize, mType);
    mPosEmb.convertAndCopy(d, mLd * mPosVocabSize, mType);
    mTokEmb.convertAndCopy(d, mLd * mTokVocabSize, mType);

    copyToDevice(mGamma, sizeof(float) * mGamma.count, mGammaDev);
    copyToDevice(mBeta, sizeof(float) * mBeta.count, mBetaDev);
    copyToDevice(mWordEmb, getWeightsSize(mWordEmb, mType), mWordEmbDev);
    copyToDevice(mPosEmb, getWeightsSize(mPosEmb, mType), mPosEmbDev);
    copyToDevice(mTokEmb, getWeightsSize(mTokEmb, mType), mTokEmbDev);
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* EmbLayerNormPluginDynamic::clone() const noexcept
{
    try
    {
        BERT_DEBUG_MSG("EmbLayerNormPluginDynamic clone.");

        auto p = new EmbLayerNormPluginDynamic(
            mLayerName, mType, mMhaType, mBeta, mGamma, mWordEmb, mPosEmb, mTokEmb, mUseFullMask);
        p->mS = mS;
        p->setPluginNamespace(mNamespace.c_str());

        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs EmbLayerNormPluginDynamic::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        // Input should be input ids and token ids and the input mask
        // Output should be the embeddings tensor and mask indices
        PLUGIN_ASSERT(nbInputs == 3);

        PLUGIN_ASSERT(inputs[0].nbDims == 2); // BxS
        PLUGIN_ASSERT(inputs[0].nbDims == inputs[1].nbDims);
        PLUGIN_ASSERT(inputs[0].nbDims == inputs[2].nbDims);

        PLUGIN_ASSERT(outputIndex == 0 || outputIndex == 1);

        if (outputIndex == 0)
        {
            DimsExprs ret;
            ret.nbDims = 5;
            ret.d[0] = inputs[0].d[0];
            ret.d[1] = inputs[0].d[1];
            ret.d[2] = exprBuilder.constant(mLd);
            ret.d[3] = exprBuilder.constant(1);
            ret.d[4] = exprBuilder.constant(1);
            return ret;
        }

        DimsExprs ret;
        ret.nbDims = 2;
        ret.d[0] = inputs[0].d[BDIM];
        auto cms0 = exprBuilder.constant(unfusedMaskSize);

        // this code must match getMHAMaskPackedSize in bertCommon.h
        bool const isSmOK = (mSM == kSM_75 || mSM == kSM_80 || mSM == kSM_86 || mSM == kSM_87 || mSM == kSM_89 ||mSM == kSM_90);
        bool const isPrecisionOK = (mMhaType == nvinfer1::DataType::kHALF || mMhaType == nvinfer1::DataType::kINT8);
        if (mUseFullMask || (isSmOK && isPrecisionOK))
        {
            // support 128, 384 in both int8 and fp16
            auto cms128 = exprBuilder.constant(packedMaskSize128);
            auto cms384 = exprBuilder.constant(packedMaskSize384);
            auto c128 = exprBuilder.constant(128);
            auto c384 = exprBuilder.constant(384);
            auto is128 = exprBuilder.operation(DimensionOperation::kEQUAL, *inputs[0].d[SDIM], *c128);
            auto is384 = exprBuilder.operation(DimensionOperation::kEQUAL, *inputs[0].d[SDIM], *c384);
            auto sel128 = exprBuilder.operation(DimensionOperation::kPROD, *is128, *cms128);
            auto sel384 = exprBuilder.operation(DimensionOperation::kPROD, *is384, *cms384);
            auto maskSize = exprBuilder.operation(DimensionOperation::kSUM, *sel384, *sel128);

            // support 64, 96 in both int8 and fp16
            auto cms64 = exprBuilder.constant(packedMaskSize64);
            auto cms96 = exprBuilder.constant(packedMaskSize96);
            auto c64 = exprBuilder.constant(64);
            auto c96 = exprBuilder.constant(96);

            auto is64 = exprBuilder.operation(DimensionOperation::kEQUAL, *inputs[0].d[SDIM], *c64);
            auto is96 = exprBuilder.operation(DimensionOperation::kEQUAL, *inputs[0].d[SDIM], *c96);
            auto sel64 = exprBuilder.operation(DimensionOperation::kPROD, *is64, *cms64);
            auto sel96 = exprBuilder.operation(DimensionOperation::kPROD, *is96, *cms96);
            auto maskSize2 = exprBuilder.operation(DimensionOperation::kSUM, *sel64, *sel96);
            maskSize = exprBuilder.operation(DimensionOperation::kSUM, *maskSize, *maskSize2);

            auto is0 = exprBuilder.operation(DimensionOperation::kEQUAL, *maskSize, *exprBuilder.constant(0));
            auto sel0 = exprBuilder.operation(DimensionOperation::kPROD, *is0, *cms0);
            auto combinedMaskSize = exprBuilder.operation(DimensionOperation::kSUM, *maskSize, *sel0);
            ret.d[1] = combinedMaskSize;
        }
        else
        {
            ret.d[1] = cms0;
        }

        return ret;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool EmbLayerNormPluginDynamic::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // 3 inputs of size BxS
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(nbOutputs == 2);

    PluginTensorDesc const& desc = inOut[pos];
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }
    if (pos == 0)
    {
        return desc.type == DataType::kINT32 && desc.dims.nbDims == 2;
    }

    PluginTensorDesc const& prev = inOut[pos - 1];
    if (pos == 1 || pos == 2)
    {
        return desc.type == DataType::kINT32 && desc.dims.nbDims == 2 && desc.dims.d[BDIM] == prev.dims.d[BDIM]
            && desc.dims.d[SDIM] == prev.dims.d[SDIM];
    }

    // embedded sequence
    if (pos == 3)
    {
        return desc.type == mType && desc.dims.nbDims == 5 && desc.dims.d[BDIM] == prev.dims.d[BDIM]
            && desc.dims.d[SDIM] == prev.dims.d[SDIM] && desc.dims.d[3] == 1 && desc.dims.d[4] == 1;
    }
    // mask
    return desc.type == DataType::kFLOAT;
}

void EmbLayerNormPluginDynamic::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    BERT_DEBUG_MSG("EmbLayerNormPluginDynamic configurePlugin.");

    // Validate input arguments
    PLUGIN_ASSERT(nbOutputs == 2);
    PLUGIN_ASSERT(nbInputs == 3);

    PLUGIN_ASSERT(inputs[0].desc.dims.nbDims == 2);
    int32_t const S = inputs[0].desc.dims.d[SDIM];
    mS = S;
    int32_t const B = inputs[0].desc.dims.d[BDIM];
    TRT_UNUSED B;
    PLUGIN_ASSERT(mS == static_cast<size_t>(inputs[1].desc.dims.d[SDIM]));
    PLUGIN_ASSERT(B == inputs[1].desc.dims.d[BDIM]);
    PLUGIN_ASSERT(mS == static_cast<size_t>(inputs[2].desc.dims.d[SDIM]));
    PLUGIN_ASSERT(B == inputs[2].desc.dims.d[BDIM]);

    PLUGIN_ASSERT(outputs[0].desc.dims.nbDims == 5);
    PLUGIN_ASSERT(static_cast<size_t>(outputs[0].desc.dims.d[SDIM]) == mS);
    PLUGIN_ASSERT(outputs[0].desc.dims.d[BDIM] == B);
    PLUGIN_ASSERT(static_cast<size_t>(outputs[0].desc.dims.d[2]) == mLd);
    PLUGIN_ASSERT(outputs[0].desc.dims.d[3] == 1);
    PLUGIN_ASSERT(outputs[0].desc.dims.d[4] == 1);

    if (mUseFullMask)
    {
        // user force full_mask
        PLUGIN_ASSERT(outputs[1].desc.dims.nbDims == 2);
        PLUGIN_ASSERT(outputs[1].desc.dims.d[0] == B);
        PLUGIN_ASSERT((outputs[1].desc.dims.d[1] == -1) || (outputs[1].desc.dims.d[1] == packedMaskSize384)
            || (outputs[1].desc.dims.d[1] == packedMaskSize128));
    }
    else
    {
        // auto detect using mhatype
        if (S != -1 && B != -1)
        {
            PLUGIN_ASSERT(outputs[1].desc.dims.nbDims == 2);
            PLUGIN_ASSERT(outputs[1].desc.dims.d[0] == B);
            int32_t packedSize = getMHAMaskPackedSize(mSM, mMhaType, S);
            TRT_UNUSED packedSize;
            PLUGIN_ASSERT(outputs[1].desc.dims.d[1] == -1 || outputs[1].desc.dims.d[1] == packedSize);
        }
    }

    PLUGIN_ASSERT(inputs[0].desc.type == DataType::kINT32);
    PLUGIN_ASSERT(inputs[1].desc.type == DataType::kINT32);
    PLUGIN_ASSERT(inputs[2].desc.type == DataType::kINT32);
    PLUGIN_ASSERT(outputs[0].desc.type == mType);
    PLUGIN_ASSERT(outputs[1].desc.type == DataType::kFLOAT);
}

size_t EmbLayerNormPluginDynamic::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t EmbLayerNormPluginDynamic::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        int32_t const batchSize = inputDesc->dims.d[BDIM];
        int32_t const S = inputDesc->dims.d[SDIM];
        int32_t status = STATUS_FAILURE;

        // Our plugin outputs only one tensor
        auto const inputIds = static_cast<int32_t const*>(inputs[0]);
        auto const segmentIds = static_cast<int32_t const*>(inputs[1]);
        auto const inputMask = static_cast<int32_t const*>(inputs[2]);

        const float* beta = mBetaDev.get();
        const float* gamma = mGammaDev.get();
        if (mType == DataType::kFLOAT)
        {
            auto output = static_cast<float*>(outputs[0]);
            auto const wordEmb = static_cast<const float*>(mWordEmbDev.get());
            auto const tokEmb = static_cast<const float*>(mTokEmbDev.get());
            auto const posEmb = static_cast<const float*>(mPosEmbDev.get());
            status = embSkipLayerNorm<float>(stream, static_cast<int>(mLd), batchSize, S, inputIds, segmentIds, beta,
                gamma, wordEmb, posEmb, tokEmb, mWordVocabSize, mTokVocabSize, output);

            if (status != cudaSuccess)
            {
                return status;
            }
        }
        else if (mType == DataType::kHALF)
        {
            auto output = static_cast<half*>(outputs[0]);
            auto const wordEmb = static_cast<const half*>(mWordEmbDev.get());
            auto const tokEmb = static_cast<const half*>(mTokEmbDev.get());
            auto const posEmb = static_cast<const half*>(mPosEmbDev.get());
            status = embSkipLayerNorm<half>(stream, static_cast<int>(mLd), batchSize, S, inputIds, segmentIds, beta,
                gamma, wordEmb, posEmb, tokEmb, mWordVocabSize, mTokVocabSize, output);

            if (status != cudaSuccess)
            {
                return status;
            }
        }
        else
        {
            gLogError << "Unsupported type error, expected [kHALF,kFLOAT], but received " << static_cast<int>(mType)
                      << std::endl;

            return STATUS_NOT_SUPPORTED;
        }

        // check mha use fused kernel
        if (mUseFullMask || unfusedMaskSize != getMHAMaskPackedSize(mSM, mMhaType, S))
        {
            size_t warps_m = 0, warps_n = 0, warps_k = 1;
            if (S == 64 || S == 96 || S == 128)
            {
                warps_m = 2;
                warps_n = 2;
            }
            else if (S == 384)
            {
                warps_m = 1;
                warps_n = 8;
            }
            uint32_t* inputMaskX = static_cast<uint32_t*>(outputs[1]);

            status = convertMask(S, batchSize, warps_m, warps_n, warps_k, inputMask, inputMaskX, stream);
        }
        else
        {
            int32_t* maskIdx = static_cast<int32_t*>(outputs[1]);
            status = computeMaskIdx(stream, S, batchSize, inputMask, maskIdx);
        }

        return status;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

// IPluginV2Ext Methods
DataType EmbLayerNormPluginDynamic::getOutputDataType(
    int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{

    PLUGIN_ASSERT(index == 0 || index == 1);
    if (index == 0)
    {
        PLUGIN_ASSERT(mType == DataType::kHALF || mType == DataType::kFLOAT);
        return mType;
    }
    return DataType::kFLOAT;
}

// IPluginV2 Methods
char const* EmbLayerNormPluginDynamic::getPluginType() const noexcept
{
    return EMB_LAYER_NORM_NAME;
}

char const* EmbLayerNormPluginDynamic::getPluginVersion() const noexcept
{
    return EMB_LAYER_NORM_VERSION;
}

int32_t EmbLayerNormPluginDynamic::getNbOutputs() const noexcept
{
    return 2;
}

int32_t EmbLayerNormPluginDynamic::initialize() noexcept
{
    return 0;
}

void EmbLayerNormPluginDynamic::terminate() noexcept
{
    BERT_DEBUG_MSG("EmbLayerNormPluginDynamic terminate.");
}

size_t EmbLayerNormPluginDynamic::getSerializationSize() const noexcept
{
    size_t const wordSize = getElementSize(mType);
    return sizeof(mType)                  // type
        + sizeof(mMhaType)                // mha plugin datatype
        + sizeof(mLd) * 5                 // mLd, mS, m*VocabSize
        + sizeof(mUseFullMask)            // mask type
        + sizeof(mSM)                     // smversion
        + 2 * sizeof(float) * mLd         // beta + gamma
        + wordSize * mLd * mWordVocabSize // word emb
        + wordSize * mLd * mPosVocabSize  // pos emb
        + wordSize * mLd * mTokVocabSize  // tok emb
        ;
}

void EmbLayerNormPluginDynamic::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mMhaType);
    serialize_value(&buffer, mLd);
    serialize_value(&buffer, mS);
    serialize_value(&buffer, mWordVocabSize);
    serialize_value(&buffer, mPosVocabSize);
    serialize_value(&buffer, mTokVocabSize);
    serialize_value(&buffer, mUseFullMask);
    serialize_value(&buffer, mSM);

    char* d = static_cast<char*>(buffer);
    serFromDev(d, mBetaDev.get(), mLd);
    serFromDev(d, mGammaDev.get(), mLd);
    size_t const wordSize = getElementSize(mType);
    serFromDev(d, static_cast<char*>(mWordEmbDev.get()), mLd * mWordVocabSize * wordSize);
    serFromDev(d, static_cast<char*>(mPosEmbDev.get()), mLd * mPosVocabSize * wordSize);
    serFromDev(d, static_cast<char*>(mTokEmbDev.get()), mLd * mTokVocabSize * wordSize);
}

void EmbLayerNormPluginDynamic::destroy() noexcept
{
    BERT_DEBUG_MSG("EmbLayerNormPluginDynamic destroy.");
    // This gets called when the network containing plugin is destroyed
    mGammaDev.reset(nullptr);
    mBetaDev.reset(nullptr);
    mWordEmbDev.reset(nullptr);
    mPosEmbDev.reset(nullptr);
    mTokEmbDev.reset(nullptr);
    delete this;
}

void EmbLayerNormPluginDynamic::setPluginNamespace(char const* libNamespace) noexcept
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

char const* EmbLayerNormPluginDynamic::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////////////

EmbLayerNormPluginDynamicCreator::EmbLayerNormPluginDynamicCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_layernorm_beta"));
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_layernorm_gamma"));
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_word_embeddings"));
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_token_type_embeddings"));
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_position_embeddings"));
    mPluginAttributes.emplace_back(PluginField("output_fp16"));
    mPluginAttributes.emplace_back(PluginField("full_mask"));
    mPluginAttributes.emplace_back(PluginField("mha_type_id"));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* EmbLayerNormPluginDynamicCreator::getPluginName() const noexcept
{
    return EMB_LAYER_NORM_NAME;
}

char const* EmbLayerNormPluginDynamicCreator::getPluginVersion() const noexcept
{
    return EMB_LAYER_NORM_VERSION;
}

const PluginFieldCollection* EmbLayerNormPluginDynamicCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* EmbLayerNormPluginDynamicCreator::createPlugin(char const* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        BERT_DEBUG_MSG("EmbLayerNormPluginDynamic createPlugin.");

        bool output_fp16 = false;
        bool useFullMask = false;
        Weights beta{};     // required attribute - validateRequiredAttributesExist() will verify existence
        Weights gamma{};    // required attribute - validateRequiredAttributesExist() will verify existence
        Weights word_emb{}; // required attribute - validateRequiredAttributesExist() will verify existence
        Weights pos_emb{};  // required attribute - validateRequiredAttributesExist() will verify existence
        Weights tok_emb{};  // required attribute - validateRequiredAttributesExist() will verify existence
        int32_t mhaTypeId = 0;
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

            if (field_name.compare("bert_embeddings_layernorm_gamma") == 0)
            {
                BERT_DEBUG_MSG("Building bert_embeddings_layernorm_gamma...");
                gamma.values = fc->fields[i].data;
                gamma.count = fc->fields[i].length;
                gamma.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("bert_embeddings_word_embeddings") == 0)
            {
                BERT_DEBUG_MSG("Building bert_embeddings_word_embeddings...");
                word_emb.values = fc->fields[i].data;
                word_emb.count = fc->fields[i].length;
                word_emb.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("bert_embeddings_token_type_embeddings") == 0)
            {
                BERT_DEBUG_MSG("Building bert_embeddings_token_type_embeddings...");
                tok_emb.values = fc->fields[i].data;
                tok_emb.count = fc->fields[i].length;
                tok_emb.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("bert_embeddings_position_embeddings") == 0)
            {
                BERT_DEBUG_MSG("Building bert_embeddings_position_embeddings...");
                pos_emb.values = fc->fields[i].data;
                pos_emb.count = fc->fields[i].length;
                pos_emb.type = fieldTypeToDataType(fc->fields[i].type);
            }
            if (field_name.compare("output_fp16") == 0)
            {
                BERT_DEBUG_MSG("Building output_fp16...");
                PLUGIN_VALIDATE(fc->fields[i].type == PluginFieldType::kINT32);
                output_fp16 = static_cast<int32_t const*>(fc->fields[i].data)[0] != 0;
            }
            if (field_name.compare("full_mask") == 0)
            {
                BERT_DEBUG_MSG("Building full_mask...");
                PLUGIN_VALIDATE(fc->fields[i].type == PluginFieldType::kINT32);
                useFullMask = static_cast<int32_t const*>(fc->fields[i].data)[0] != 0;
            }
            if (field_name.compare("mha_type_id") == 0)
            {
                mhaTypeId = *static_cast<int32_t const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(mhaTypeId >= 0 && mhaTypeId <= 3);
                BERT_DEBUG_VALUE("Building mha typeId: ", mhaTypeId);
            }
        }

        BERT_DEBUG_MSG("Building the Plugin...");
        DataType mhaType = static_cast<DataType>(mhaTypeId);
        EmbLayerNormPluginDynamic* p
            = new EmbLayerNormPluginDynamic(name, output_fp16 ? DataType::kHALF : DataType::kFLOAT, mhaType, beta,
                gamma, word_emb, pos_emb, tok_emb, useFullMask);
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* EmbLayerNormPluginDynamicCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call EmbLayerNormPluginDynamic::destroy()
        return new EmbLayerNormPluginDynamic(name, serialData, serialLength);
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void EmbLayerNormPluginDynamicCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        mNamespace = libNamespace;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

char const* EmbLayerNormPluginDynamicCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

#endif // CUDA_VERSION >= 10010
