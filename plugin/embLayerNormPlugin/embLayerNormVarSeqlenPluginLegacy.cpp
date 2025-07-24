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
#include "embLayerNormVarSeqlenPluginLegacy.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace nvinfer1::plugin::bert;

namespace
{
constexpr char const* kEMB_LAYER_NORM_VAR_SEQLEN_VERSION_HFACE{"2"};
constexpr char const* kEMB_LAYER_NORM_VAR_SEQLEN_VERSION_MTRON{"3"};
constexpr char const* kEMB_LAYER_NORM_VAR_SEQLEN_NAME{"CustomEmbLayerNormPluginDynamic"};
} // namespace

REGISTER_TENSORRT_PLUGIN(EmbLayerNormVarSeqlenPluginLegacyHFaceCreator);
REGISTER_TENSORRT_PLUGIN(EmbLayerNormVarSeqlenPluginLegacyMTronCreator);

EmbLayerNormVarSeqlenPluginLegacyBase::EmbLayerNormVarSeqlenPluginLegacyBase(std::string const& name, DataType type,
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

EmbLayerNormVarSeqlenPluginLegacyBase::EmbLayerNormVarSeqlenPluginLegacyBase(
    std::string const& name, void const* data, size_t length)
    : mLayerName(name)
    , mGammaDev(nullptr)
    , mBetaDev(nullptr)
    , mWordEmbDev(nullptr)
    , mTokEmbDev(nullptr)
    , mPosEmbDev(nullptr)
{
    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mLd);
    deserialize_value(&data, &length, &mWordVocabSize);
    deserialize_value(&data, &length, &mPosVocabSize);
    deserialize_value(&data, &length, &mTokVocabSize);
    deserialize_value(&data, &length, &mMaskType);

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

EmbLayerNormVarSeqlenPluginLegacyHFace::EmbLayerNormVarSeqlenPluginLegacyHFace(std::string const& name,
    DataType const type, Weights const& beta, Weights const& gamma, Weights const& wordEmb, Weights const& posEmb,
    Weights const& tokEmb)
    : EmbLayerNormVarSeqlenPluginLegacyBase(name, type, beta, gamma, wordEmb, posEmb, tokEmb, DataType::kINT32)
{
}

EmbLayerNormVarSeqlenPluginLegacyHFace::EmbLayerNormVarSeqlenPluginLegacyHFace(
    std::string const& name, void const* data, size_t length)
    : EmbLayerNormVarSeqlenPluginLegacyBase(name, data, length)
{
    BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginLegacyHFace deserialize");
}

EmbLayerNormVarSeqlenPluginLegacyMTron::EmbLayerNormVarSeqlenPluginLegacyMTron(std::string const& name,
    DataType const type, Weights const& beta, Weights const& gamma, Weights const& wordEmb, Weights const& posEmb,
    Weights const& tokEmb)
    : EmbLayerNormVarSeqlenPluginLegacyBase(name, type, beta, gamma, wordEmb, posEmb, tokEmb, type)
{
}

EmbLayerNormVarSeqlenPluginLegacyMTron::EmbLayerNormVarSeqlenPluginLegacyMTron(
    std::string const& name, void const* data, size_t length)
    : EmbLayerNormVarSeqlenPluginLegacyBase(name, data, length)
{
    BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginLegacyMTron deserialize");
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* EmbLayerNormVarSeqlenPluginLegacyHFace::clone() const noexcept
{
    try
    {
        BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginLegacyHFace clone");

        auto p
            = new EmbLayerNormVarSeqlenPluginLegacyHFace(mLayerName, mType, mBeta, mGamma, mWordEmb, mPosEmb, mTokEmb);
        p->setPluginNamespace(mNamespace.c_str());

        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* EmbLayerNormVarSeqlenPluginLegacyMTron::clone() const noexcept
{
    try
    {
        BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginLegacyMTron clone");

        auto p
            = new EmbLayerNormVarSeqlenPluginLegacyMTron(mLayerName, mType, mBeta, mGamma, mWordEmb, mPosEmb, mTokEmb);
        p->setPluginNamespace(mNamespace.c_str());

        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs EmbLayerNormVarSeqlenPluginLegacyHFace::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    // Input should be input ids and token ids and cumulative seqlens
    // Output should be the embeddings tensor and mask indices
    PLUGIN_ASSERT(nbInputs == 4);

    PLUGIN_ASSERT(inputs[0].nbDims == 1); // sum of all s
    PLUGIN_ASSERT(inputs[0].nbDims == inputs[1].nbDims);

    PLUGIN_ASSERT(inputs[2].nbDims == 1); // B+1

    PLUGIN_ASSERT(outputIndex == 0 || outputIndex == 1);

    if (outputIndex == 0)
    {
        DimsExprs ret;
        ret.nbDims = 4;
        ret.d[0] = inputs[0].d[0];
        ret.d[1] = exprBuilder.constant(mLd);
        ret.d[2] = exprBuilder.constant(1);
        ret.d[3] = exprBuilder.constant(1);
        return ret;
    }

    // Return empty tensor since this is dummy output, we do not delete it for backward compatibility.
    DimsExprs ret{};
    ret.nbDims = 0;
    return ret;
}

DimsExprs EmbLayerNormVarSeqlenPluginLegacyMTron::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    // Input should be input ids and token ids and cumulative seqlens
    // Output should be the embeddings tensor and mask indices
    PLUGIN_ASSERT(nbInputs == 4);

    PLUGIN_ASSERT(inputs[0].nbDims == 1); // sum of all s
    PLUGIN_ASSERT(inputs[0].nbDims == inputs[1].nbDims);

    PLUGIN_ASSERT(inputs[2].nbDims == 1); // B+1

    PLUGIN_ASSERT(outputIndex == 0 || outputIndex == 1);

    DimsExprs ret;
    ret.nbDims = 4;
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = exprBuilder.constant(mLd);
    ret.d[2] = exprBuilder.constant(1);
    ret.d[3] = exprBuilder.constant(1);
    return ret;
}

bool EmbLayerNormVarSeqlenPluginLegacyBase::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // The four inputs to this plugin input_ids, segment_ids, cu_seqlens and a dummy input with the
    // size of the max seq length in that order
    PLUGIN_ASSERT(nbInputs == 4);
    // The two outputs of the plugin are embedding and the mask
    PLUGIN_ASSERT(nbOutputs == 2);

    PluginTensorDesc const& desc = inOut[pos];
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }
    if (pos == 0 || pos == 2) // input_ids and cu_seqlens
    {
        return desc.type == DataType::kINT32 && desc.dims.nbDims == 1;
    }

    PluginTensorDesc const& prev = inOut[pos - 1];
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
        return desc.type == mType && desc.dims.nbDims == 4 && desc.dims.d[0] == inOut[0].dims.d[0]
            && desc.dims.d[2] == 1 && desc.dims.d[3] == 1;
    }
    // mask
    return desc.type == mMaskType;
}

void checkConfigurationInputs(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    // Validate input arguments
    PLUGIN_ASSERT(nbInputs == 4);
    PLUGIN_ASSERT(nbOutputs == 2);

    PLUGIN_ASSERT(inputs[0].desc.dims.nbDims == 1);
    PLUGIN_ASSERT(inputs[1].desc.dims.nbDims == 1);

    PLUGIN_ASSERT(inputs[1].desc.dims.d[0] == inputs[0].desc.dims.d[0]);

    PLUGIN_ASSERT(inputs[2].desc.dims.nbDims == 1);

    PLUGIN_ASSERT(outputs[0].desc.dims.nbDims == 4);
    PLUGIN_ASSERT(static_cast<size_t>(outputs[0].desc.dims.d[0]) == static_cast<size_t>(inputs[0].desc.dims.d[0]));
    PLUGIN_ASSERT(outputs[0].desc.dims.d[2] == 1);
    PLUGIN_ASSERT(outputs[0].desc.dims.d[3] == 1);

    PLUGIN_ASSERT(inputs[0].desc.type == DataType::kINT32);
    PLUGIN_ASSERT(inputs[1].desc.type == DataType::kINT32);
    PLUGIN_ASSERT(inputs[2].desc.type == DataType::kINT32);
}

void EmbLayerNormVarSeqlenPluginLegacyHFace::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginLegacyHFace configurePlugin");
    checkConfigurationInputs(inputs, nbInputs, outputs, nbOutputs);
    PLUGIN_ASSERT(static_cast<size_t>(outputs[0].desc.dims.d[1]) == static_cast<size_t>(mLd));

    // check mask
    PLUGIN_ASSERT(outputs[1].desc.dims.nbDims == 0);
    PLUGIN_ASSERT(outputs[0].desc.type == mType);
    PLUGIN_ASSERT(outputs[1].desc.type == mMaskType);
}

void EmbLayerNormVarSeqlenPluginLegacyMTron::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginLegacyMTron configurePlugin");
    checkConfigurationInputs(inputs, nbInputs, outputs, nbOutputs);
    PLUGIN_ASSERT(static_cast<size_t>(outputs[0].desc.dims.d[1]) == static_cast<size_t>(mLd));

    PLUGIN_ASSERT(outputs[1].desc.dims.nbDims == 4);
    PLUGIN_ASSERT(static_cast<size_t>(outputs[1].desc.dims.d[0]) == static_cast<size_t>(inputs[0].desc.dims.d[0]));
    PLUGIN_ASSERT(static_cast<size_t>(outputs[1].desc.dims.d[1]) == static_cast<size_t>(mLd));
    PLUGIN_ASSERT(outputs[1].desc.dims.d[2] == 1);
    PLUGIN_ASSERT(outputs[1].desc.dims.d[3] == 1);

    PLUGIN_ASSERT(outputs[0].desc.type == mType);
    PLUGIN_ASSERT(outputs[1].desc.type == mMaskType);
}

size_t EmbLayerNormVarSeqlenPluginLegacyBase::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t EmbLayerNormVarSeqlenPluginLegacyHFace::enqueue(PluginTensorDesc const* inputDesc,
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

int32_t EmbLayerNormVarSeqlenPluginLegacyMTron::enqueue(PluginTensorDesc const* inputDesc,
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

// IPluginV2Ext Methods
DataType EmbLayerNormVarSeqlenPluginLegacyBase::getOutputDataType(
    int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0 || index == 1);
    PLUGIN_ASSERT(mType == DataType::kHALF || mType == DataType::kFLOAT);
    return index == 0 ? mType : mMaskType;
}

// IPluginV2 Methods
char const* EmbLayerNormVarSeqlenPluginLegacyBase::getPluginType() const noexcept
{
    return kEMB_LAYER_NORM_VAR_SEQLEN_NAME;
}

char const* EmbLayerNormVarSeqlenPluginLegacyHFace::getPluginVersion() const noexcept
{
    return kEMB_LAYER_NORM_VAR_SEQLEN_VERSION_HFACE;
}

char const* EmbLayerNormVarSeqlenPluginLegacyMTron::getPluginVersion() const noexcept
{
    return kEMB_LAYER_NORM_VAR_SEQLEN_VERSION_MTRON;
}

int32_t EmbLayerNormVarSeqlenPluginLegacyBase::getNbOutputs() const noexcept
{
    return 2;
}

int32_t EmbLayerNormVarSeqlenPluginLegacyHFace::initialize() noexcept
{
    BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginLegacyHFace initialize");
    return 0;
}

int32_t EmbLayerNormVarSeqlenPluginLegacyMTron::initialize() noexcept
{
    BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginLegacyMTron initialize");
    return 0;
}

void EmbLayerNormVarSeqlenPluginLegacyHFace::terminate() noexcept
{
    BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginLegacyHFace terminate");
}

void EmbLayerNormVarSeqlenPluginLegacyMTron::terminate() noexcept
{
    BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginLegacyMTron terminate");
}

size_t EmbLayerNormVarSeqlenPluginLegacyBase::getSerializationSize() const noexcept
{
    size_t const wordSize = getElementSize(mType);
    return 2 * sizeof(float) * mLd        // beta + gamma
        + sizeof(mType)                   //
        + sizeof(mLd)                     //
        + sizeof(mWordVocabSize)          //
        + sizeof(mPosVocabSize)           //
        + sizeof(mTokVocabSize)           //
        + wordSize * mLd * mWordVocabSize // word emb
        + wordSize * mLd * mPosVocabSize  // pos emb
        + wordSize * mLd * mTokVocabSize  // tok emb
        + sizeof(mMaskType)               // mask type
        ;
}

void EmbLayerNormVarSeqlenPluginLegacyBase::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mLd);
    serialize_value(&buffer, mWordVocabSize);
    serialize_value(&buffer, mPosVocabSize);
    serialize_value(&buffer, mTokVocabSize);
    serialize_value(&buffer, mMaskType);

    char* d = static_cast<char*>(buffer);
    size_t const wordSize = getElementSize(mType);

    serFromDev(d, mBetaDev.get(), mLd);
    serFromDev(d, mGammaDev.get(), mLd);
    serFromDev(d, static_cast<char*>(mWordEmbDev.get()), mLd * mWordVocabSize * wordSize);
    serFromDev(d, static_cast<char*>(mPosEmbDev.get()), mLd * mPosVocabSize * wordSize);
    serFromDev(d, static_cast<char*>(mTokEmbDev.get()), mLd * mTokVocabSize * wordSize);
}

void EmbLayerNormVarSeqlenPluginLegacyBase::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    mGammaDev.reset(nullptr);
    mBetaDev.reset(nullptr);
    mWordEmbDev.reset(nullptr);
    mPosEmbDev.reset(nullptr);
    mTokEmbDev.reset(nullptr);
    delete this;
}

void EmbLayerNormVarSeqlenPluginLegacyHFace::destroy() noexcept
{
    BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginLegacyHFace destroy");
    EmbLayerNormVarSeqlenPluginLegacyBase::destroy();
}

void EmbLayerNormVarSeqlenPluginLegacyMTron::destroy() noexcept
{
    BERT_DEBUG_MSG("EmbLayerNormVarSeqlenPluginLegacyMTron destroy");
    EmbLayerNormVarSeqlenPluginLegacyBase::destroy();
}

void EmbLayerNormVarSeqlenPluginLegacyBase::setPluginNamespace(char const* libNamespace) noexcept
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

char const* EmbLayerNormVarSeqlenPluginLegacyBase::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////////////

EmbLayerNormVarSeqlenPluginLegacyBaseCreator::EmbLayerNormVarSeqlenPluginLegacyBaseCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_layernorm_beta"));
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_layernorm_gamma"));
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_word_embeddings"));
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_token_type_embeddings"));
    mPluginAttributes.emplace_back(PluginField("bert_embeddings_position_embeddings"));
    mPluginAttributes.emplace_back(PluginField("output_fp16"));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* EmbLayerNormVarSeqlenPluginLegacyBaseCreator::getPluginName() const noexcept
{
    return kEMB_LAYER_NORM_VAR_SEQLEN_NAME;
}

char const* EmbLayerNormVarSeqlenPluginLegacyHFaceCreator::getPluginVersion() const noexcept
{
    return kEMB_LAYER_NORM_VAR_SEQLEN_VERSION_HFACE;
}

char const* EmbLayerNormVarSeqlenPluginLegacyMTronCreator::getPluginVersion() const noexcept
{
    return kEMB_LAYER_NORM_VAR_SEQLEN_VERSION_MTRON;
}

PluginFieldCollection const* EmbLayerNormVarSeqlenPluginLegacyBaseCreator::getFieldNames() noexcept
{
    return &mFC;
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
    }
    return output_fp16;
}

IPluginV2* EmbLayerNormVarSeqlenPluginLegacyHFaceCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
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
        EmbLayerNormVarSeqlenPluginLegacyHFace* p = new EmbLayerNormVarSeqlenPluginLegacyHFace(
            name, output_fp16 ? DataType::kHALF : DataType::kFLOAT, beta, gamma, word_emb, pos_emb, tok_emb);
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* EmbLayerNormVarSeqlenPluginLegacyMTronCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
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
        EmbLayerNormVarSeqlenPluginLegacyMTron* p = new EmbLayerNormVarSeqlenPluginLegacyMTron(
            name, output_fp16 ? DataType::kHALF : DataType::kFLOAT, beta, gamma, word_emb, pos_emb, tok_emb);
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* EmbLayerNormVarSeqlenPluginLegacyHFaceCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call EmbLayerNormVarSeqlen::destroy()
        return new EmbLayerNormVarSeqlenPluginLegacyHFace(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* EmbLayerNormVarSeqlenPluginLegacyMTronCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call EmbLayerNormVarSeqlen::destroy()
        return new EmbLayerNormVarSeqlenPluginLegacyMTron(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void EmbLayerNormVarSeqlenPluginLegacyBaseCreator::setPluginNamespace(char const* libNamespace) noexcept
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

char const* EmbLayerNormVarSeqlenPluginLegacyBaseCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
