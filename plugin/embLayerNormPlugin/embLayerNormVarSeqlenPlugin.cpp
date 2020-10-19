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

#include <cuda.h>
#include <cstring>
#include <vector>

#include "NvInfer.h"
#include "embLayerNormVarSeqlenPlugin.h"
#include "serialize.hpp"

using namespace nvinfer1;

namespace bert
{
// For full mask mode, we must produce the compressed mask format expected by the fused attention path. Currently, only
// two sequence lengths are supported. We hard code the sizes here.
// The number of threads per CTA: warps_m * warps_n * warps_k * 32;
constexpr size_t threadsPerCta128 = 2 * 2 * 32;
constexpr size_t threadsPerCta256 = 1 * 4 * 32;
constexpr size_t threadsPerCta384 = 1 * 8 * 32;
// The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M dimension: (s + 16*warps_m - 1)
// / (16*warps_m);
constexpr size_t xmmasM128 = 4;
constexpr size_t xmmasM256 = 16;
constexpr size_t xmmasM384 = 24;
// Packed mask size per batch. Layout is XMMAS_M * THREADS_PER_CTA.
constexpr size_t packedMaskSize128 = xmmasM128 * threadsPerCta128;
constexpr size_t packedMaskSize256 = xmmasM256 * threadsPerCta256;
constexpr size_t packedMaskSize384 = xmmasM384 * threadsPerCta384;

namespace
{
static const char* EMB_LAYER_NORM_VAR_SEQLEN_VERSION{"2"};
static const char* EMB_LAYER_NORM_VAR_SEQLEN_NAME{"CustomEmbLayerNormPluginDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection EmbLayerNormVarSeqlenPluginCreator::mFC{};
std::vector<PluginField> EmbLayerNormVarSeqlenPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(EmbLayerNormVarSeqlenPluginCreator);

EmbLayerNormVarSeqlenPlugin::EmbLayerNormVarSeqlenPlugin(const std::string& name, const DataType type,
    const Weights& beta, const Weights& gamma, const Weights& wordEmb, const Weights& posEmb, const Weights& tokEmb)
    : mLayerName(name)
    , mLd(beta.count)
    , mType(type)
{
    // Assuming Weights.count is the number of elements and not bytes
    ASSERT(beta.count == gamma.count);
    ASSERT(wordEmb.count % mLd == 0);
    ASSERT(posEmb.count % mLd == 0);
    ASSERT(tokEmb.count % mLd == 0);
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

EmbLayerNormVarSeqlenPlugin::EmbLayerNormVarSeqlenPlugin(const std::string& name, const void* data, size_t length)
    : mLayerName(name)
    , mGammaDev(nullptr)
    , mBetaDev(nullptr)
    , mWordEmbDev(nullptr)
    , mTokEmbDev(nullptr)
    , mPosEmbDev(nullptr)
{
    gLogVerbose << "EmbLayerNormVarSeqlenPlugin deserialize\n";

    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mLd);
    deserialize_value(&data, &length, &mWordVocabSize);
    deserialize_value(&data, &length, &mPosVocabSize);
    deserialize_value(&data, &length, &mTokVocabSize);

    const char* d = static_cast<const char*>(data);
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
IPluginV2DynamicExt* EmbLayerNormVarSeqlenPlugin::clone() const
{
    gLogVerbose << "EmbLayerNormVarSeqlenPlugin clone\n";

    auto p = new EmbLayerNormVarSeqlenPlugin(mLayerName, mType, mBeta, mGamma, mWordEmb, mPosEmb, mTokEmb);
    p->setPluginNamespace(mNamespace.c_str());

    return p;
}

DimsExprs EmbLayerNormVarSeqlenPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder)
{
    // Input should be input ids and token ids and cumulative seqlens
    // Output should be the embeddings tensor and mask indices
    ASSERT(nbInputs == 4);

    ASSERT(inputs[0].nbDims == 1); // sum of all s
    ASSERT(inputs[0].nbDims == inputs[1].nbDims);

    ASSERT(inputs[2].nbDims == 1); // B+1

    ASSERT(outputIndex == 0 || outputIndex == 1);

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

    // This is a hack: we just report some mask size and rely the plugins to play nicely together.
    //      At runtime, depending on the actual maxSeqlen, the size might be different.
    int maskSize_ = packedMaskSize384;

    auto maskSize = exprBuilder.constant(maskSize_);
    auto fp16maskSize = exprBuilder.operation(DimensionOperation::kPROD, *maskSize, *exprBuilder.constant(2));

    auto Bplus1 = inputs[2].d[0];
    auto one = exprBuilder.constant(1);
    auto B = exprBuilder.operation(DimensionOperation::kSUB, *Bplus1, *one);

    DimsExprs ret;
    ret.nbDims = 2;
    ret.d[0] = B;
    ret.d[1] = fp16maskSize;
    return ret;
}

bool EmbLayerNormVarSeqlenPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    // The four inputs to this plugin input_ids, segment_ids, cu_seqlens and a dummy input with the
    // size of the max seq length in that order
    ASSERT(nbInputs == 4);
    // The two outputs of the plugin are embedding and the mask
    ASSERT(nbOutputs == 2);

    const PluginTensorDesc& desc = inOut[pos];
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }
    if (pos == 0 || pos == 2) // input_ids and cu_seqlens
    {
        return desc.type == DataType::kINT32 && desc.dims.nbDims == 1;
    }

    const PluginTensorDesc& prev = inOut[pos - 1];
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
    return desc.type == DataType::kHALF;
}

void EmbLayerNormVarSeqlenPlugin::configurePlugin(
    const DynamicPluginTensorDesc* inputs, int nbInputs, const DynamicPluginTensorDesc* outputs, int nbOutputs)
{
    gLogVerbose << "EmbLayerNormVarSeqlenPlugin configurePlugin\n";

    // Validate input arguments
    ASSERT(nbInputs == 4);
    ASSERT(nbOutputs == 2);

    ASSERT(inputs[0].desc.dims.nbDims == 1);
    ASSERT(inputs[1].desc.dims.nbDims == 1);

    ASSERT(inputs[1].desc.dims.d[0] == inputs[0].desc.dims.d[0]);

    ASSERT(inputs[2].desc.dims.nbDims == 1);

    ASSERT(outputs[0].desc.dims.nbDims == 4);
    ASSERT(static_cast<size_t>(outputs[0].desc.dims.d[0]) == static_cast<size_t>(inputs[0].desc.dims.d[0]));
    ASSERT(static_cast<size_t>(outputs[0].desc.dims.d[1]) == static_cast<size_t>(mLd));
    ASSERT(outputs[0].desc.dims.d[2] == 1);
    ASSERT(outputs[0].desc.dims.d[3] == 1);

    const int B = inputs[2].desc.dims.d[0] - 1;

    // check mask
    ASSERT(outputs[1].desc.dims.nbDims == 2);
    if (B > 0)
    {
        ASSERT(outputs[1].desc.dims.d[0] == B);
    }
    ASSERT((outputs[1].desc.dims.d[1] == 2 * packedMaskSize384) || (outputs[1].desc.dims.d[1] == 2 * packedMaskSize128)
        || (outputs[1].desc.dims.d[1] == 2 * packedMaskSize256));

    ASSERT(inputs[0].desc.type == DataType::kINT32);
    ASSERT(inputs[1].desc.type == DataType::kINT32);
    ASSERT(inputs[2].desc.type == DataType::kINT32);
    ASSERT(outputs[0].desc.type == mType);
    ASSERT(outputs[1].desc.type == DataType::kHALF);
}

size_t EmbLayerNormVarSeqlenPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const
{
    return 0;
}

int EmbLayerNormVarSeqlenPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    int status = -1;
    const int batchSize = inputDesc[2].dims.d[0] - 1;
    // read out the maximum sequence length from the dummy input
    const int maxSeqlen = inputDesc[3].dims.d[0];

    // There are four versions of the kernel which are optimized for sequence lengths 384, 256, 192 and 128.
    // Find the closest sequence length bigger than the max seq length in this batch.
    int S = 384;
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
    const auto inputIds = static_cast<const int*>(inputs[0]);
    const auto segmentIds = static_cast<const int*>(inputs[1]);
    const int* cuSeqlens = static_cast<const int*>(inputs[2]);

    const float* beta = mBetaDev.get();
    const float* gamma = mGammaDev.get();
    if (mType == DataType::kFLOAT)
    {
        auto output = static_cast<float*>(outputs[0]);
        const auto wordEmb = static_cast<const float*>(mWordEmbDev.get());
        const auto tokEmb = static_cast<const float*>(mTokEmbDev.get());
        const auto posEmb = static_cast<const float*>(mPosEmbDev.get());

        embSkipLayerNorm2<float>(stream, static_cast<int>(mLd), batchSize, S, inputIds, segmentIds, cuSeqlens, beta,
            gamma, wordEmb, posEmb, tokEmb, output);
    }
    else if (mType == DataType::kHALF)
    {
        auto output = static_cast<half*>(outputs[0]);
        const auto wordEmb = static_cast<const half*>(mWordEmbDev.get());
        const auto tokEmb = static_cast<const half*>(mTokEmbDev.get());
        const auto posEmb = static_cast<const half*>(mPosEmbDev.get());

        embSkipLayerNorm2<half>(stream, static_cast<int>(mLd), batchSize, S, inputIds, segmentIds, cuSeqlens, beta,
            gamma, wordEmb, posEmb, tokEmb, output);
    }
    else
    {
        gLogError << "Unsupported type error, expected [kHALF,kFLOAT], but received " << static_cast<int>(mType)
                  << std::endl;
        ASSERT(false);
    }

    CHECK(cudaPeekAtLastError());

    return status;
}

// IPluginV2Ext Methods
DataType EmbLayerNormVarSeqlenPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const
{

    ASSERT(index == 0 || index == 1);
    if (index == 0)
    {
        ASSERT(mType == DataType::kHALF || mType == DataType::kFLOAT);
        return mType;
    }
    return DataType::kHALF;
}

// IPluginV2 Methods
const char* EmbLayerNormVarSeqlenPlugin::getPluginType() const
{
    return EMB_LAYER_NORM_VAR_SEQLEN_NAME;
}

const char* EmbLayerNormVarSeqlenPlugin::getPluginVersion() const
{
    return EMB_LAYER_NORM_VAR_SEQLEN_VERSION;
}

int EmbLayerNormVarSeqlenPlugin::getNbOutputs() const
{
    return 2;
}

int EmbLayerNormVarSeqlenPlugin::initialize()
{
    return 0;
}

void EmbLayerNormVarSeqlenPlugin::terminate()
{
    gLogVerbose << "EmbLayerNormVarSeqlenPlugin terminate\n";
}

size_t EmbLayerNormVarSeqlenPlugin::getSerializationSize() const
{
    const size_t wordSize = getElementSize(mType);
    return 2 * sizeof(float) * mLd        // beta + gamma
        + sizeof(mType)                   //
        + sizeof(mLd)                     //
        + sizeof(mWordVocabSize)          //
        + sizeof(mPosVocabSize)           //
        + sizeof(mTokVocabSize)           //
        + wordSize * mLd * mWordVocabSize // word emb
        + wordSize * mLd * mPosVocabSize  // pos emb
        + wordSize * mLd * mTokVocabSize  // tok emb
        ;
}

void EmbLayerNormVarSeqlenPlugin::serialize(void* buffer) const
{
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mLd);
    serialize_value(&buffer, mWordVocabSize);
    serialize_value(&buffer, mPosVocabSize);
    serialize_value(&buffer, mTokVocabSize);

    char* d = static_cast<char*>(buffer);
    const size_t wordSize = getElementSize(mType);

    serFromDev(d, mBetaDev.get(), mLd);
    serFromDev(d, mGammaDev.get(), mLd);
    serFromDev(d, static_cast<char*>(mWordEmbDev.get()), mLd * mWordVocabSize * wordSize);
    serFromDev(d, static_cast<char*>(mPosEmbDev.get()), mLd * mPosVocabSize * wordSize);
    serFromDev(d, static_cast<char*>(mTokEmbDev.get()), mLd * mTokVocabSize * wordSize);
}

void EmbLayerNormVarSeqlenPlugin::destroy()
{
    gLogVerbose << "EmbLayerNormVarSeqlenPlugin destroy\n";
    // This gets called when the network containing plugin is destroyed
    mGammaDev.release();
    mBetaDev.release();
    mWordEmbDev.release();
    mPosEmbDev.release();
    mTokEmbDev.release();
    delete this;
}

void EmbLayerNormVarSeqlenPlugin::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* EmbLayerNormVarSeqlenPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

///////////////////////

EmbLayerNormVarSeqlenPluginCreator::EmbLayerNormVarSeqlenPluginCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* EmbLayerNormVarSeqlenPluginCreator::getPluginName() const
{
    return EMB_LAYER_NORM_VAR_SEQLEN_NAME;
}

const char* EmbLayerNormVarSeqlenPluginCreator::getPluginVersion() const
{
    return EMB_LAYER_NORM_VAR_SEQLEN_VERSION;
}

const PluginFieldCollection* EmbLayerNormVarSeqlenPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* EmbLayerNormVarSeqlenPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    gLogVerbose << "EmbLayerNormVarSeqlen createPlugin\n";

    bool output_fp16 = false;
    Weights beta;
    Weights gamma;
    Weights word_emb;
    Weights pos_emb;
    Weights tok_emb;
    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);
        if (field_name.compare("bert_embeddings_layernorm_beta") == 0)
        {
            gLogVerbose << "Building bert_embeddings_layernorm_beta...\n";
            beta.values = fc->fields[i].data;
            beta.count = fc->fields[i].length;
            beta.type = fieldTypeToDataType(fc->fields[i].type);
        }

        if (field_name.compare("bert_embeddings_layernorm_gamma") == 0)
        {
            gLogVerbose << "Building bert_embeddings_layernorm_gamma...\n";
            gamma.values = fc->fields[i].data;
            gamma.count = fc->fields[i].length;
            gamma.type = fieldTypeToDataType(fc->fields[i].type);
        }

        if (field_name.compare("bert_embeddings_word_embeddings") == 0)
        {
            gLogVerbose << "Building bert_embeddings_word_embeddings...\n";
            word_emb.values = fc->fields[i].data;
            word_emb.count = fc->fields[i].length;
            word_emb.type = fieldTypeToDataType(fc->fields[i].type);
        }

        if (field_name.compare("bert_embeddings_token_type_embeddings") == 0)
        {
            gLogVerbose << "Building bert_embeddings_token_type_embeddings...\n";
            tok_emb.values = fc->fields[i].data;
            tok_emb.count = fc->fields[i].length;
            tok_emb.type = fieldTypeToDataType(fc->fields[i].type);
        }

        if (field_name.compare("bert_embeddings_position_embeddings") == 0)
        {
            gLogVerbose << "Building bert_embeddings_position_embeddings...\n";
            pos_emb.values = fc->fields[i].data;
            pos_emb.count = fc->fields[i].length;
            pos_emb.type = fieldTypeToDataType(fc->fields[i].type);
        }
        if (field_name.compare("output_fp16") == 0)
        {
            gLogVerbose << "Building output_fp16...\n";
            ASSERT(fc->fields[i].type == PluginFieldType::kINT32);
            output_fp16 = static_cast<const int*>(fc->fields[i].data)[0] != 0;
        }
    }

    gLogVerbose << "Building the Plugin...\n";
    EmbLayerNormVarSeqlenPlugin* p = new EmbLayerNormVarSeqlenPlugin(
        name, output_fp16 ? DataType::kHALF : DataType::kFLOAT, beta, gamma, word_emb, pos_emb, tok_emb);
    return p;
}

IPluginV2* EmbLayerNormVarSeqlenPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call EmbLayerNormVarSeqlen::destroy()
    return new EmbLayerNormVarSeqlenPlugin(name, serialData, serialLength);
}

void EmbLayerNormVarSeqlenPluginCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* EmbLayerNormVarSeqlenPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
} // namespace bert
