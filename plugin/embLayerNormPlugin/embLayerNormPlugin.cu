/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#include <cassert>
#include <cstring>
#include <vector>

#include "NvInfer.h"
#include "embLayerNormPlugin.h"
#include "bertCommon.h"
#include "common.h"
#include "serialize.hpp"

using namespace nvinfer1;
using bert::operator+;

namespace bert
{

template <unsigned TPB>
__global__ void maskIdxKernelSmall(int ld, const int* mask, int* maskIdx)
{

    using BlockReduce = cub::BlockReduce<int, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    // ld is S
    // blockIdx.x is b

    const int offset = blockIdx.x * ld; // batch strides of S

    cub::Min min;
    int threadData(ld); // if the mask admits all values

    const int idx = offset + threadIdx.x;
    if (threadIdx.x < ld)
    {
        const int val = mask[idx];
        if (val == 0) // masked position: report thread idx
        {
            threadData = threadIdx.x;
        }
    }

    const auto minIdx = BlockReduce(tmpStorage).Reduce(threadData, min);

    if (threadIdx.x == 0)
    {
        maskIdx[blockIdx.x] = minIdx;
    }
}

template <unsigned TPB>
__global__ void maskIdxKernel(int ld, const int* mask, int* maskIdx)
{

    using BlockReduce = cub::BlockReduce<int, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    // ld is S
    // blockIdx.x is b

    const int offset = blockIdx.x * ld; // batch strides of S

    cub::Min min;
    int threadData(ld); // if the mask admits all values

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        const int idx = offset + i;
        const int val = mask[idx];
        if (val == 0) // masked position: report thread idx
        {
            threadData = min(threadData, i);
        }
    }

    const auto minIdx = BlockReduce(tmpStorage).Reduce(threadData, min);

    if (threadIdx.x == 0)
    {
        maskIdx[blockIdx.x] = minIdx;
    }
}

inline int computeMaskIdx(cudaStream_t stream, const int S, const int B, const int* mask, int* maskIdx)
{
    // Mask idx is of length B and assumes the valid region is contiguous starting
    // from the beginning of the sequence

    // Assume n = BxS
    if (S <= 32)
    {
        maskIdxKernelSmall<32><<<B, 32, 0, stream>>>(S, mask, maskIdx);
    }
    else if (S <= 128)
    {
        maskIdxKernelSmall<128><<<B, 128, 0, stream>>>(S, mask, maskIdx);
    }
    else if (S == 384)
    {
        maskIdxKernelSmall<384><<<B, 384, 0, stream>>>(S, mask, maskIdx);
    }
    else
    {
        maskIdxKernel<256><<<B, 256, 0, stream>>>(S, mask, maskIdx);
    }

    CHECK(cudaPeekAtLastError());

    return 0;
}

template <typename T, unsigned TPB>
__global__ void embLayerNormKernel(int ld, const int* inputIds, const int* tokenIds, const float* beta,
    const float* gamma, const T* wordEmb, const T* posEmb, const T* tokEmb, T* output)
{

    cub::Sum pairSum;
    // 1. lookup word and token of the block
    // blockIdx.x = position in the sequence
    // blockIdx.y = batch
    // gridDim.x = S
    // gridDim.y = B
    __shared__ int wordId;
    __shared__ int tokenId;

    const T rld = T(1.f) / T(ld);
    const int seqPos = blockIdx.y + blockIdx.x * gridDim.y;
    if (threadIdx.x == 0)
    {
        wordId = inputIds[seqPos];
        tokenId = tokenIds[seqPos];
    }
    __syncthreads();

    // 2. load pos/tok/word embeddings and add them toghether
    // offset into embeddings is given by wordId * hidden_size
    const int poffset = blockIdx.x * ld;
    const int woffset = wordId * ld;
    const int toffset = tokenId * ld;
    // the output offset is given by b * (S*hidden_size) + s * hidden_size
    const int outOffset = seqPos * ld;

    kvp<T> threadData(0, 0);

    for (int it = threadIdx.x; it < ld; it += TPB)
    {
        const T w(wordEmb[woffset + it]);
        const T t(tokEmb[toffset + it]);
        const T p(posEmb[poffset + it]);
        const T val = w + t + p;

        output[outOffset + it] = val;
        const T rldval = rld * val;
        threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
    }

    // 3. layer norm on the sum
    layerNorm<T, T, TPB>(threadData, ld, outOffset, beta, gamma, output);
}

template <typename T>
inline int embSkipLayerNorm(cudaStream_t stream, int ld, int B, int S, const int* inputIds, const int* token_ids,
    const float* beta, const float* gamma, const T* wordEmb, const T* posEmb, const T* tokEmb, T* output)
{

    constexpr int tpb = 256;
    const dim3 grid(S, B, 1);
    const dim3 block(tpb, 1, 1);

    embLayerNormKernel<T, tpb>
        <<<grid, block, 0, stream>>>(ld, inputIds, token_ids, beta, gamma, wordEmb, posEmb, tokEmb, output);
    CHECK(cudaPeekAtLastError());

    return 0;
}

// Clip plugin specific constants
namespace
{
static const char* EMB_LAYER_NORM_VERSION{"1"};
static const char* EMB_LAYER_NORM_NAME{"CustomEmbLayerNormPluginDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection EmbLayerNormPluginDynamicCreator::mFC{};
std::vector<PluginField> EmbLayerNormPluginDynamicCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(EmbLayerNormPluginDynamicCreator);

EmbLayerNormPluginDynamic::EmbLayerNormPluginDynamic(const std::string& name, const bool outputFp16,
    const Weights& beta, const Weights& gamma, const Weights& wordEmb, const Weights& posEmb, const Weights& tokEmb)
    : mLayerName(name)
    , mLd(beta.count)
    , mGamma(gamma)
    , mBeta(beta)
    , mWordEmb(wordEmb)
    , mPosEmb(posEmb)
    , mTokEmb(tokEmb)
    , mGammaDev(nullptr)
    , mBetaDev(nullptr)
    , mWordEmbDev(nullptr)
    , mTokEmbDev(nullptr)
    , mPosEmbDev(nullptr)
{
    // Assuming Weights.count is the number of elements and not bytes
    assert(beta.count == gamma.count);
    assert(wordEmb.count % mLd == 0);
    assert(posEmb.count % mLd == 0);
    assert(tokEmb.count % mLd == 0);
    mWordVocabSize = wordEmb.count / mLd;
    mPosVocabSize = posEmb.count / mLd;
    mTokVocabSize = tokEmb.count / mLd;
    // We set mS in configure
    mType = outputFp16 ? DataType::kHALF : DataType::kFLOAT;
}

EmbLayerNormPluginDynamic::EmbLayerNormPluginDynamic(const std::string& name, const void* data, size_t length)
    : mLayerName(name)
{
    gLogVerbose << "EMB LN Deser start\n";
    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mLd);
    deserialize_value(&data, &length, &mS);
    deserialize_value(&data, &length, &mWordVocabSize);
    deserialize_value(&data, &length, &mPosVocabSize);
    deserialize_value(&data, &length, &mTokVocabSize);

    const char* d = static_cast<const char*>(data);
    mBetaDev = deserToDev<float>(d, mLd);
    mGammaDev = deserToDev<float>(d, mLd);

    const size_t wordSize = samplesCommon::getElementSize(mType);
    mWordEmbDev = deserToDev<char>(d, mLd * mWordVocabSize * wordSize);
    mPosEmbDev = deserToDev<char>(d, mLd * mPosVocabSize * wordSize);
    mTokEmbDev = deserToDev<char>(d, mLd * mTokVocabSize * wordSize);
    // this signals init not to allocate/copy
    mGamma.count = -1;
    mBeta.count = -1;
    mWordEmb.count = -1;
    mTokEmb.count = -1;
    mPosEmb.count = -1;
    mGamma.values = nullptr;
    mBeta.values = nullptr;
    mWordEmb.values = nullptr;
    mTokEmb.values = nullptr;
    mPosEmb.values = nullptr;

    gLogVerbose << "EMB LN Deser done\n";
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* EmbLayerNormPluginDynamic::clone() const
{
    gLogVerbose << "EMBLN clone start" << std::endl;
    auto ret = new EmbLayerNormPluginDynamic(
        mLayerName, mType == DataType::kHALF, mBeta, mGamma, mWordEmb, mPosEmb, mTokEmb);
    ret->mS = mS;

    ret->mWordEmbDev = mWordEmbDev;
    ret->mPosEmbDev = mPosEmbDev;
    ret->mTokEmbDev = mTokEmbDev;
    ret->mBetaDev = mBetaDev;
    ret->mGammaDev = mGammaDev;
    gLogVerbose << "EMBLN clone done" << std::endl;
    return ret;
}

DimsExprs EmbLayerNormPluginDynamic::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder)
{
    // Input should be input ids and token ids and the input mask
    // Output should be the embeddings tensor and mask indices
    assert(nbInputs == 3);

    assert(inputs[0].nbDims == 2); // BxS
    assert(inputs[0].nbDims == inputs[1].nbDims);
    assert(inputs[0].nbDims == inputs[2].nbDims);

    assert(outputIndex == 0 || outputIndex == 1);

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
    ret.nbDims = 1;
    ret.d[0] = inputs[0].d[BDIM];
    return ret;
}

bool EmbLayerNormPluginDynamic::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    // 3 inputs of size BxS
    assert(nbInputs == 3);
    assert(nbOutputs == 2);

    const PluginTensorDesc& desc = inOut[pos];
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }
    if (pos == 0)
    {
        return desc.type == DataType::kINT32 && desc.dims.nbDims == 2;
    }

    const PluginTensorDesc& prev = inOut[pos - 1];
    if (pos == 1 || pos == 2)
    {
        return desc.type == DataType::kINT32 && desc.dims.nbDims == 2 && desc.dims.d[BDIM] == prev.dims.d[BDIM]
            && desc.dims.d[SDIM] == prev.dims.d[SDIM];
    }

    if (pos == 3)
    { // embedded sequence

        return desc.type == mType && desc.dims.nbDims == 5 && desc.dims.d[BDIM] == prev.dims.d[BDIM]
            && desc.dims.d[SDIM] == prev.dims.d[SDIM] && desc.dims.d[3] == 1 && desc.dims.d[4] == 1;
    }

    // pos == 4: mask
    return desc.type == DataType::kINT32 && desc.dims.nbDims == 1 && desc.dims.d[0] == prev.dims.d[1];
}

void EmbLayerNormPluginDynamic::configurePlugin(
    const DynamicPluginTensorDesc* inputs, int nbInputs, const DynamicPluginTensorDesc* outputs, int nbOutputs)
{
    // Validate input arguments
    assert(nbOutputs == 2);
    assert(nbInputs == 3);

    assert(inputs[0].desc.dims.nbDims == 2);
    mS = inputs[0].desc.dims.d[SDIM];
    const int B = inputs[0].desc.dims.d[BDIM];
    TRT_UNUSED B;
    assert(mS == inputs[1].desc.dims.d[SDIM]);
    assert(B == inputs[1].desc.dims.d[BDIM]);
    assert(mS == inputs[2].desc.dims.d[SDIM]);
    assert(B == inputs[2].desc.dims.d[BDIM]);

    assert(outputs[0].desc.dims.nbDims == 5);
    assert(outputs[0].desc.dims.d[SDIM] == mS);
    assert(outputs[0].desc.dims.d[BDIM] == B);
    assert(outputs[0].desc.dims.d[2] == mLd);
    assert(outputs[0].desc.dims.d[3] == 1);
    assert(outputs[0].desc.dims.d[4] == 1);

    assert(outputs[1].desc.dims.nbDims == 1);
    assert(outputs[1].desc.dims.d[0] == B);

    assert(inputs[0].desc.type == DataType::kINT32);
    assert(inputs[1].desc.type == DataType::kINT32);
    assert(inputs[2].desc.type == DataType::kINT32);
    assert(outputs[0].desc.type == DataType::kFLOAT || outputs[0].desc.type == DataType::kHALF);
    assert(outputs[1].desc.type == DataType::kINT32);
}

size_t EmbLayerNormPluginDynamic::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const
{
    return 0;
}

int EmbLayerNormPluginDynamic::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    const int batchSize = inputDesc->dims.d[BDIM];
    const int S = inputDesc->dims.d[SDIM];
    int status = -1;

    // Our plugin outputs only one tensor
    const int* inputIds = static_cast<const int*>(inputs[0]);
    const int* segmentIds = static_cast<const int*>(inputs[1]);
    const int* inputMask = static_cast<const int*>(inputs[2]);

    if (mType == DataType::kFLOAT)
    {
        float* output = static_cast<float*>(outputs[0]);
        float* wordEmb = static_cast<float*>(mWordEmbDev);
        float* tokEmb = static_cast<float*>(mTokEmbDev);
        float* posEmb = static_cast<float*>(mPosEmbDev);
        embSkipLayerNorm<float>(
            stream, mLd, batchSize, S, inputIds, segmentIds, mBetaDev, mGammaDev, wordEmb, posEmb, tokEmb, output);
    }
    else if (mType == DataType::kHALF)
    {
        half* output = static_cast<half*>(outputs[0]);

        half* wordEmb = static_cast<half*>(mWordEmbDev);
        half* tokEmb = static_cast<half*>(mTokEmbDev);
        half* posEmb = static_cast<half*>(mPosEmbDev);
        embSkipLayerNorm<half>(
            stream, mLd, batchSize, S, inputIds, segmentIds, mBetaDev, mGammaDev, wordEmb, posEmb, tokEmb, output);
    }
    else
    {
        assert(false);
    }
    int* maskIdx = static_cast<int*>(outputs[1]);
    computeMaskIdx(stream, S, batchSize, inputMask, maskIdx);

    return status;
}

// IPluginV2Ext Methods
DataType EmbLayerNormPluginDynamic::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const
{

    assert(index == 0 || index == 1);
    if (index == 0)
    {
        assert(mType == DataType::kHALF || mType == DataType::kFLOAT);
        return mType;
    }
    return DataType::kINT32;
}

// IPluginV2 Methods
const char* EmbLayerNormPluginDynamic::getPluginType() const
{
    return EMB_LAYER_NORM_NAME;
}

const char* EmbLayerNormPluginDynamic::getPluginVersion() const
{
    return EMB_LAYER_NORM_VERSION;
}

int EmbLayerNormPluginDynamic::getNbOutputs() const
{
    return 2;
}

int EmbLayerNormPluginDynamic::initialize()
{
    if (mGamma.values)
    {
        CHECK(cudaMalloc(&mGammaDev, sizeof(float) * mGamma.count));
        CHECK(cudaMemcpy(mGammaDev, mGamma.values, sizeof(float) * mGamma.count, cudaMemcpyHostToDevice));
    }
    if (mBeta.values)
    {
        CHECK(cudaMalloc(&mBetaDev, sizeof(float) * mBeta.count));
        CHECK(cudaMemcpy(mBetaDev, mBeta.values, sizeof(float) * mBeta.count, cudaMemcpyHostToDevice));
    }
    const size_t wordSize = samplesCommon::getElementSize(mType);

    if (mWordEmb.values)
    {
        CHECK(cudaMalloc(&mWordEmbDev, wordSize * mWordEmb.count));
        if (mType == DataType::kFLOAT)
        {
            convertAndCopyToDevice(mWordEmb, reinterpret_cast<float*>(mWordEmbDev));
        }
        else
        {
            convertAndCopyToDevice(mWordEmb, reinterpret_cast<half*>(mWordEmbDev));
        }
    }
    if (mTokEmb.values)
    {
        CHECK(cudaMalloc(&mTokEmbDev, wordSize * mTokEmb.count));
        if (mType == DataType::kFLOAT)
        {
            convertAndCopyToDevice(mTokEmb, reinterpret_cast<float*>(mTokEmbDev));
        }
        else
        {
            convertAndCopyToDevice(mTokEmb, reinterpret_cast<half*>(mTokEmbDev));
        }
    }

    if (mPosEmb.values)
    {
        CHECK(cudaMalloc(&mPosEmbDev, wordSize * mPosEmb.count));
        if (mType == DataType::kFLOAT)
        {
            convertAndCopyToDevice(mPosEmb, reinterpret_cast<float*>(mPosEmbDev));
        }
        else
        {
            convertAndCopyToDevice(mPosEmb, reinterpret_cast<half*>(mPosEmbDev));
        }
    }
    return 0;
}

void EmbLayerNormPluginDynamic::terminate()
{
    gLogVerbose << "EMBLN terminate start" << std::endl;
    CHECK(cudaFree(mGammaDev));
    CHECK(cudaFree(mBetaDev));
    CHECK(cudaFree(mWordEmbDev));
    CHECK(cudaFree(mTokEmbDev));
    CHECK(cudaFree(mPosEmbDev));
    gLogVerbose << "EMBLN terminate done" << std::endl;
}

size_t EmbLayerNormPluginDynamic::getSerializationSize() const
{
    const size_t wordSize = samplesCommon::getElementSize(mType);
    return 2 * sizeof(float) * mLd        // beta + gamma
        + sizeof(mType) + sizeof(mLd) * 5 //mLd, mS, m*VocabSize
        + wordSize * mLd * mWordVocabSize // word emb
        + wordSize * mLd * mPosVocabSize  // pos emb
        + wordSize * mLd * mTokVocabSize  // tok emb
        ;
}

void EmbLayerNormPluginDynamic::serialize(void* buffer) const
{
    const size_t wordSize = samplesCommon::getElementSize(mType);

    serialize_value(&buffer, mType);
    serialize_value(&buffer, mLd);
    serialize_value(&buffer, mS);
    serialize_value(&buffer, mWordVocabSize);
    serialize_value(&buffer, mPosVocabSize);
    serialize_value(&buffer, mTokVocabSize);

    char* d = static_cast<char*>(buffer);
    serFromDev(d, mBetaDev, mLd);
    serFromDev(d, mGammaDev, mLd);
    serFromDev(d, static_cast<char*>(mWordEmbDev), mLd * mWordVocabSize * wordSize);
    serFromDev(d, static_cast<char*>(mPosEmbDev), mLd * mPosVocabSize * wordSize);
    serFromDev(d, static_cast<char*>(mTokEmbDev), mLd * mTokVocabSize * wordSize);
}

void EmbLayerNormPluginDynamic::destroy()
{
    gLogVerbose << "EMBLN destroy start" << std::endl;
    // This gets called when the network containing plugin is destroyed
    delete this;
    gLogVerbose << "EMBLN destroy start" << std::endl;
}

void EmbLayerNormPluginDynamic::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* EmbLayerNormPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

///////////////////////

EmbLayerNormPluginDynamicCreator::EmbLayerNormPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* EmbLayerNormPluginDynamicCreator::getPluginName() const
{
    return EMB_LAYER_NORM_NAME;
}

const char* EmbLayerNormPluginDynamicCreator::getPluginVersion() const
{
    return EMB_LAYER_NORM_VERSION;
}

const PluginFieldCollection* EmbLayerNormPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* EmbLayerNormPluginDynamicCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    gLogVerbose << "Creating EmbLayerNormPluginDynamic...\n";

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
            assert(fc->fields[i].type == PluginFieldType::kINT32);
            output_fp16 = reinterpret_cast<const int*>(fc->fields[i].data)[0] != 0;
        }
    }

    gLogVerbose << "Building the Plugin...\n";
    EmbLayerNormPluginDynamic* p
        = new EmbLayerNormPluginDynamic(name, output_fp16, beta, gamma, word_emb, pos_emb, tok_emb);
    return p;
}

IPluginV2* EmbLayerNormPluginDynamicCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call EmbLayerNormPluginDynamic::destroy()
    return new EmbLayerNormPluginDynamic(name, serialData, serialLength);
}

void EmbLayerNormPluginDynamicCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* EmbLayerNormPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
}
