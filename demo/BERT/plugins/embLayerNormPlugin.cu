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

#include "NvInfer.h"
#include "embLayerNormPlugin.h"
#include "logger.h"
#include "pluginKernels.h"
#include "pluginUtil.h"

#include <cassert>
#include <cstring>
#include <vector>

using bert::operator+;

namespace bert
{

template <typename T, unsigned TPB>
__global__ void embLayerNormKernel(int ld, const int* inputIds, const int* tokenIds, const float* beta,
    const float* gamma, const float* wordEmb, const float* posEmb, const float* tokEmb, T* output)
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
    const int seqPos = blockIdx.y * gridDim.x + blockIdx.x;
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
    layerNorm<T, TPB>(threadData, ld, outOffset, beta, gamma, output);
}

template <typename T>
int embSkipLayerNorm(cudaStream_t stream, int ld, int B, int S, const int* inputIds, const int* token_ids,
    const float* beta, const float* gamma, const float* wordEmb, const float* posEmb, const float* tokEmb, T* output)
{

    constexpr int tpb = 256;
    const dim3 grid(S, B, 1);
    const dim3 block(tpb, 1, 1);

    embLayerNormKernel<T, tpb>
        <<<grid, block, 0, stream>>>(ld, inputIds, token_ids, beta, gamma, wordEmb, posEmb, tokEmb, output);
    CHECK(cudaPeekAtLastError());

    return 0;
}

using namespace nvinfer1;

// Clip plugin specific constants
namespace
{
static const char* EMB_LAYER_NORM_VERSION{"1"};
static const char* EMB_LAYER_NORM_NAME{"CustomEmbLayerNormPlugin"};
} // namespace

// Static class fields initialization
PluginFieldCollection EmbLayerNormPluginCreator::mFC{};
std::vector<PluginField> EmbLayerNormPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(EmbLayerNormPluginCreator);

EmbLayerNormPlugin::EmbLayerNormPlugin(const std::string& name, const bool outputFp16, const Weights& beta,
    const Weights& gamma, const Weights& wordEmb, const Weights& posEmb, const Weights& tokEmb)
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

EmbLayerNormPlugin::EmbLayerNormPlugin(const std::string& name, const void* data, size_t length)
    : mLayerName(name)
{
    gLogVerbose << "EMB LN Deser start\n";
    // Deserialize in the same order as serialization
    const char* d = static_cast<const char*>(data);
    const char* a = d;
    DESER(d, mType);
    DESER(d, mLd);
    DESER(d, mS);
    DESER(d, mWordVocabSize);
    DESER(d, mPosVocabSize);
    DESER(d, mTokVocabSize);
    mBetaDev = deserToDev<float>(d, mLd);
    mGammaDev = deserToDev<float>(d, mLd);

    mWordEmbDev = deserToDev<float>(d, mLd * mWordVocabSize);
    mPosEmbDev = deserToDev<float>(d, mLd * mPosVocabSize);
    mTokEmbDev = deserToDev<float>(d, mLd * mTokVocabSize);
    assert(d == (a + length));
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

const char* EmbLayerNormPlugin::getPluginType() const
{
    return EMB_LAYER_NORM_NAME;
}

const char* EmbLayerNormPlugin::getPluginVersion() const
{
    return EMB_LAYER_NORM_VERSION;
}

int EmbLayerNormPlugin::getNbOutputs() const
{
    return 2;
}

DataType EmbLayerNormPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    assert(index == 0 || index == 1);
    if (index == 0)
    {
        assert(mType == DataType::kHALF || mType == DataType::kFLOAT);
        return mType;
    }
    return DataType::kINT32;
}

Dims EmbLayerNormPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Input should be input ids and token ids and the input mask
    // Output should be the embeddings tensor and mask indices
    assert(nbInputDims == 3);
    assert(inputs[0].nbDims == 1); // S
    assert(inputs[0].nbDims == inputs[1].nbDims);
    const int S = inputs[0].d[0];
    assert(inputs[1].d[0] == S);
    assert(inputs[2].d[0] == S);

    assert(index == 0 || index == 1);

    if (index == 0)
    {
        const int hidden_size = mLd;
        return Dims4{S, hidden_size, 1, 1};
    }
    return Dims{1, 1};
}

int EmbLayerNormPlugin::initialize()
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

    if (mWordEmb.values)
    {
        CHECK(cudaMalloc(&mWordEmbDev, sizeof(float) * mWordEmb.count));
        CHECK(cudaMemcpy(mWordEmbDev, mWordEmb.values, sizeof(float) * mWordEmb.count, cudaMemcpyHostToDevice));
    }
    if (mTokEmb.values)
    {
        CHECK(cudaMalloc(&mTokEmbDev, sizeof(float) * mTokEmb.count));
        CHECK(cudaMemcpy(mTokEmbDev, mTokEmb.values, sizeof(float) * mTokEmb.count, cudaMemcpyHostToDevice));
    }

    if (mPosEmb.values)
    {
        CHECK(cudaMalloc(&mPosEmbDev, sizeof(float) * mPosEmb.count));
        CHECK(cudaMemcpy(mPosEmbDev, mPosEmb.values, sizeof(float) * mPosEmb.count, cudaMemcpyHostToDevice));
    }
    return 0;
}

int EmbLayerNormPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    int status = -1;

    // Our plugin outputs only one tensor
    const int* inputIds = static_cast<const int*>(inputs[0]);
    const int* segmentIds = static_cast<const int*>(inputs[1]);
    const int* inputMask = static_cast<const int*>(inputs[2]);

    if (mType == DataType::kFLOAT)
    {
        float* output = static_cast<float*>(outputs[0]);
        embSkipLayerNorm<float>(stream, mLd, batchSize, mS, inputIds, segmentIds, mBetaDev, mGammaDev, mWordEmbDev,
            mPosEmbDev, mTokEmbDev, output);
    }
    else if (mType == DataType::kHALF)
    {
        half* output = static_cast<half*>(outputs[0]);
        embSkipLayerNorm<half>(stream, mLd, batchSize, mS, inputIds, segmentIds, mBetaDev, mGammaDev, mWordEmbDev,
            mPosEmbDev, mTokEmbDev, output);
    }
    else
    {
        assert(false);
    }
    int* maskIdx = static_cast<int*>(outputs[1]);
    computeMaskIdx(stream, mS, batchSize, inputMask, maskIdx);

    return status;
}

size_t EmbLayerNormPlugin::getSerializationSize() const
{
    return 2 * sizeof(float) * mLd             // beta + gamma
        + sizeof(mType) + sizeof(mLd) * 5      //mLd, mS, m*VocabSize
        + sizeof(float) * mLd * mWordVocabSize // word emb
        + sizeof(float) * mLd * mPosVocabSize  // pos emb
        + sizeof(float) * mLd * mTokVocabSize  // tok emb
        ;
}

void EmbLayerNormPlugin::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer);
    const char* a = d;
    writeToBuffer(d, mType);
    writeToBuffer(d, mLd);
    writeToBuffer(d, mS);
    writeToBuffer(d, mWordVocabSize);
    writeToBuffer(d, mPosVocabSize);
    writeToBuffer(d, mTokVocabSize);
    serFromDev(d, mBetaDev, mLd);
    serFromDev(d, mGammaDev, mLd);
    serFromDev(d, mWordEmbDev, mLd * mWordVocabSize);
    serFromDev(d, mPosEmbDev, mLd * mPosVocabSize);
    serFromDev(d, mTokEmbDev, mLd * mTokVocabSize);

    assert(d == a + getSerializationSize());
}

void EmbLayerNormPlugin::configurePlugin(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat format, int maxBatchSize)
{

    // Validate input arguments
    assert(nbOutputs == 2);
    assert(nbInputs == 3);

    assert(inputs[0].nbDims == 1);
    mS = inputs[0].d[0];
    assert(mS == inputs[1].d[0]);
    assert(mS == inputs[2].d[0]);

    assert(outputs[0].nbDims == 4);
    assert(outputs[0].d[0] == mS);
    assert(outputs[0].d[1] == mLd);
    assert(outputs[0].d[2] == 1);
    assert(outputs[0].d[3] == 1);

    assert(outputs[1].nbDims == 1);
    assert(outputs[1].d[0] == 1);

    assert(format == PluginFormat::kNCHW);
    assert(inputTypes[0] == DataType::kINT32);
    assert(inputTypes[1] == DataType::kINT32);
    assert(inputTypes[2] == DataType::kINT32);
    const DataType out_type = outputTypes[0];
    assert(out_type == DataType::kFLOAT || out_type == DataType::kHALF);
    assert(outputTypes[1] == DataType::kINT32);
}

bool EmbLayerNormPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    if (type == DataType::kINT32 || type == DataType::kFLOAT || type == DataType::kHALF)
    {
        return format == PluginFormat::kNCHW;
    }
    else
    {
        return false;
    }
}

void EmbLayerNormPlugin::terminate()
{
    gLogVerbose << "EMBLN terminate start" << std::endl;
    cudaFree(mGammaDev);
    cudaFree(mBetaDev);
    cudaFree(mWordEmbDev);
    cudaFree(mTokEmbDev);
    cudaFree(mPosEmbDev);
    gLogVerbose << "EMBLN terminate done" << std::endl;
}

void EmbLayerNormPlugin::destroy()
{
    gLogVerbose << "EMBLN destroy start" << std::endl;
    // This gets called when the network containing plugin is destroyed
    delete this;
    gLogVerbose << "EMBLN destroy start" << std::endl;
}

IPluginV2Ext* EmbLayerNormPlugin::clone() const
{
    gLogVerbose << "EMBLN clone start" << std::endl;
    auto ret = new EmbLayerNormPlugin(mLayerName, mType == DataType::kHALF, mBeta, mGamma, mWordEmb, mPosEmb, mTokEmb);
    ret->mS = mS;

    ret->mWordEmbDev = mWordEmbDev;
    ret->mPosEmbDev = mPosEmbDev;
    ret->mTokEmbDev = mTokEmbDev;
    ret->mBetaDev = mBetaDev;
    ret->mGammaDev = mGammaDev;
    gLogVerbose << "EMBLN clone done" << std::endl;
    return ret;
}

void EmbLayerNormPlugin::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* EmbLayerNormPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

EmbLayerNormPluginCreator::EmbLayerNormPluginCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* EmbLayerNormPluginCreator::getPluginName() const
{
    return EMB_LAYER_NORM_NAME;
}

const char* EmbLayerNormPluginCreator::getPluginVersion() const
{
    return EMB_LAYER_NORM_VERSION;
}

const PluginFieldCollection* EmbLayerNormPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* EmbLayerNormPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    gLogVerbose << "Creating EmbLayerNormPlugin...\n";

    bool output_fp16 = true;
    Weights beta;
    Weights gamma;
    Weights word_emb;
    Weights pos_emb;
    Weights tok_emb;
    for(int i=0; i< fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);
        if (field_name.compare("bert_embeddings_layernorm_beta")==0)
        {
            gLogVerbose << "Building bert_embeddings_layernorm_beta...\n";
            beta.values = fc->fields[i].data;
            beta.count = fc->fields[i].length;
            beta.type = static_cast<DataType>(fc->fields[i].type);
        }

        if (field_name.compare("bert_embeddings_layernorm_gamma")==0)
        {
            gLogVerbose << "Building bert_embeddings_layernorm_gamma...\n";
            gamma.values = fc->fields[i].data;
            gamma.count = fc->fields[i].length;
            gamma.type = static_cast<DataType>(fc->fields[i].type);
        }

        if (field_name.compare("bert_embeddings_word_embeddings")==0)
        {
            gLogVerbose << "Building bert_embeddings_word_embeddings...\n";
            word_emb.values = fc->fields[i].data;
            word_emb.count = fc->fields[i].length;
            word_emb.type = static_cast<DataType>(fc->fields[i].type);
        }

        if (field_name.compare("bert_embeddings_token_type_embeddings")==0)
        {
            gLogVerbose << "Building bert_embeddings_token_type_embeddings...\n";
            tok_emb.values = fc->fields[i].data;
            tok_emb.count = fc->fields[i].length;
            tok_emb.type = static_cast<DataType>(fc->fields[i].type);
        }

        if (field_name.compare("bert_embeddings_position_embeddings")==0)
        {
            gLogVerbose << "Building bert_embeddings_position_embeddings...\n";
            pos_emb.values = fc->fields[i].data;
            pos_emb.count = fc->fields[i].length;
            pos_emb.type = static_cast<DataType>(fc->fields[i].type);
        }
    }

    gLogVerbose << "Building the Plugin...\n";
    EmbLayerNormPlugin* p =  new EmbLayerNormPlugin(name, output_fp16, beta, gamma, word_emb, pos_emb, tok_emb);
    return p;
}

IPluginV2* EmbLayerNormPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call EmbLayerNormPlugin::destroy()
    return new EmbLayerNormPlugin(name, serialData, serialLength);
}

void EmbLayerNormPluginCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* EmbLayerNormPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
}
