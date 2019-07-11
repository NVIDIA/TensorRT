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
#include "emb_layer_norm_plugin.hpp"
#include "logger.h"
#include "plugin_kernels.hpp"
#include "plugin_util.hpp"

#include <cassert>
#include <cstring>
#include <vector>

template <typename T, unsigned TPB>
__global__ void emb_layer_norm_kernel(int ld, const int* input_ids, const int* tok_ids, const float* beta,
    const float* gamma, const float* word_emb, const float* pos_emb, const float* tok_emb, T* output)
{

    cub::Sum pairSum;
    // 1. lookup word and token of the block
    // blockIdx.x = position in the sequence
    // blockIdx.y = batch
    // gridDim.x = S
    // gridDim.y = B
    __shared__ int word_id;
    __shared__ int tok_id;

    T rld = T(1.f) / T(ld);
    int seq_pos = blockIdx.y * gridDim.x + blockIdx.x;
    if (threadIdx.x == 0)
    {
        word_id = input_ids[seq_pos];
        tok_id = tok_ids[seq_pos];
    }
    __syncthreads();

    // 2. load pos/tok/word embeddings and add them toghether
    // offset into embeddings is given by word_id * hidden_size
    int poffset = blockIdx.x * ld;
    int woffset = word_id * ld;
    int toffset = tok_id * ld; 
    // the output offset is given by b * (S*hidden_size) + s * hidden_size
    int out_offset = seq_pos * ld;

    kvp<T> thread_data(0, 0);

    for (int it = threadIdx.x; it < ld; it += TPB)
    {
        T w(word_emb[woffset + it]);
        T t(tok_emb[toffset + it]);
        T p(pos_emb[poffset + it]);
        T val = w + t + p;

        output[out_offset + it] = val;
        T rldval = rld * val;
        thread_data = pairSum(thread_data, kvp<T>(rldval, rldval * val));
    }

    // 3. layer norm on the sum
    layer_norm<T, TPB>(thread_data, ld, out_offset, beta, gamma, output);
}

template <typename T>
int emb_skip_layer_norm(cudaStream_t stream, int ld, int B, int S, const int* input_ids, const int* token_ids,
    const float* beta, const float* gamma, const float* word_emb, const float* pos_emb, const float* tok_emb, T* output)
{

    const int tpb = 256;
    dim3 grid(S, B, 1);
    dim3 block(tpb, 1, 1);

    emb_layer_norm_kernel<T, tpb>
        <<<grid, block, 0, stream>>>(ld, input_ids, token_ids, beta, gamma, word_emb, pos_emb, tok_emb, output);
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

EmbLayerNormPlugin::EmbLayerNormPlugin(const std::string name, const bool output_fp16, const Weights& beta,
    const Weights& gamma, const Weights& word_emb, const Weights& pos_emb, const Weights& tok_emb)
    : mLayerName(name)
    , m_ld(beta.count)
    , m_gamma(gamma)
    , m_beta(beta)
    , mWordEmb(word_emb)
    , mPosEmb(pos_emb)
    , mTokEmb(tok_emb)
    , gamma_dev(nullptr)
    , beta_dev(nullptr)
    , wemb_dev(nullptr)
    , temb_dev(nullptr)
    , pemb_dev(nullptr)
{
    // Assuming Weights.count is the number of elements and not bytes
    assert(beta.count == gamma.count);
    assert(word_emb.count % m_ld == 0);
    assert(pos_emb.count % m_ld == 0);
    assert(tok_emb.count % m_ld == 0);
    mWordVocabSize = word_emb.count / m_ld;
    mPosVocabSize = pos_emb.count / m_ld;
    mTokVocabSize = tok_emb.count / m_ld;
    // We set mB and mS in configure
    mType = output_fp16 ? DataType::kHALF : DataType::kFLOAT;
}

EmbLayerNormPlugin::EmbLayerNormPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    gLogInfo << "EMB LN Deser start\n";
    // Deserialize in the same order as serialization
    const char* d = static_cast<const char*>(data);
    const char* a = d;
    DESER(d, mType);
    DESER(d, m_ld);
    DESER(d, mB);
    DESER(d, mS);
    DESER(d, mWordVocabSize);
    DESER(d, mPosVocabSize);
    DESER(d, mTokVocabSize);
    beta_dev = deser2dev<float>(d, m_ld);
    gamma_dev = deser2dev<float>(d, m_ld);

    wemb_dev = deser2dev<float>(d, m_ld * mWordVocabSize);
    pemb_dev = deser2dev<float>(d, m_ld * mPosVocabSize);
    temb_dev = deser2dev<float>(d, m_ld * mTokVocabSize);
    assert(d == (a + length));
    // this signals init not to allocate/copy
    m_gamma.count = -1;
    m_beta.count = -1;
    mWordEmb.count = -1;
    mTokEmb.count = -1;
    mPosEmb.count = -1;
    m_gamma.values = nullptr;
    m_beta.values = nullptr;
    mWordEmb.values = nullptr;
    mTokEmb.values = nullptr;
    mPosEmb.values = nullptr;

    gLogInfo << "EMB LN Deser done\n";
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
    assert(inputs[0].nbDims == inputs[1].nbDims);
    assert(inputs[0].nbDims == 2); // B x S
    int B = inputs[0].d[0];
    int S = inputs[0].d[1];
    assert(inputs[1].d[0] == B);
    assert(inputs[1].d[1] == S);
    assert(inputs[2].d[0] == B);
    assert(inputs[2].d[1] == S);

    assert(index == 0 || index == 1);

    if (index == 0)
    {
        int hidden_size = m_ld;
        return Dims{5, B, S, hidden_size, 1, 1};
    }
    return Dims2{B, 1};
}

int EmbLayerNormPlugin::initialize()
{
    if (m_gamma.values)
    {
        cudaMalloc(&gamma_dev, sizeof(float) * m_gamma.count);
        cudaMemcpy(gamma_dev, m_gamma.values, sizeof(float) * m_gamma.count, cudaMemcpyHostToDevice);
    }
    if (m_beta.values)
    {
        cudaMalloc(&beta_dev, sizeof(float) * m_beta.count);
        cudaMemcpy(beta_dev, m_beta.values, sizeof(float) * m_beta.count, cudaMemcpyHostToDevice);
    }

    if (mWordEmb.values)
    {
        cudaMalloc(&wemb_dev, sizeof(float) * mWordEmb.count);
        cudaMemcpy(wemb_dev, mWordEmb.values, sizeof(float) * mWordEmb.count, cudaMemcpyHostToDevice);
    }
    if (mTokEmb.values)
    {
        cudaMalloc(&temb_dev, sizeof(float) * mTokEmb.count);
        cudaMemcpy(temb_dev, mTokEmb.values, sizeof(float) * mTokEmb.count, cudaMemcpyHostToDevice);
    }

    if (mPosEmb.values)
    {
        cudaMalloc(&pemb_dev, sizeof(float) * mPosEmb.count);
        cudaMemcpy(pemb_dev, mPosEmb.values, sizeof(float) * mPosEmb.count, cudaMemcpyHostToDevice);
    }
    return 0;
}

int EmbLayerNormPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    int status = -1;

    // Our plugin outputs only one tensor
    const int* input_ids = static_cast<const int*>(inputs[0]);
    const int* segment_ids = static_cast<const int*>(inputs[1]);
    const int* input_mask = static_cast<const int*>(inputs[2]);

    if (mType == DataType::kFLOAT)
    {
        float* output = static_cast<float*>(outputs[0]);
        emb_skip_layer_norm<float>(
            stream, m_ld, mB, mS, input_ids, segment_ids, beta_dev, gamma_dev, wemb_dev, pemb_dev, temb_dev, output);
    }
    else if (mType == DataType::kHALF)
    {
        half* output = static_cast<half*>(outputs[0]);
        emb_skip_layer_norm<half>(
            stream, m_ld, mB, mS, input_ids, segment_ids, beta_dev, gamma_dev, wemb_dev, pemb_dev, temb_dev, output);
    }
    else
    {
        assert(false);
    }
    int* mask_idx = static_cast<int*>(outputs[1]);
    compute_mask_idx(stream, mS, mB, input_mask, mask_idx);

    return status;
}

size_t EmbLayerNormPlugin::getSerializationSize() const
{
    return 2 * sizeof(float) * m_ld             // beta + gamma
        + sizeof(mType) + sizeof(m_ld) * 6      //m_ld, mB,mS, m*VocabSize
        + sizeof(float) * m_ld * mWordVocabSize // word emb
        + sizeof(float) * m_ld * mPosVocabSize  // pos emb
        + sizeof(float) * m_ld * mTokVocabSize  // tok emb
        ;
}

void EmbLayerNormPlugin::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer);
    const char* a = d;
    writeToBuffer(d, mType);
    writeToBuffer(d, m_ld);
    writeToBuffer(d, mB);
    writeToBuffer(d, mS);
    writeToBuffer(d, mWordVocabSize);
    writeToBuffer(d, mPosVocabSize);
    writeToBuffer(d, mTokVocabSize);
    serFromDev(d, beta_dev, m_ld);
    serFromDev(d, gamma_dev, m_ld);
    serFromDev(d, wemb_dev, m_ld * mWordVocabSize);
    serFromDev(d, pemb_dev, m_ld * mPosVocabSize);
    serFromDev(d, temb_dev, m_ld * mTokVocabSize);

    assert(d == a + getSerializationSize());
}

void EmbLayerNormPlugin::configurePlugin(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat format, int maxBatchSize)
{

    // Validate input arguments
    assert(nbOutputs == 2);
    assert(nbInputs == 3);

    assert(inputs[0].nbDims == 2);
    mB = inputs[0].d[0];
    mS = inputs[0].d[1];
    assert(mB == inputs[1].d[0]);
    assert(mS == inputs[1].d[1]);
    assert(mB == inputs[2].d[0]);
    assert(mS == inputs[2].d[1]);

    assert(outputs[0].nbDims == 5);
    assert(outputs[0].d[0] == mB);
    assert(outputs[0].d[1] == mS);
    assert(outputs[0].d[2] == m_ld);
    assert(outputs[0].d[3] == 1);
    assert(outputs[0].d[4] == 1);

    assert(outputs[1].nbDims == 2);
    assert(outputs[1].d[0] == mB);
    assert(outputs[1].d[1] == 1);

    assert(format == PluginFormat::kNCHW);
    assert(inputTypes[0] == DataType::kINT32);
    assert(inputTypes[1] == DataType::kINT32);
    assert(inputTypes[2] == DataType::kINT32);
    DataType out_type = outputTypes[0];
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
    gLogInfo << "EMBLN terminate start" << std::endl;
    cudaFree(gamma_dev);
    cudaFree(beta_dev);
    cudaFree(wemb_dev);
    cudaFree(temb_dev);
    cudaFree(pemb_dev);
    gLogInfo << "EMBLN terminate done" << std::endl;
}

void EmbLayerNormPlugin::destroy()
{
    gLogInfo << "EMBLN destroy start" << std::endl;
    // This gets called when the network containing plugin is destroyed
    delete this;
    gLogInfo << "EMBLN destroy start" << std::endl;
}

IPluginV2Ext* EmbLayerNormPlugin::clone() const
{
    gLogInfo << "EMBLN clone start" << std::endl;
    auto ret
        = new EmbLayerNormPlugin(mLayerName, mType == DataType::kHALF, m_beta, m_gamma, mWordEmb, mPosEmb, mTokEmb);
    ret->mB = mB;
    ret->mS = mS;

    ret->wemb_dev = wemb_dev;
    ret->pemb_dev = pemb_dev;
    ret->temb_dev = temb_dev;
    ret->beta_dev = beta_dev;
    ret->gamma_dev = gamma_dev;
    gLogInfo << "EMBLN clone done" << std::endl;
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

    gLogError << "EmbLayerNormPluginCreator::createPlugin - not implemented\n";
    assert(false);
    return nullptr; // new EmbLayerNormPlugin(name, ld);
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
