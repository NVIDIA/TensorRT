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
#include "bertCommon.h"
#include "common.h"
#include "serialize.hpp"
#include "skipLayerNormPlugin.h"

#include <cassert>
#include <cstring>
#include <vector>

using namespace nvinfer1;
using bert::operator+;

namespace bert
{

template <typename T, int TPB, int VPT, bool hasBias>
__global__ void skipln_vec(
    const int ld, const T* input, const T* skip, T* output, const T* beta, const T* gamma, const T* bias)
{
    const int idx = ld * blockIdx.x + threadIdx.x * VPT;
    // 4 * 1024 * 4 * 2 Bytes = 16KB per block
    T in_local[VPT];
    T skip_local[VPT];
    T bias_local[VPT];
    copy<sizeof(T) * VPT>(&input[idx], in_local);
    copy<sizeof(T) * VPT>(&skip[idx], skip_local);
    copy<sizeof(T) * VPT>(&bias[threadIdx.x * VPT], bias_local);
    T local = 0.f;
    T local2 = 0.f;

    const T rld = T(1) / T(ld);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        in_local[it] += skip_local[it];
        if (hasBias)
            in_local[it] += bias_local[it];
        const T tmp = rld * in_local[it];
        local += tmp;
        local2 += tmp * in_local[it];
    }

    copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], bias_local);
    copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], skip_local);

    using BlockReduce = cub::BlockReduce<kvp<T>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T mu;     // mean
    __shared__ T rsigma; // 1 / std.dev.

    const auto sumKV = BlockReduce(temp_storage).Reduce(kvp<T>(local, local2), cub::Sum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu + T(1e-5));
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        in_local[it] = skip_local[it] * (in_local[it] - mu) * rsigma + bias_local[it];
    }

    copy<sizeof(T) * VPT>(in_local, &output[idx]);
}

template <typename T, unsigned TPB, bool hasBias>
__global__ void skipLayerNormKernelSmall(
    const int ld, const T* input, const T* skip, const T* beta, const T* gamma, T* output, const T* bias)
{

    const T rld = T(1) / T(ld);
    const int offset = blockIdx.x * ld;

    cub::Sum pairSum;
    // reduce x and x^2
    kvp<T> threadData(0, 0);
    const int idx = offset + threadIdx.x;
    T val = 0;

    if (threadIdx.x < ld)
    {

        val = input[idx] + skip[idx];
        if (hasBias)
        {
            val += bias[threadIdx.x];
        }

        const T rldval = rld * val;
        threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
    }

    layerNormSmall<T, T, TPB>(val, threadData, ld, idx, beta, gamma, output);
}

template <typename T, unsigned TPB, bool hasBias>
__global__ void skipLayerNormKernel(
    const int ld, const T* input, const T* skip, const T* beta, const T* gamma, T* output, const T* bias)
{
    const T rld = T(1) / T(ld);
    const int offset = blockIdx.x * ld;

    cub::Sum pairSum;
    // reduce x and x^2
    kvp<T> threadData(0, 0);

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        const int idx = offset + i;
        T val = T(input[idx]) + T(skip[idx]);

        if (hasBias)
        {
            val += T(bias[i]);
        }
        const T rldval = rld * val;
        threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
        output[idx] = val;
    }

    layerNorm<T, T, T, TPB>(threadData, ld, offset, beta, gamma, output);
}

template <typename T, bool hasBias>
int computeSkipLayerNorm(cudaStream_t stream, const int ld, const int n, const T* input, const T* skip, const T* beta,
    const T* gamma, T* output, const T* bias)
{

    // this must be true because n is the total size of the tensor
    assert(n % ld == 0);
    const int gridSize = n / ld;
    constexpr int VPT = 16 / sizeof(T);
    if (ld <= 32)
    {
        constexpr int blockSize = 32;
        skipLayerNormKernelSmall<T, blockSize, hasBias>
            <<<gridSize, blockSize, 0, stream>>>(ld, input, skip, beta, gamma, output, bias);
    }
    else if (ld == 768)
    {
        constexpr int TPB = 768 / VPT;
        skipln_vec<T, TPB, VPT, hasBias><<<gridSize, TPB, 0, stream>>>(ld, input, skip, output, beta, gamma, bias);
    }
    else if (ld == 1024)
    {
        constexpr int TPB = 1024 / VPT;
        skipln_vec<T, TPB, VPT, hasBias><<<gridSize, TPB, 0, stream>>>(ld, input, skip, output, beta, gamma, bias);
    }
    else
    {
        constexpr int blockSize = 256;
        skipLayerNormKernel<T, blockSize, hasBias>
            <<<gridSize, blockSize, 0, stream>>>(ld, input, skip, beta, gamma, output, bias);
    }
    CHECK(cudaPeekAtLastError());

    return 0;
}

// Clip plugin specific constants
namespace
{
static const char* SKIP_LAYER_NORM_VERSION{"1"};
static const char* SKIP_LAYER_NORM_NAME{"CustomSkipLayerNormPluginDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection SkipLayerNormPluginDynamicCreator::mFC{};
std::vector<PluginField> SkipLayerNormPluginDynamicCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(SkipLayerNormPluginDynamicCreator);

SkipLayerNormPluginDynamic::SkipLayerNormPluginDynamic(
    const std::string name, const DataType type, const int ld, const Weights& beta, const Weights& gamma)
    : mLayerName(name)
    , mLd(ld)
    , mGamma(gamma)
    , mBeta(beta)
    , mHasBias(false)
    , mType(type)
{
    mBias.values = nullptr;
    mBias.count = 0;
}

SkipLayerNormPluginDynamic::SkipLayerNormPluginDynamic(const std::string name, const DataType type, const int ld,
    const Weights& beta, const Weights& gamma, const Weights& bias)
    : mLayerName(name)
    , mLd(ld)
    , mGamma(gamma)
    , mBeta(beta)
    , mHasBias(true)
    , mBias(bias)
    , mType(type)
{
}

SkipLayerNormPluginDynamic::SkipLayerNormPluginDynamic(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    gLogVerbose << "Starting to deserialize SkipLayerNorm plugin" << std::endl;
    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mLd);
    deserialize_value(&data, &length, &mHasBias);

    const char* d = static_cast<const char*>(data);

    const size_t wordSize = samplesCommon::getElementSize(mType);

    mBetaDev = deserToDev<char>(d, mLd * wordSize);
    mGammaDev = deserToDev<char>(d, mLd * wordSize);
    if (mHasBias)
    {
        mBiasDev = deserToDev<char>(d, mLd * wordSize);
    }
    // this signals init not to allocate/copy
    mGamma.count = mLd;
    mGamma.values = nullptr;
    mBeta.count = mLd;
    mBeta.values = nullptr;
    mBias.count = mLd;
    mBias.values = nullptr;

    gLogVerbose << "Finished deserializing SkipLayerNorm plugin" << std::endl;
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* SkipLayerNormPluginDynamic::clone() const
{
    if (mHasBias)
    {
        return new SkipLayerNormPluginDynamic(mLayerName, mType, mLd, mBeta, mGamma, mBias);
    }
    return new SkipLayerNormPluginDynamic(mLayerName, mType, mLd, mBeta, mGamma);
}

DimsExprs SkipLayerNormPluginDynamic::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder)
{
    assert(nbInputs == 2);
    assert(outputIndex == 0);
    assert(inputs[0].nbDims == inputs[1].nbDims);
    return inputs[0];
}

bool SkipLayerNormPluginDynamic::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    assert(nbInputs == 2);
    assert(nbOutputs == 1);

    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)
    {
        return (in.type == mType) && (in.format == TensorFormat::kLINEAR);
    }
    const PluginTensorDesc& prev = inOut[pos - 1];

    if (pos == 1)
    {
        return in.type == prev.type && in.format == prev.format;
    }
    // output
    return in.type == prev.type && in.format == prev.format;
}

void SkipLayerNormPluginDynamic::configurePlugin(
    const DynamicPluginTensorDesc* inputs, int nbInputs, const DynamicPluginTensorDesc* outputs, int nbOutputs)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(nbInputs == 2);
    assert(mType == inputs[0].desc.type);
    assert(mType == inputs[1].desc.type);
    const auto& inDims0 = inputs[0].desc.dims;
    const auto& inDims1 = inputs[1].desc.dims;
    TRT_UNUSED inDims1;
    assert(inDims0.nbDims == inDims1.nbDims);

    assert(std::equal(inDims0.d, inDims0.d + inDims0.nbDims, inDims1.d));

    assert(inDims0.nbDims == 5);
    mLd = inDims0.d[HDIM]; // hiddensize
    assert(inDims0.d[3] == 1);
    assert(inDims0.d[4] == 1);
}

size_t SkipLayerNormPluginDynamic::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const
{
    return 0;
}

int SkipLayerNormPluginDynamic::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    const int inputVolume = volume(inputDesc[0].dims);
    int status = -1;

    // Our plugin outputs only one tensor
    // Launch CUDA kernel wrapper and save its return value
    if (mType == DataType::kFLOAT)
    {
        const float* input = static_cast<const float*>(inputs[0]);
        const float* skip = static_cast<const float*>(inputs[1]);
        float* output = static_cast<float*>(outputs[0]);

        float* bias = reinterpret_cast<float*>(mBiasDev);
        const float* beta = static_cast<const float*>(mBetaDev);
        const float* gamma = static_cast<const float*>(mGammaDev);
        if (mHasBias)
        {
            status
                = computeSkipLayerNorm<float, true>(stream, mLd, inputVolume, input, skip, beta, gamma, output, bias);
        }
        else
        {
            status
                = computeSkipLayerNorm<float, false>(stream, mLd, inputVolume, input, skip, beta, gamma, output, bias);
        }
    }
    else if (mType == DataType::kHALF)
    {
        const half* input = static_cast<const half*>(inputs[0]);
        const half* skip = static_cast<const half*>(inputs[1]);
        half* output = static_cast<half*>(outputs[0]);
        half* bias = reinterpret_cast<half*>(mBiasDev);

        const half* beta = static_cast<const half*>(mBetaDev);
        const half* gamma = static_cast<const half*>(mGammaDev);
        if (mHasBias)
        {
            status = computeSkipLayerNorm<half, true>(stream, mLd, inputVolume, input, skip, beta, gamma, output, bias);
        }
        else
        {
            status
                = computeSkipLayerNorm<half, false>(stream, mLd, inputVolume, input, skip, beta, gamma, output, bias);
        }
    }
    else
    {
        gLogError << "Unsupported Type\n";
        assert(false);
    }
    return status;
}

// IPluginV2Ext Methods
DataType SkipLayerNormPluginDynamic::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const
{
    assert(index == 0);
    assert(nbInputs == 2);
    assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
    assert(inputTypes[0] == inputTypes[1]);
    return inputTypes[0];
}

// IPluginV2 Methods
const char* SkipLayerNormPluginDynamic::getPluginType() const
{
    return SKIP_LAYER_NORM_NAME;
}

const char* SkipLayerNormPluginDynamic::getPluginVersion() const
{
    return SKIP_LAYER_NORM_VERSION;
}

int SkipLayerNormPluginDynamic::getNbOutputs() const
{
    return 1;
}
int SkipLayerNormPluginDynamic::initialize()
{
    const size_t wordSize = samplesCommon::getElementSize(mType);
    if (mGamma.values)
    {
        CHECK(cudaMalloc(&mGammaDev, sizeof(float) * mGamma.count));

        // target size
        const size_t nbBytes = mGamma.count * wordSize;
        CHECK(cudaMalloc(&mGammaDev, nbBytes));

        if (mType == DataType::kFLOAT)
        {
            convertAndCopyToDevice(mGamma, static_cast<float*>(mGammaDev));
        }
        else
        {
            convertAndCopyToDevice(mGamma, static_cast<half*>(mGammaDev));
        }
    }
    if (mBeta.values)
    {
        CHECK(cudaMalloc(&mBetaDev, sizeof(float) * mBeta.count));
        const size_t nbBytes = mBeta.count * wordSize;
        CHECK(cudaMalloc(&mBetaDev, nbBytes));

        if (mType == DataType::kFLOAT)
        {
            convertAndCopyToDevice(mBeta, static_cast<float*>(mBetaDev));
        }
        else
        {
            convertAndCopyToDevice(mBeta, static_cast<half*>(mBetaDev));
        }
    }

    if (mHasBias && mBias.values)
    {
        // target size
        const size_t nbBytes = mBias.count * wordSize;
        CHECK(cudaMalloc(&mBiasDev, nbBytes));

        if (mType == DataType::kFLOAT)
        {
            convertAndCopyToDevice(mBias, reinterpret_cast<float*>(mBiasDev));
        }
        else
        {
            convertAndCopyToDevice(mBias, reinterpret_cast<half*>(mBiasDev));
        }
    }
    return 0;
}

void SkipLayerNormPluginDynamic::terminate()
{
    gLogVerbose << "SKIPLN terminate start" << std::endl;
    cudaFree(mGammaDev);
    cudaFree(mBetaDev);
    if (mHasBias)
    {
        cudaFree(mBiasDev);
    }
    gLogVerbose << "SKIPLN terminate done" << std::endl;
}

size_t SkipLayerNormPluginDynamic::getSerializationSize() const
{
    const size_t wordSize = samplesCommon::getElementSize(mType);
    const size_t biasSize = mHasBias ? (mLd * wordSize) : 0;
    return 2 * wordSize * mLd + sizeof(DataType) + sizeof(mLd) + biasSize + sizeof(mHasBias);
}

void SkipLayerNormPluginDynamic::serialize(void* buffer) const
{
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mLd);
    serialize_value(&buffer, mHasBias);

    const size_t wordSize = samplesCommon::getElementSize(mType);
    char* d = static_cast<char*>(buffer);
    serFromDev(d, static_cast<char*>(mBetaDev), mLd * wordSize);
    serFromDev(d, static_cast<char*>(mGammaDev), mLd * wordSize);
    if (mHasBias)
    {
        const size_t wordSize = samplesCommon::getElementSize(mType);
        serFromDev(d, mBiasDev, mLd * wordSize);
    }
}

void SkipLayerNormPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void SkipLayerNormPluginDynamic::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* SkipLayerNormPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

/////////////////////////////////////////////////////////

SkipLayerNormPluginDynamicCreator::SkipLayerNormPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SkipLayerNormPluginDynamicCreator::getPluginName() const
{
    return SKIP_LAYER_NORM_NAME;
}

const char* SkipLayerNormPluginDynamicCreator::getPluginVersion() const
{
    return SKIP_LAYER_NORM_VERSION;
}

const PluginFieldCollection* SkipLayerNormPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* SkipLayerNormPluginDynamicCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    gLogVerbose << "Creating SkipLayerNormPluginDynamicCreator...\n";

    int ld = 0;
    Weights beta{DataType::kFLOAT, nullptr, 0};
    Weights gamma{DataType::kFLOAT, nullptr, 0};
    Weights bias{DataType::kFLOAT, nullptr, 0};
    int typeId = -1;

    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);
        if (field_name.compare("ld") == 0)
        {
            ld = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building ld: " << ld << std::endl;
        }

        if (field_name.compare("type_id") == 0)
        {
            typeId = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building typeId: " << typeId << std::endl;
        }

        if (field_name.compare("beta") == 0)
        {
            gLogVerbose << "Building beta...\n";
            beta.values = fc->fields[i].data;
            beta.count = fc->fields[i].length;
            beta.type = fieldTypeToDataType(fc->fields[i].type);
        }

        if (field_name.compare("gamma") == 0)
        {
            gLogVerbose << "Building gamma...\n";
            gamma.values = fc->fields[i].data;
            gamma.count = fc->fields[i].length;
            gamma.type = fieldTypeToDataType(fc->fields[i].type);
        }

        if (field_name.compare("bias") == 0)
        {
            gLogVerbose << "Building bias...\n";
            bias.values = fc->fields[i].data;
            bias.count = fc->fields[i].length;
            bias.type = fieldTypeToDataType(fc->fields[i].type);
        }
    }

    if (typeId < 0 || typeId > 3)
    {
        gLogError << "SkipLayerNorm: Invalid type ID: " << typeId << std::endl;
    }

    if (beta.count <= 0 || beta.values == nullptr)
    {
        gLogError << "SkipLayerNorm: invalid beta" << std::endl;
    }

    if (gamma.count <= 0 || gamma.values == nullptr)
    {
        gLogError << "SkipLayerNorm: invalid gamma" << std::endl;
    }
    DataType type = static_cast<DataType>(typeId);
    if (bias.values == nullptr)
    {
        return new SkipLayerNormPluginDynamic(name, type, ld, beta, gamma);
    }

    return new SkipLayerNormPluginDynamic(name, type, ld, beta, gamma, bias);
}

IPluginV2* SkipLayerNormPluginDynamicCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call SkipLayerNormPluginDynamic::destroy()
    return new SkipLayerNormPluginDynamic(name, serialData, serialLength);
}

void SkipLayerNormPluginDynamicCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* SkipLayerNormPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
} // namespace bert
