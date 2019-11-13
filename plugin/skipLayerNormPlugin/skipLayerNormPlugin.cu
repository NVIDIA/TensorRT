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
#include "skipLayerNormPlugin.h"
#include "common.h"
#include "serialize.hpp"

#include <cassert>
#include <cstring>
#include <vector>

using namespace nvinfer1;
using bert::operator+;

namespace bert
{

template <typename T, unsigned TPB, bool hasBias>
__global__ void skipLayerNormKernelSmall(
    const int ld, const T* input, const T* skip, const float* beta, const float* gamma, T* output, const T* bias)
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

    layerNormSmall<T, TPB>(val, threadData, ld, idx, beta, gamma, output);
}

template <typename T, unsigned TPB, bool hasBias>
__global__ void skipLayerNormKernel(
    const int ld, const T* input, const T* skip, const float* beta, const float* gamma, T* output, const T* bias)
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

    layerNorm<T, T, TPB>(threadData, ld, offset, beta, gamma, output);
}

template <typename T, bool hasBias>
int computeSkipLayerNorm(cudaStream_t stream, const int ld, const int n, const T* input, const T* skip,
    const float* beta, const float* gamma, T* output, const T* bias)
{

    // this must be true because n is the total size of the tensor
    assert(n % ld == 0);
    const int gridSize = n / ld;

    if (ld <= 32)
    {
        constexpr int blockSize = 32;
        skipLayerNormKernelSmall<T, blockSize, hasBias>
            <<<gridSize, blockSize, 0, stream>>>(ld, input, skip, beta, gamma, output, bias);
    }
    else if (ld <= 128)
    {
        constexpr int blockSize = 128;
        skipLayerNormKernelSmall<T, blockSize, hasBias>
            <<<gridSize, blockSize, 0, stream>>>(ld, input, skip, beta, gamma, output, bias);
    }
    else if (ld == 384)
    {
        constexpr int blockSize = 384;
        skipLayerNormKernelSmall<T, blockSize, hasBias>
            <<<gridSize, blockSize, 0, stream>>>(ld, input, skip, beta, gamma, output, bias);
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
    mBetaDev = deserToDev<float>(d, mLd);
    mGammaDev = deserToDev<float>(d, mLd);
    if (mHasBias)
    {
        const size_t wordSize = samplesCommon::getElementSize(mType);
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
        // return (in.type == DataType::kFLOAT || in.type == DataType::kHALF) && (in.format == TensorFormat::kLINEAR);
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
        if (mHasBias)
        {
            status = computeSkipLayerNorm<float, true>(
                stream, mLd, inputVolume, input, skip, mBetaDev, mGammaDev, output, bias);
        }
        else
        {
            status = computeSkipLayerNorm<float, false>(
                stream, mLd, inputVolume, input, skip, mBetaDev, mGammaDev, output, bias);
        }
    }
    else if (mType == DataType::kHALF)
    {
        const half* input = static_cast<const half*>(inputs[0]);
        const half* skip = static_cast<const half*>(inputs[1]);
        half* output = static_cast<half*>(outputs[0]);
        half* bias = reinterpret_cast<half*>(mBiasDev);

        if (mHasBias)
        {
            status = computeSkipLayerNorm<half, true>(
                stream, mLd, inputVolume, input, skip, mBetaDev, mGammaDev, output, bias);
        }
        else
        {
            status = computeSkipLayerNorm<half, false>(
                stream, mLd, inputVolume, input, skip, mBetaDev, mGammaDev, output, bias);
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
    if (mGamma.values)
    {
        CHECK(cudaMalloc(&mGammaDev, sizeof(float) * mGamma.count));
        CHECK(cudaMemcpy(mGammaDev, mGamma.values, sizeof(float) * mGamma.count, cudaMemcpyHostToDevice));
    }
    if (mBeta.values)
    {
        CHECK(cudaMalloc(&mBetaDev, sizeof(float) * mBeta.count));
        CHECK(cudaMemcpy(mBetaDev, mBeta.values, sizeof(float) * mGamma.count, cudaMemcpyHostToDevice));
    }

    if (mHasBias && mBias.values)
    {
        // target size
        const size_t wordSize = samplesCommon::getElementSize(mType);
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
    size_t biasSize = mHasBias ? (mLd * samplesCommon::getElementSize(mType)) : 0;
    return 2 * sizeof(float) * mLd + sizeof(DataType) + sizeof(mLd) + biasSize + sizeof(mHasBias);
}

void SkipLayerNormPluginDynamic::serialize(void* buffer) const
{
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mLd);
    serialize_value(&buffer, mHasBias);

    char* d = static_cast<char*>(buffer);
    serFromDev(d, mBetaDev, mLd);
    serFromDev(d, mGammaDev, mLd);
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
}
