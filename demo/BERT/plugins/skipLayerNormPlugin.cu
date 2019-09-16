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
#include "logger.h"
#include "pluginKernels.h"
#include "pluginUtil.h"
#include "skipLayerNormPlugin.h"
#include "common.h"

#include <cassert>
#include <cstring>
#include <vector>

using namespace nvinfer1;
using bert::operator+;

namespace bert
{

namespace test
{

template <typename T, unsigned TPB>
__global__ void skipLayerNormKernelSmall(
    const int ld, const T* input, const T* skip, const float* beta, const float* gamma, T* output)
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
        const T rldval = rld * val;
        threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
    }

    layerNormSmall<T, TPB>(val, threadData, ld, idx, beta, gamma, output);
}

template <typename T, unsigned TPB>
__global__ void skipLayerNormKernel(
    const int ld, const T* input, const T* skip, const float* beta, const float* gamma, T* output)
{
    const T rld = T(1) / T(ld);
    const int offset = blockIdx.x * ld;

    cub::Sum pairSum;
    // reduce x and x^2
    kvp<T> threadData(0, 0);

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        const int idx = offset + i;
        const T val = input[idx] + skip[idx];
        const T rldval = rld * val;
        threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
        output[idx] = val;
    }

    layerNorm<T, TPB>(threadData, ld, offset, beta, gamma, output);
}

template <typename T>
int computeSkipLayerNorm(cudaStream_t stream, const int ld, const int n, const T* input, const T* skip,
    const float* beta, const float* gamma, T* output)
{

    // this must be true because n is the total size of the tensor
    assert(n % ld == 0);
    const int gridSize = n / ld;

    if (ld <= 32)
    {
        constexpr int blockSize = 32;
        skipLayerNormKernelSmall<T, blockSize>
            <<<gridSize, blockSize, 0, stream>>>(ld, input, skip, beta, gamma, output);
    }
    else if (ld <= 128)
    {
        constexpr int blockSize = 128;
        skipLayerNormKernelSmall<T, blockSize>
            <<<gridSize, blockSize, 0, stream>>>(ld, input, skip, beta, gamma, output);
    }
    else if (ld == 384)
    {
        constexpr int blockSize = 384;
        skipLayerNormKernelSmall<T, blockSize>
            <<<gridSize, blockSize, 0, stream>>>(ld, input, skip, beta, gamma, output);
    }
    else
    {
        constexpr int blockSize = 256;
        skipLayerNormKernel<T, blockSize><<<gridSize, blockSize, 0, stream>>>(ld, input, skip, beta, gamma, output);
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
    const std::string name, const int ld, const Weights& beta, const Weights& gamma)
    : mLayerName(name)
    , mLd(ld)
    , mGamma(gamma)
    , mBeta(beta)
{
}

SkipLayerNormPluginDynamic::SkipLayerNormPluginDynamic(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    gLogVerbose << "Skip LN Deser start\n";
    // Deserialize in the same order as serialization
    const char* d = static_cast<const char*>(data);
    const char* a = d;
    DESER(d, mType);
    DESER(d, mLd);
    mBetaDev = deserToDev<float>(d, mLd);
    mGammaDev = deserToDev<float>(d, mLd);
    assert(d == (a + length));
    // this signals init not to allocate/copy
    mGamma.count = mLd;
    mGamma.values = nullptr;
    mBeta.count = mLd;
    mBeta.values = nullptr;

    gLogVerbose << "Skip LN Deser done\n";
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* SkipLayerNormPluginDynamic::clone() const
{
    return new SkipLayerNormPluginDynamic(mLayerName, mLd, mBeta, mGamma);
}

DimsExprs SkipLayerNormPluginDynamic::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder)
{
    assert(nbInputs == 2);
    assert(outputIndex == 0);
    assert(inputs[0].nbDims == inputs[1].nbDims);
    return inputs[0];
}

bool SkipLayerNormPluginDynamic::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    assert(nbInputs == 2);
    assert(nbOutputs == 1);

    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)
    {
        return (in.type == DataType::kFLOAT || in.type == DataType::kHALF) && (in.format == TensorFormat::kLINEAR);
    }
    const PluginTensorDesc& prev = inOut[pos - 1];

    if (pos == 1)
    {
        return in.type == prev.type && in.format == prev.format;
    }
    // output
    return in.type == prev.type && in.format == prev.format;
}

void SkipLayerNormPluginDynamic::configurePlugin(const DynamicPluginTensorDesc* inputs, int nbInputs,
    const DynamicPluginTensorDesc* outputs, int nbOutputs)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(nbInputs == 2);
    mType = inputs[0].desc.type;
    assert(mType == inputs[1].desc.type);
    const auto& inDims0 = inputs[0].desc.dims;
    const auto& inDims1 = inputs[1].desc.dims;
    assert(inDims0.nbDims == inDims1.nbDims);

    assert(std::equal(inDims0.d, inDims0.d + inDims0.nbDims, inDims1.d));

    assert(inDims0.nbDims== 5);
    mLd = inDims0.d[2]; // hiddensize
    assert(inDims0.d[3] == 1);
    assert(inDims0.d[4] == 1);
}

size_t SkipLayerNormPluginDynamic::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs,
    const PluginTensorDesc* outputs, int nbOutputs) const
{
    return 0;
}

int SkipLayerNormPluginDynamic::enqueue(const PluginTensorDesc* inputDesc,
    const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    const int inputVolume = samplesCommon::volume(inputDesc[0].dims);
    int status = -1;

    // Our plugin outputs only one tensor
    // Launch CUDA kernel wrapper and save its return value
    if (mType == DataType::kFLOAT)
    {
        const float* input = static_cast<const float*>(inputs[0]);
        const float* skip = static_cast<const float*>(inputs[1]);
        float* output = static_cast<float*>(outputs[0]);
        status = computeSkipLayerNorm<float>(stream, mLd, inputVolume, input, skip, mBetaDev, mGammaDev, output);
    }
    else if (mType == DataType::kHALF)
    {
        const half* input = static_cast<const half*>(inputs[0]);
        const half* skip = static_cast<const half*>(inputs[1]);
        half* output = static_cast<half*>(outputs[0]);

        status = computeSkipLayerNorm<half>(stream, mLd, inputVolume, input, skip, mBetaDev, mGammaDev, output);
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
    return 0;
}

void SkipLayerNormPluginDynamic::terminate()
{
    gLogVerbose << "SKIPLN terminate start" << std::endl;
    cudaFree(mGammaDev);
    cudaFree(mBetaDev);
    gLogVerbose << "SKIPLN terminate done" << std::endl;
}

size_t SkipLayerNormPluginDynamic::getSerializationSize() const
{
    return 2 * sizeof(float) * mLd + sizeof(DataType) + sizeof(mLd) ;
}

void SkipLayerNormPluginDynamic::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer);
    const char* a = d;

    writeToBuffer(d, mType);
    writeToBuffer(d, mLd);
    serFromDev(d, mBetaDev, mLd);
    serFromDev(d, mGammaDev, mLd);
    assert(d == a + getSerializationSize());
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

    int ld;
    Weights beta;
    Weights gamma;

    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);
        if (field_name.compare("ld") == 0)
        {
            ld = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building ld: " << ld << std::endl;
        }

        if (field_name.compare("beta") == 0)
        {
            gLogVerbose << "Building beta...\n";
            beta.values = fc->fields[i].data;
            beta.count = fc->fields[i].length;
            beta.type = static_cast<DataType>(fc->fields[i].type);
        }

        if (field_name.compare("gamma") == 0)
        {
            gLogVerbose << "Building gamma...\n";
            gamma.values = fc->fields[i].data;
            gamma.count = fc->fields[i].length;
            gamma.type = static_cast<DataType>(fc->fields[i].type);
        }
    }

    SkipLayerNormPluginDynamic* p = new SkipLayerNormPluginDynamic(name, ld, beta, gamma);
    return p;
}

IPluginV2* SkipLayerNormPluginDynamicCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
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
}
