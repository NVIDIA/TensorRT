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

#include <cassert>
#include <cstring>
#include <vector>

using bert::operator+;

namespace bert
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

using namespace nvinfer1;

// Clip plugin specific constants
namespace
{
static const char* SKIP_LAYER_NORM_VERSION{"1"};
static const char* SKIP_LAYER_NORM_NAME{"CustomSkipLayerNormPlugin"};
} // namespace

// Static class fields initialization
PluginFieldCollection SkipLayerNormPluginCreator::mFC{};
std::vector<PluginField> SkipLayerNormPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(SkipLayerNormPluginCreator);

SkipLayerNormPlugin::SkipLayerNormPlugin(
    const std::string name, const int ld, const Weights& beta, const Weights& gamma)
    : mLayerName(name)
    , mLd(ld)
    , mGamma(gamma)
    , mBeta(beta)
{
}

SkipLayerNormPlugin::SkipLayerNormPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    gLogVerbose << "Skip LN Deser start\n";
    // Deserialize in the same order as serialization
    const char* d = static_cast<const char*>(data);
    const char* a = d;
    DESER(d, mType);
    DESER(d, mLd);
    DESER(d, mInputVolume);
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

const char* SkipLayerNormPlugin::getPluginType() const
{
    return SKIP_LAYER_NORM_NAME;
}

const char* SkipLayerNormPlugin::getPluginVersion() const
{
    return SKIP_LAYER_NORM_VERSION;
}

int SkipLayerNormPlugin::getNbOutputs() const
{
    return 1;
}

Dims SkipLayerNormPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Validate input arguments
    assert(nbInputDims == 2);
    assert(index == 0);
    assert(inputs[0].nbDims == inputs[1].nbDims);
    for (int d = 0; d < inputs[0].nbDims; d++)
    {
        assert(inputs[0].d[d] == inputs[1].d[d]);
    }

    return inputs[0];
}

int SkipLayerNormPlugin::initialize()
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

int SkipLayerNormPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    int status = -1;

    // Our plugin outputs only one tensor

    // Launch CUDA kernel wrapper and save its return value
    if (mType == DataType::kFLOAT)
    {

        const float* input = static_cast<const float*>(inputs[0]);
        const float* skip = static_cast<const float*>(inputs[1]);
        float* output = static_cast<float*>(outputs[0]);
        status = computeSkipLayerNorm<float>(
            stream, mLd, mInputVolume * batchSize, input, skip, mBetaDev, mGammaDev, output);
    }
    else if (mType == DataType::kHALF)
    {
        const half* input = static_cast<const half*>(inputs[0]);
        const half* skip = static_cast<const half*>(inputs[1]);
        half* output = static_cast<half*>(outputs[0]);

        status = computeSkipLayerNorm<half>(
            stream, mLd, mInputVolume * batchSize, input, skip, mBetaDev, mGammaDev, output);
    }
    else
    {
        assert(false);
    }
    return status;
}

size_t SkipLayerNormPlugin::getSerializationSize() const
{
    return 2 * sizeof(float) * mLd + sizeof(DataType) + sizeof(mLd) + sizeof(mInputVolume);
}

void SkipLayerNormPlugin::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer);
    const char* a = d;

    writeToBuffer(d, mType);
    writeToBuffer(d, mLd);
    writeToBuffer(d, mInputVolume);
    serFromDev(d, mBetaDev, mLd);
    serFromDev(d, mGammaDev, mLd);
    assert(d == a + getSerializationSize());
}

void SkipLayerNormPlugin::configureWithFormat(
    const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(nbInputs == 2);

    // Fetch volume for future enqueue() operations
    size_t volume = 1;
    for (int i = 0; i < inputs->nbDims; i++)
    {
        volume *= inputs->d[i];
    }
    mInputVolume = volume;
    assert(inputs->nbDims == 4);
    mLd = inputs->d[1]; // hiddensize
    assert(inputs->d[2] == 1);
    assert(inputs->d[3] == 1);

    mType = type;
}

bool SkipLayerNormPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    // This plugin only supports ordinary floats, and NCHW input format
    if (type == DataType::kFLOAT || type == DataType::kHALF)
    {
        return format == PluginFormat::kNCHW;
    }
    else
    {
        return false;
    }
}

void SkipLayerNormPlugin::terminate()
{
    gLogVerbose << "SKIPLN terminate start" << std::endl;
    cudaFree(mGammaDev);
    cudaFree(mBetaDev);
    gLogVerbose << "SKIPLN terminate done" << std::endl;
}

void SkipLayerNormPlugin::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2* SkipLayerNormPlugin::clone() const
{
    return new SkipLayerNormPlugin(mLayerName, mLd, mBeta, mGamma);
}

void SkipLayerNormPlugin::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* SkipLayerNormPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

SkipLayerNormPluginCreator::SkipLayerNormPluginCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SkipLayerNormPluginCreator::getPluginName() const
{
    return SKIP_LAYER_NORM_NAME;
}

const char* SkipLayerNormPluginCreator::getPluginVersion() const
{
    return SKIP_LAYER_NORM_VERSION;
}

const PluginFieldCollection* SkipLayerNormPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* SkipLayerNormPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    gLogVerbose << "Creating SkipLayerNormPluginCreator...\n";

    int ld;
    Weights beta;
    Weights gamma;

    for(int i=0; i< fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);
        if (field_name.compare("ld")==0)
        {
            ld = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building ld: " << ld << std::endl;
        }

        if (field_name.compare("beta")==0)
        {
            gLogVerbose << "Building beta...\n";
            beta.values = fc->fields[i].data;
            beta.count = fc->fields[i].length;
            beta.type = static_cast<DataType>(fc->fields[i].type);
        }

        if (field_name.compare("gamma")==0)
        {
            gLogVerbose << "Building gamma...\n";
            gamma.values = fc->fields[i].data;
            gamma.count = fc->fields[i].length;
            gamma.type = static_cast<DataType>(fc->fields[i].type);
        }
    }

    SkipLayerNormPlugin* p = new SkipLayerNormPlugin(name, ld, beta, gamma);
    return p;
}

IPluginV2* SkipLayerNormPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call SkipLayerNormPlugin::destroy()
    return new SkipLayerNormPlugin(name, serialData, serialLength);
}

void SkipLayerNormPluginCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* SkipLayerNormPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
}
