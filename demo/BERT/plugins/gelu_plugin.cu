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
#include "gelu_plugin.hpp"
#include "plugin_kernels.hpp"

#include "logger.h"
#include <cassert>
#include <cstring>
#include <vector>

////// CUDA KERNELS ///////////////////////

template <unsigned TPB>
__global__ void gelu_kernel(int n, const float* input, float* output)
{

    const float b = sqrt(2.0 / M_PI);
    const float c = 0.044715 * sqrt(2.0 / M_PI);

    int idx = blockIdx.x * TPB + threadIdx.x;

    if (idx < n)
    {
        float in = input[idx];
        float cdf = (0.5f) + (0.5f) * myTanh(in * (c * in * in + b));
        output[idx] = in * cdf;
    }
}

template <unsigned TPB>
__global__ void gelu_kernel(int n, const half* input, half* output)
{
    const int n2 = n / 2;
    const half2* in2 = (half2*) input;
    half2* out2 = (half2*) output;

    const half a = 0.5;

    const half b = sqrt(2.0 / M_PI);
    const half c = 0.044715 * sqrt(2.0 / M_PI);

    const half2 a2 = __halves2half2(a, a);
    const half2 b2 = __halves2half2(b, b);
    const half2 c2 = __halves2half2(c, c);

    int idx = blockIdx.x * TPB + threadIdx.x;

    if (idx < n2)
    {
        half2 in = in2[idx];
        half2 cpow3pb = __hmul2(__hfma2(__hmul2(in, in), c2, b2), in);
        half2 th = myTanh(cpow3pb);
        half2 cdf = __hfma2(th, a2, a2);
        out2[idx] = __hmul2(in, cdf);
    }

    if ((n & 1) && blockIdx.x == 0 && threadIdx.x == 0)
    {
        int idx = n - 1;
        half in = input[idx];
        half cpow3pb = __hmul(__hfma(__hmul(in, in), c, b), in);
        half th = myTanh(cpow3pb);
        half cdf = __hfma(th, a, a);
        output[idx] = __hmul(in, cdf);
    }
}

template <typename T>
int compute_gelu(cudaStream_t stream, int n, const T* input, T* output)
{
    const int blockSize = 256;
    const int gridSize = (n / blockSize) + ((blockSize * (n / blockSize)) < n);
    gelu_kernel<blockSize><<<gridSize, blockSize, 0, stream>>>(n, input, output);

    CHECK(cudaPeekAtLastError());
    return 0;
}

////////////////////////////////////////////

using namespace nvinfer1;

namespace
{
static const char* GELU_PLUGIN_VERSION{"1"};
static const char* GELU_PLUGIN_NAME{"CustomGeluPlugin"};
} // namespace

// Static class fields initialization
PluginFieldCollection GeluPluginCreator::mFC{};
std::vector<PluginField> GeluPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GeluPluginCreator);

GeluPlugin::GeluPlugin(const std::string name)
    : mLayerName(name)
{
}

GeluPlugin::GeluPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{

    gLogInfo << "Gelu Deser start" << std::endl;
    const char *d = static_cast<const char*>(data), *a = d;
    mInputVolume = readFromBuffer<decltype(mInputVolume)>(d);
    mType = readFromBuffer<DataType>(d);
    assert(d == a + length);
    gLogInfo << "Gelu Deser done" << std::endl;
}

const char* GeluPlugin::getPluginType() const
{
    return GELU_PLUGIN_NAME;
}

const char* GeluPlugin::getPluginVersion() const
{
    return GELU_PLUGIN_VERSION;
}

int GeluPlugin::getNbOutputs() const
{
    return 1;
}

Dims GeluPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Validate input arguments
    assert(nbInputDims == 1);
    assert(index == 0);

    // doesn't change input dimension, so output Dims will be the same as
    // input Dims
    return *inputs;
}

int GeluPlugin::initialize()
{
    return 0;
}

int GeluPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    int status = -1;

    // Our plugin outputs only one tensor
    // Launch CUDA kernel wrapper and save its return value
    if (mType == DataType::kFLOAT)
    {
        const float* input = static_cast<const float*>(inputs[0]);
        float* output = static_cast<float*>(outputs[0]);
        status = compute_gelu<float>(stream, mInputVolume, input, output);
    }
    else if (mType == DataType::kHALF)
    {
        const half* input = static_cast<const half*>(inputs[0]);
        half* output = static_cast<half*>(outputs[0]);
        status = compute_gelu<half>(stream, mInputVolume, input, output);
    }
    else
    {
        assert(false);
    }

    return status;
}

size_t GeluPlugin::getSerializationSize() const
{
    return sizeof(mInputVolume) + sizeof(DataType);
}

void GeluPlugin::serialize(void* buffer) const
{
    char *d = static_cast<char*>(buffer), *a = d;
    writeToBuffer(d, mInputVolume);
    writeToBuffer(d, mType);
    assert(d == a + getSerializationSize());
}

void GeluPlugin::configureWithFormat(
    const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int)
{

    // Validate input arguments
    assert(nbOutputs == 1);
    assert(format == PluginFormat::kNCHW);

    // Fetch volume for future enqueue() operations
    size_t volume = 1;
    for (int i = 0; i < inputs->nbDims; i++)
    {
        volume *= inputs->d[i];
    }
    mInputVolume = volume;
    mType = type;
}

bool GeluPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    if (type == DataType::kFLOAT || type == DataType::kHALF)
        return format == PluginFormat::kNCHW;
    else
        return false;
}

void GeluPlugin::terminate() {}

void GeluPlugin::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2* GeluPlugin::clone() const
{
    return new GeluPlugin(mLayerName);
}

void GeluPlugin::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* GeluPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

GeluPluginCreator::GeluPluginCreator()
{

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GeluPluginCreator::getPluginName() const
{
    return GELU_PLUGIN_NAME;
}

const char* GeluPluginCreator::getPluginVersion() const
{
    return GELU_PLUGIN_VERSION;
}

const PluginFieldCollection* GeluPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* GeluPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    gLogError << "GeluPluginCreator::createPlugin not implemented\n" << std::endl;
    assert(false);
    return nullptr;
}

IPluginV2* GeluPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call GeluPlugin::destroy()
    return new GeluPlugin(name, serialData, serialLength);
}

void GeluPluginCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* GeluPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
