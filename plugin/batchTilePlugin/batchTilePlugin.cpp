/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "batchTilePlugin.h"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

using namespace nvinfer1;
using nvinfer1::plugin::BatchTilePlugin;
using nvinfer1::plugin::BatchTilePluginCreator;

static const char* BATCH_TILE_PLUGIN_VERSION{"1"};
static const char* BATCH_TILE_PLUGIN_NAME{"BatchTilePlugin_TRT"};

PluginFieldCollection BatchTilePluginCreator::mFC{};

BatchTilePlugin::BatchTilePlugin(const std::string name)
    : mLayerName(name)
{
}

BatchTilePlugin::BatchTilePlugin(const std::string name, size_t copy_size)
    : mLayerName(name)
    , mCopySize(copy_size)
{
}

BatchTilePlugin::BatchTilePlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mCopySize = readFromBuffer<size_t>(d);
    assert(d == a + length);
}

int BatchTilePlugin::getNbOutputs() const
{
    return 1;
}

Dims BatchTilePlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims == 2);
    assert(index == 0);
    assert(inputs[1].nbDims == 4);
    return DimsCHW(inputs[1].d[1], inputs[1].d[2], inputs[1].d[3]);
}

int BatchTilePlugin::initialize()
{
    return STATUS_SUCCESS;
}

size_t BatchTilePlugin::getWorkspaceSize(int) const
{
    return 0;
}

DataType BatchTilePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    assert(index == 0);
    return DataType::kFLOAT;
}

int BatchTilePlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    float* output = reinterpret_cast<float*>(outputs[0]);
    // expand to batch size
    for (int i = 0; i < batchSize; i++)
    {
        auto ret = cudaMemcpyAsync(output + i * mCopySize, inputs[1], mCopySize, cudaMemcpyDeviceToDevice, stream);
        if (ret != 0)
        {
            std::cout << "Cuda failure: " << ret;
            abort();
        }
    }
    return 0;
}

void BatchTilePlugin::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    writeToBuffer<size_t>(d, mCopySize);
    assert(d == a + getSerializationSize());
}

void BatchTilePlugin::terminate() {}

size_t BatchTilePlugin::getSerializationSize() const
{
    return sizeof(size_t);
}

bool BatchTilePlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool BatchTilePlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

void BatchTilePlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    assert(nbOutputs == 1);
    assert(inputDims[1].nbDims == 4);
    assert(inputDims[1].d[0] == 1);
    mCopySize = std::accumulate(inputDims[1].d, inputDims[1].d + 4, 1, std::multiplies<int>()) * sizeof(float);
}

bool BatchTilePlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

const char* BatchTilePlugin::getPluginType() const
{
    return BATCH_TILE_PLUGIN_NAME;
}
const char* BatchTilePlugin::getPluginVersion() const
{
    return BATCH_TILE_PLUGIN_VERSION;
}

void BatchTilePlugin::destroy()
{
    delete this;
}

IPluginV2Ext* BatchTilePlugin::clone() const
{
    auto* plugin = new BatchTilePlugin(mLayerName, mCopySize);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void BatchTilePlugin::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* BatchTilePlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

BatchTilePluginCreator::BatchTilePluginCreator()
{
    mFC.nbFields = 0;
    mFC.fields = nullptr;
}

const char* BatchTilePluginCreator::getPluginName() const
{
    return BATCH_TILE_PLUGIN_NAME;
}

const char* BatchTilePluginCreator::getPluginVersion() const
{
    return BATCH_TILE_PLUGIN_VERSION;
}

const PluginFieldCollection* BatchTilePluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* BatchTilePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    auto* plugin = new BatchTilePlugin(name);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* BatchTilePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    return new BatchTilePlugin(name, serialData, serialLength);
}