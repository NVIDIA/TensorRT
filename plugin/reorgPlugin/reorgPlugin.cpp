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
#include "reorgPlugin.h"

using namespace nvinfer1;
using nvinfer1::PluginType;
using nvinfer1::plugin::Reorg;
using nvinfer1::plugin::ReorgPluginCreator;

static const char* REORG_PLUGIN_VERSION{"1"};
static const char* REORG_PLUGIN_NAME{"Reorg_TRT"};
PluginFieldCollection ReorgPluginCreator::mFC{};
std::vector<PluginField> ReorgPluginCreator::mPluginAttributes;

Reorg::Reorg(int stride)
    : stride(stride)
{
}

Reorg::Reorg(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char*>(buffer), *a = d;
    C = read<int>(d);
    H = read<int>(d);
    W = read<int>(d);
    stride = read<int>(d);
    ASSERT(d == a + length);
}

int Reorg::getNbOutputs() const
{
    return 1;
}

Dims Reorg::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims == 1);
    ASSERT(index == 0);
    return DimsCHW(inputs[0].d[0] * stride * stride, inputs[0].d[1] / stride, inputs[0].d[2] / stride);
}

int Reorg::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    pluginStatus_t status = reorgInference(stream, batchSize, C, H, W, stride, inputData, outputData);
    ASSERT(status == STATUS_SUCCESS);
    return status;
}

size_t Reorg::getSerializationSize() const
{
    // C, H, W, stride
    return sizeof(int) * 4;
}

void Reorg::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, C);
    write(d, H);
    write(d, W);
    write(d, stride);
    ASSERT(d == a + getSerializationSize());
}

bool Reorg::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

int Reorg::initialize()
{
    return STATUS_SUCCESS;
}

void Reorg::terminate() {}

size_t Reorg::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

const char* Reorg::getPluginType() const
{
    return REORG_PLUGIN_NAME;
}

const char* Reorg::getPluginVersion() const
{
    return REORG_PLUGIN_VERSION;
}

void Reorg::destroy()
{
    delete this;
}

// Set plugin namespace
void Reorg::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* Reorg::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
DataType Reorg::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Only 1 input and 1 output from the plugin layer
    ASSERT(index == 0);

    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool Reorg::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool Reorg::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Configure the layer with input and output data types.
void Reorg::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    ASSERT(*inputTypes == DataType::kFLOAT && floatFormat == PluginFormat::kNCHW);
    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);
    ASSERT(stride > 0);
    C = inputDims[0].d[0];
    H = inputDims[0].d[1];
    W = inputDims[0].d[2];
    ASSERT(H % stride == 0);
    ASSERT(W % stride == 0);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void Reorg::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) {}

// Detach the plugin object from its execution context.
void Reorg::detachFromContext() {}

IPluginV2Ext* Reorg::clone() const
{
    IPluginV2Ext* plugin = new Reorg(stride);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

ReorgPluginCreator::ReorgPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ReorgPluginCreator::getPluginName() const
{
    return REORG_PLUGIN_NAME;
}

const char* ReorgPluginCreator::getPluginVersion() const
{
    return REORG_PLUGIN_VERSION;
}

const PluginFieldCollection* ReorgPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* ReorgPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    ASSERT(fc->nbFields == 1);
    ASSERT(fields[0].type == PluginFieldType::kINT32);
    stride = static_cast<int>(*(static_cast<const int*>(fields[0].data)));

    Reorg* obj = new Reorg(stride);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2Ext* ReorgPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call ReorgPlugin::destroy()
    Reorg* obj = new Reorg(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
