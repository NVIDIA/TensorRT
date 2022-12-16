/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "reorgPlugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::Reorg;
using nvinfer1::plugin::ReorgPluginCreator;

static const char* REORG_PLUGIN_VERSION{"1"};
static const char* REORG_PLUGIN_NAME{"Reorg_TRT"};
PluginFieldCollection ReorgPluginCreator::mFC{};
std::vector<PluginField> ReorgPluginCreator::mPluginAttributes;

Reorg::Reorg(int C, int H, int W, int stride)
    : C(C),
      H(H),
      W(W), 
      stride(stride)
{
}

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
    PLUGIN_VALIDATE(d == a + length);
}

int Reorg::getNbOutputs() const noexcept
{
    return 1;
}

Dims Reorg::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    PLUGIN_ASSERT(nbInputDims == 1);
    PLUGIN_ASSERT(index == 0);
    return Dims3(inputs[0].d[0] * stride * stride, inputs[0].d[1] / stride, inputs[0].d[2] / stride);
}

int Reorg::enqueue(
    int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    pluginStatus_t status = reorgInference(stream, batchSize, C, H, W, stride, inputData, outputData);    
    return status;
}

size_t Reorg::getSerializationSize() const noexcept
{
    // C, H, W, stride
    return sizeof(int) * 4;
}

void Reorg::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, C);
    write(d, H);
    write(d, W);
    write(d, stride);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

bool Reorg::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

int Reorg::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void Reorg::terminate() noexcept {}

size_t Reorg::getWorkspaceSize(int maxBatchSize) const noexcept
{
    return 0;
}

const char* Reorg::getPluginType() const noexcept
{
    return REORG_PLUGIN_NAME;
}

const char* Reorg::getPluginVersion() const noexcept
{
    return REORG_PLUGIN_VERSION;
}

void Reorg::destroy() noexcept
{
    delete this;
}

// Set plugin namespace
void Reorg::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const char* Reorg::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index
DataType Reorg::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // Only 1 input and 1 output from the plugin layer
    PLUGIN_ASSERT(index == 0);

    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool Reorg::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool Reorg::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void Reorg::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    PLUGIN_ASSERT(*inputTypes == DataType::kFLOAT && floatFormat == PluginFormat::kLINEAR);
    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(stride > 0);
    C = inputDims[0].d[0];
    H = inputDims[0].d[1];
    W = inputDims[0].d[2];
    PLUGIN_ASSERT(H % stride == 0);
    PLUGIN_ASSERT(W % stride == 0);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void Reorg::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept {}

// Detach the plugin object from its execution context.
void Reorg::detachFromContext() noexcept {}

IPluginV2Ext* Reorg::clone() const noexcept
{
    try
    {
        IPluginV2Ext* plugin = new Reorg(C, H, W, stride);
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

ReorgPluginCreator::ReorgPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ReorgPluginCreator::getPluginName() const noexcept
{
    return REORG_PLUGIN_NAME;
}

const char* ReorgPluginCreator::getPluginVersion() const noexcept
{
    return REORG_PLUGIN_VERSION;
}

const PluginFieldCollection* ReorgPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* ReorgPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        PLUGIN_VALIDATE(fc->nbFields == 1);
        PLUGIN_VALIDATE(fields[0].type == PluginFieldType::kINT32);
        PLUGIN_VALIDATE(!strcmp(fields[0].name, "stride"));
        stride = static_cast<int32_t>(*(static_cast<int32_t const*>(fields[0].data)));

        PLUGIN_VALIDATE(stride > 0);

        Reorg* obj = new Reorg(stride);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* ReorgPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call ReorgPlugin::destroy()
        Reorg* obj = new Reorg(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
