/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "resizeNearestPlugin.h"
#include "common/plugin.h"
#include <algorithm>
#include <cuda_runtime_api.h>
#include <iostream>

#define DEBUG 0

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::ResizeNearest;
using nvinfer1::plugin::ResizeNearestPluginCreator;

namespace
{
char const* const kRESIZE_PLUGIN_VERSION{"1"};
char const* const kRESIZE_PLUGIN_NAME{"ResizeNearest_TRT"};
} // namespace

ResizeNearestPluginCreator::ResizeNearestPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("scale", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* ResizeNearestPluginCreator::getPluginName() const noexcept
{
    return kRESIZE_PLUGIN_NAME;
}

char const* ResizeNearestPluginCreator::getPluginVersion() const noexcept
{
    return kRESIZE_PLUGIN_VERSION;
}

PluginFieldCollection const* ResizeNearestPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* ResizeNearestPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        plugin::validateRequiredAttributesExist({"scale"}, fc);
        PluginField const* fields = fc->fields;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "scale"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                mScale = *(static_cast<float const*>(fields[i].data));
            }
        }
        return new ResizeNearest(mScale);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* ResizeNearestPluginCreator::deserializePlugin(char const* name, void const* data, size_t length) noexcept
{
    try
    {
        return new ResizeNearest(data, length);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

ResizeNearest::ResizeNearest(float scale)
    : mScale(scale)
{
    PLUGIN_VALIDATE(mScale > 0);
}

int32_t ResizeNearest::getNbOutputs() const noexcept
{
    return 1;
}

Dims ResizeNearest::getOutputDimensions(int32_t index, Dims const* inputDims, int32_t nbInputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 1);
    nvinfer1::Dims const& input = inputDims[0];
    PLUGIN_ASSERT(index == 0);
    nvinfer1::Dims output{};
    output.nbDims = input.nbDims;
    for (int32_t d = 0; d < input.nbDims; ++d)
    {
        if (d == input.nbDims - 2 || d == input.nbDims - 1)
        {
            output.d[d] = int32_t(input.d[d] * mScale);
        }
        else
        {
            output.d[d] = input.d[d];
        }
    }
    return output;
}

int32_t ResizeNearest::initialize() noexcept
{
    return 0;
}

void ResizeNearest::terminate() noexcept {}

void ResizeNearest::destroy() noexcept
{
    delete this;
}

size_t ResizeNearest::getWorkspaceSize(int32_t) const noexcept
{
    return 0;
}

size_t ResizeNearest::getSerializationSize() const noexcept
{
    // scale, dimensions: 3 * 2
    return sizeof(float) + sizeof(int32_t) * 3 * 2;
}

void ResizeNearest::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mScale);
    write(d, mInputDims.d[0]);
    write(d, mInputDims.d[1]);
    write(d, mInputDims.d[2]);
    write(d, mOutputDims.d[0]);
    write(d, mOutputDims.d[1]);
    write(d, mOutputDims.d[2]);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

ResizeNearest::ResizeNearest(void const* data, size_t length)
{
    deserialize(static_cast<int8_t const*>(data), length);
}

void ResizeNearest::deserialize(int8_t const* data, size_t length)
{
    auto const* d{data};
    mScale = read<float>(d);
    mInputDims = Dims3();
    mInputDims.d[0] = read<int32_t>(d);
    mInputDims.d[1] = read<int32_t>(d);
    mInputDims.d[2] = read<int32_t>(d);
    mOutputDims = Dims3();
    mOutputDims.d[0] = read<int32_t>(d);
    mOutputDims.d[1] = read<int32_t>(d);
    mOutputDims.d[2] = read<int32_t>(d);
    PLUGIN_VALIDATE(d == data + length);
}

char const* ResizeNearest::getPluginType() const noexcept
{
    return "ResizeNearest_TRT";
}

char const* ResizeNearest::getPluginVersion() const noexcept
{
    return "1";
}

IPluginV2Ext* ResizeNearest::clone() const noexcept
{
    try
    {
        auto plugin = new ResizeNearest(*this);
        plugin->setPluginNamespace(mNameSpace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void ResizeNearest::setPluginNamespace(char const* libNamespace) noexcept
{
    mNameSpace = libNamespace;
}

char const* ResizeNearest::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

bool ResizeNearest::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

int32_t ResizeNearest::enqueue(
    int32_t batch_size, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{

    int32_t nchan = mOutputDims.d[0];
    float scale = mScale;
    int2 osize = {dimToInt32(mOutputDims.d[2]), dimToInt32(mOutputDims.d[1])};
    int32_t istride = mInputDims.d[2];
    int32_t ostride = mOutputDims.d[2];
    int32_t ibatchstride = mInputDims.d[1] * istride;
    int32_t obatchstride = mOutputDims.d[1] * ostride;
    dim3 block(32, 16);
    dim3 grid((osize.x - 1) / block.x + 1, (osize.y - 1) / block.y + 1, std::min(batch_size * nchan, 65535));

    resizeNearest(grid, block, stream, batch_size * nchan, scale, osize, static_cast<float const*>(inputs[0]), istride,
        ibatchstride, static_cast<float*>(outputs[0]), ostride, obatchstride);

    return cudaGetLastError() != cudaSuccess;
}

// Return the DataType of the plugin output at the requested index
DataType ResizeNearest::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    // Only 1 input and 1 output from the plugin layer
    PLUGIN_ASSERT(index == 0);

    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool ResizeNearest::isOutputBroadcastAcrossBatch(
    int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool ResizeNearest::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void ResizeNearest::configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
    DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
    bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept
{
    PLUGIN_ASSERT(nbInputs == 1);
    mInputDims = inputDims[0];

    PLUGIN_ASSERT(nbOutputs == 1);
    mOutputDims = outputDims[0];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void ResizeNearest::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void ResizeNearest::detachFromContext() noexcept {}
