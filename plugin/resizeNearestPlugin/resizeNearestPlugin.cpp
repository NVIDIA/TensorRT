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
const char* RESIZE_PLUGIN_VERSION{"1"};
const char* RESIZE_PLUGIN_NAME{"ResizeNearest_TRT"};
} // namespace

PluginFieldCollection ResizeNearestPluginCreator::mFC{};
std::vector<PluginField> ResizeNearestPluginCreator::mPluginAttributes;

ResizeNearestPluginCreator::ResizeNearestPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("scale", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ResizeNearestPluginCreator::getPluginName() const noexcept
{
    return RESIZE_PLUGIN_NAME;
}

const char* ResizeNearestPluginCreator::getPluginVersion() const noexcept
{
    return RESIZE_PLUGIN_VERSION;
}

const PluginFieldCollection* ResizeNearestPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* ResizeNearestPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        plugin::validateRequiredAttributesExist({"scale"}, fc);
        const PluginField* fields = fc->fields;
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

IPluginV2Ext* ResizeNearestPluginCreator::deserializePlugin(const char* name, const void* data, size_t length) noexcept
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

int ResizeNearest::getNbOutputs() const noexcept
{
    return 1;
}

Dims ResizeNearest::getOutputDimensions(int index, const Dims* inputDims, int nbInputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 1);
    nvinfer1::Dims const& input = inputDims[0];
    PLUGIN_ASSERT(index == 0);
    nvinfer1::Dims output;
    output.nbDims = input.nbDims;
    for (int d = 0; d < input.nbDims; ++d)
    {
        if (d == input.nbDims - 2 || d == input.nbDims - 1)
        {
            output.d[d] = int(input.d[d] * mScale);
        }
        else
        {
            output.d[d] = input.d[d];
        }
    }
    return output;
}

int ResizeNearest::initialize() noexcept
{
    return 0;
}

void ResizeNearest::terminate() noexcept
{
}

void ResizeNearest::destroy() noexcept
{
    delete this;
}

size_t ResizeNearest::getWorkspaceSize(int) const noexcept
{
    return 0;
}

size_t ResizeNearest::getSerializationSize() const noexcept
{
    // scale, dimensions: 3 * 2
    return sizeof(float) + sizeof(int) * 3 * 2;
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

ResizeNearest::ResizeNearest(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mScale = read<float>(d);
    mInputDims = Dims3();
    mInputDims.d[0] = read<int>(d);
    mInputDims.d[1] = read<int>(d);
    mInputDims.d[2] = read<int>(d);
    mOutputDims = Dims3();
    mOutputDims.d[0] = read<int>(d);
    mOutputDims.d[1] = read<int>(d);
    mOutputDims.d[2] = read<int>(d);
    PLUGIN_VALIDATE(d == a + length);
}

const char* ResizeNearest::getPluginType() const noexcept
{
    return "ResizeNearest_TRT";
}

const char* ResizeNearest::getPluginVersion() const noexcept
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

void ResizeNearest::setPluginNamespace(const char* libNamespace) noexcept
{
    mNameSpace = libNamespace;
}

const char* ResizeNearest::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

bool ResizeNearest::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

int ResizeNearest::enqueue(
    int batch_size, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{

    int nchan = mOutputDims.d[0];
    float scale = mScale;
    int2 osize = {mOutputDims.d[2], mOutputDims.d[1]};
    int istride = mInputDims.d[2];
    int ostride = mOutputDims.d[2];
    int ibatchstride = mInputDims.d[1] * istride;
    int obatchstride = mOutputDims.d[1] * ostride;
    dim3 block(32, 16);
    dim3 grid((osize.x - 1) / block.x + 1, (osize.y - 1) / block.y + 1, std::min(batch_size * nchan, 65535));

    resizeNearest(grid, block, stream, batch_size * nchan, scale, osize, static_cast<float const*>(inputs[0]), istride,
        ibatchstride, static_cast<float*>(outputs[0]), ostride, obatchstride);

    return cudaGetLastError() != cudaSuccess;
}

// Return the DataType of the plugin output at the requested index
DataType ResizeNearest::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // Only 1 input and 1 output from the plugin layer
    PLUGIN_ASSERT(index == 0);

    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool ResizeNearest::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool ResizeNearest::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void ResizeNearest::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
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
