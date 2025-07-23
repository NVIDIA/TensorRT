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
#include "batchTilePlugin.h"
#include "common/dimsHelpers.h"
#include "common/templates.h"

#include <cuda_runtime.h>

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using nvinfer1::plugin::BatchTilePlugin;
using nvinfer1::plugin::BatchTilePluginCreator;

namespace
{
static char const* const kBATCH_TILE_PLUGIN_VERSION{"1"};
static char const* const kBATCH_TILE_PLUGIN_NAME{"BatchTilePlugin_TRT"};
} // namespace

BatchTilePlugin::BatchTilePlugin(std::string const& name)
    : mLayerName(name)
{
}

BatchTilePlugin::BatchTilePlugin(std::string const& name, size_t copy_size)
    : mLayerName(name)
    , mCopySize(copy_size)
{
}

BatchTilePlugin::BatchTilePlugin(std::string const& name, void const* data, size_t length)
    : mLayerName(name)
{
    auto const* d{toPointer<uint8_t const>(data)};
    auto const* a{d};
    mCopySize = readFromBuffer<size_t>(d);
    PLUGIN_VALIDATE(d == a + length);
}

int32_t BatchTilePlugin::getNbOutputs() const noexcept
{
    return 1;
}

Dims BatchTilePlugin::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(nbInputDims == 2);
        PLUGIN_VALIDATE(index == 0);
        PLUGIN_VALIDATE(inputs[1].nbDims == 4);
        return Dims3(inputs[1].d[1], inputs[1].d[2], inputs[1].d[3]);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return Dims{};
}

int32_t BatchTilePlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

size_t BatchTilePlugin::getWorkspaceSize(int32_t) const noexcept
{
    return 0;
}

DataType BatchTilePlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(index == 0);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }

    return DataType::kFLOAT;
}

int32_t BatchTilePlugin::enqueue(
    int32_t batchSize, void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);
        uint8_t* output = reinterpret_cast<uint8_t*>(outputs[0]);
        // expand to batch size
        for (int32_t i = 0; i < batchSize; i++)
        {
            PLUGIN_CHECK_CUDA(
                cudaMemcpyAsync(output + i * mCopySize, inputs[1], mCopySize, cudaMemcpyDeviceToDevice, stream));
        }
        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

void BatchTilePlugin::serialize(void* buffer) const noexcept
{
    uint8_t* d = reinterpret_cast<uint8_t*>(buffer);
    uint8_t* const a = d;
    writeToBuffer<size_t>(d, mCopySize);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void BatchTilePlugin::terminate() noexcept {}

size_t BatchTilePlugin::getSerializationSize() const noexcept
{
    return sizeof(mCopySize);
}

bool BatchTilePlugin::isOutputBroadcastAcrossBatch(
    int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    return false;
}

bool BatchTilePlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{
    return false;
}

void BatchTilePlugin::configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims,
    int32_t nbOutputs, DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
    bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDims != nullptr);
        PLUGIN_VALIDATE(outputDims != nullptr);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(inputDims[1].nbDims == 4);
        PLUGIN_VALIDATE(inputDims[1].d[0] == 1);
        mCopySize = pluginInternal::volume(inputDims[1]) * sizeof(float);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

bool BatchTilePlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

char const* BatchTilePlugin::getPluginType() const noexcept
{
    return kBATCH_TILE_PLUGIN_NAME;
}
char const* BatchTilePlugin::getPluginVersion() const noexcept
{
    return kBATCH_TILE_PLUGIN_VERSION;
}

void BatchTilePlugin::destroy() noexcept
{
    delete this;
}

IPluginV2Ext* BatchTilePlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new BatchTilePlugin(mLayerName, mCopySize);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void BatchTilePlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* BatchTilePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

BatchTilePluginCreator::BatchTilePluginCreator()
{
    mFC.nbFields = 0;
    mFC.fields = nullptr;
}

char const* BatchTilePluginCreator::getPluginName() const noexcept
{
    return kBATCH_TILE_PLUGIN_NAME;
}

char const* BatchTilePluginCreator::getPluginVersion() const noexcept
{
    return kBATCH_TILE_PLUGIN_VERSION;
}

PluginFieldCollection const* BatchTilePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* BatchTilePluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        gLogWarning << "BatchTilePlugin is deprecated since TensorRT 9.0. Use INetworkDefinition::addSlice() to add an "
                       "ISliceLayer with SampleMode::kWRAP."
                    << std::endl;
        PLUGIN_VALIDATE(name != nullptr);
        auto* plugin = new BatchTilePlugin(name);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void BatchTilePluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

IPluginV2Ext* BatchTilePluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        gLogWarning << "BatchTilePlugin is deprecated since TensorRT 9.0. Use INetworkDefinition::addSlice() to add an "
                       "ISliceLayer with SampleMode::kWRAP."
                    << std::endl;
        PLUGIN_VALIDATE(name != nullptr);
        return new BatchTilePlugin(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
