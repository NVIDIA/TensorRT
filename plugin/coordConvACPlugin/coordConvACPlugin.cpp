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

#include "coordConvACPlugin.h"
#include <cstring>
#include <iostream>
#include <vector>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace
{
char const* const kCOORDCONV_AC_PLUGIN_VERSION{"1"};
char const* const kCOORDCONV_AC_PLUGIN_NAME{"CoordConvAC"};
int32_t const kNUM_COORDCONV_CHANNELS = 2;
} // namespace

PluginFieldCollection CoordConvACPluginCreator::mFC{};
std::vector<PluginField> CoordConvACPluginCreator::mPluginAttributes;

CoordConvACPlugin::CoordConvACPlugin() {}

CoordConvACPlugin::CoordConvACPlugin(
    nvinfer1::DataType iType, int32_t iC, int32_t iH, int32_t iW, int32_t oC, int32_t oH, int32_t oW)
    : iType(iType)
    , iC(iC)
    , iH(iH)
    , iW(iW)
    , oC(oC)
    , oH(oH)
    , oW(oW)
{
}

CoordConvACPlugin::CoordConvACPlugin(void const* data, size_t length)
{
    deserialize(static_cast<uint8_t const*>(data), length);
}
void CoordConvACPlugin::deserialize(uint8_t const* data, size_t length)
{
    auto const* d{data};
    iType = read<nvinfer1::DataType>(d);
    iC = read<int32_t>(d);
    iH = read<int32_t>(d);
    iW = read<int32_t>(d);
    oC = read<int32_t>(d);
    oH = read<int32_t>(d);
    oW = read<int32_t>(d);
    PLUGIN_VALIDATE(d == data + length);
}

int32_t CoordConvACPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t CoordConvACPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void CoordConvACPlugin::terminate() noexcept {}

Dims CoordConvACPlugin::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept
{
    PLUGIN_ASSERT(index == 0);
    PLUGIN_ASSERT(nbInputDims == 1);
    PLUGIN_ASSERT(inputs != nullptr);
    // CHW
    nvinfer1::Dims dimsOutput;
    // Don't trigger null dereference since we check if inputs is nullptr above.
    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    PLUGIN_ASSERT(inputs[0].nbDims == 3);
    dimsOutput.nbDims = inputs[0].nbDims;
    dimsOutput.d[0] = inputs[0].d[0] + kNUM_COORDCONV_CHANNELS;
    dimsOutput.d[1] = inputs[0].d[1];
    dimsOutput.d[2] = inputs[0].d[2];
    return dimsOutput;
}

size_t CoordConvACPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept
{
    return 0;
}

size_t CoordConvACPlugin::getSerializationSize() const noexcept
{
    // iType, iC, iH, iW, oC, oH, oW
    return sizeof(nvinfer1::DataType) + sizeof(int32_t) * 6;
}

void CoordConvACPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, iType);
    write(d, iC);
    write(d, iH);
    write(d, iW);
    write(d, oC);
    write(d, oH);
    write(d, oW);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void CoordConvACPlugin::configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims,
    int32_t nbOutputs, DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
    bool const* outputIsBroadcast, nvinfer1::PluginFormat format, int32_t maxBatchSize) noexcept
{
    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(nbOutputs == 1);

    iC = inputDims->d[0];
    iH = inputDims->d[1];
    iW = inputDims->d[2];

    oC = outputDims->d[0];
    oH = outputDims->d[1];
    oW = outputDims->d[2];

    iType = inputTypes[0];
}

bool CoordConvACPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return ((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kLINEAR);
}

char const* CoordConvACPlugin::getPluginType() const noexcept
{
    return kCOORDCONV_AC_PLUGIN_NAME;
}

char const* CoordConvACPlugin::getPluginVersion() const noexcept
{
    return kCOORDCONV_AC_PLUGIN_VERSION;
}

void CoordConvACPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2Ext* CoordConvACPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new CoordConvACPlugin(iType, iC, iH, iW, oC, oH, oW);
        plugin->setPluginNamespace(mPluginNamespace);
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void CoordConvACPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

char const* CoordConvACPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

nvinfer1::DataType CoordConvACPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

bool CoordConvACPlugin::isOutputBroadcastAcrossBatch(
    int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    return false;
}

bool CoordConvACPlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{
    return false;
}

// Plugin creator
CoordConvACPluginCreator::CoordConvACPluginCreator() {}

char const* CoordConvACPluginCreator::getPluginName() const noexcept
{
    return kCOORDCONV_AC_PLUGIN_NAME;
}

char const* CoordConvACPluginCreator::getPluginVersion() const noexcept
{
    return kCOORDCONV_AC_PLUGIN_VERSION;
}

PluginFieldCollection const* CoordConvACPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* CoordConvACPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        CoordConvACPlugin* plugin = new CoordConvACPlugin();
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* CoordConvACPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        CoordConvACPlugin* plugin = new CoordConvACPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
