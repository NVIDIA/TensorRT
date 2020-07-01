/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "coordConvACPlugin.h"
#include <cstring>
#include <iostream>
#include <vector>

using namespace nvinfer1;
const int NUM_COORDCONV_CHANNELS = 2;

namespace
{
const char* COORDCONV_AC_PLUGIN_VERSION{"1"};
const char* COORDCONV_AC_PLUGIN_NAME{"CoordConvAC"};
} // namespace

PluginFieldCollection CoordConvACPluginCreator::mFC{};
std::vector<PluginField> CoordConvACPluginCreator::mPluginAttributes;

CoordConvACPlugin::CoordConvACPlugin() {}

CoordConvACPlugin::CoordConvACPlugin(nvinfer1::DataType iType, int iC, int iH, int iW, int oC, int oH, int oW)
    : iType(iType)
    , iC(iC)
    , iH(iH)
    , iW(iW)
    , oC(oC)
    , oH(oH)
    , oW(oW)
{
}

CoordConvACPlugin::CoordConvACPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    iC = read<int>(d);
    iH = read<int>(d);
    iW = read<int>(d);
    oC = read<int>(d);
    oH = read<int>(d);
    oW = read<int>(d);
    ASSERT(d == a + length);
}

int CoordConvACPlugin::getNbOutputs() const
{
    return 1;
}

int CoordConvACPlugin::initialize()
{
    return STATUS_SUCCESS;
}

void CoordConvACPlugin::terminate() {}

Dims CoordConvACPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // CHW
    nvinfer1::Dims dimsOutput;
    dimsOutput.nbDims = inputs->nbDims;
    dimsOutput.d[0] = inputs->d[0] + NUM_COORDCONV_CHANNELS;
    dimsOutput.d[1] = inputs->d[1];
    dimsOutput.d[2] = inputs->d[2];
    dimsOutput.d[3] = inputs->d[3];
    return dimsOutput;
}

size_t CoordConvACPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

size_t CoordConvACPlugin::getSerializationSize() const
{
    // iC, iH, iW, oC, oH, oW
    return sizeof(int) * 6;
}

void CoordConvACPlugin::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, iC);
    write(d, iH);
    write(d, iW);
    write(d, oC);
    write(d, oH);
    write(d, oW);
    ASSERT(d == a + getSerializationSize());
}

void CoordConvACPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, nvinfer1::PluginFormat format, int maxBatchSize)
{
    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);

    iC = inputDims->d[0];
    iH = inputDims->d[1];
    iW = inputDims->d[2];

    oC = outputDims->d[0];
    oH = outputDims->d[1];
    oW = outputDims->d[2];

    iType = inputTypes[0];
}

bool CoordConvACPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return ((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);
}

const char* CoordConvACPlugin::getPluginType() const
{
    return COORDCONV_AC_PLUGIN_NAME;
}

const char* CoordConvACPlugin::getPluginVersion() const
{
    return COORDCONV_AC_PLUGIN_VERSION;
}

void CoordConvACPlugin::destroy()
{
    delete this;
}

IPluginV2Ext* CoordConvACPlugin::clone() const
{
    auto* plugin = new CoordConvACPlugin(iType, iC, iH, iW, oC, oH, oW);
    return plugin;
}

void CoordConvACPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* CoordConvACPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

nvinfer1::DataType CoordConvACPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return inputTypes[0];
}

bool CoordConvACPlugin::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool CoordConvACPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Plugin creator
CoordConvACPluginCreator::CoordConvACPluginCreator() {}

const char* CoordConvACPluginCreator::getPluginName() const
{
    return COORDCONV_AC_PLUGIN_NAME;
}

const char* CoordConvACPluginCreator::getPluginVersion() const
{
    return COORDCONV_AC_PLUGIN_VERSION;
}

const PluginFieldCollection* CoordConvACPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* CoordConvACPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    CoordConvACPlugin* plugin = new CoordConvACPlugin();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* CoordConvACPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    CoordConvACPlugin* plugin = new CoordConvACPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
