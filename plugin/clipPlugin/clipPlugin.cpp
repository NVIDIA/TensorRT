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
#include "clipPlugin.h"
#include "NvInfer.h"
#include "clip.h"
#include "common/checkMacrosPlugin.h"
#include <cstring>
#include <string>
#include <utility>
#include <vector>

// Remove once if common/utils is created (see TRT-17687)
#include <set>

using namespace nvinfer1;
using nvinfer1::plugin::ClipPluginCreator;
using nvinfer1::plugin::ClipPlugin;

static char const* const kCLIP_PLUGIN_VERSION{"1"};
static char const* const kCLIP_PLUGIN_NAME{"Clip_TRT"};

ClipPlugin::ClipPlugin(std::string name, float clipMin, float clipMax)
    : mLayerName(std::move(name))
    , mClipMin(clipMin)
    , mClipMax(clipMax)
{
}

ClipPlugin::ClipPlugin(std::string name, void const* data, size_t length)
    : mLayerName(std::move(name))
{
    // Deserialize in the same order as serialization
    char const *d = static_cast<char const*>(data), *a = d;

    mClipMin = read<float>(d);
    mClipMax = read<float>(d);
    mDataType = read<DataType>(d);
    mInputVolume = read<size_t>(d);

    PLUGIN_VALIDATE(d == (a + length));
}

char const* ClipPlugin::getPluginType() const noexcept
{
    return kCLIP_PLUGIN_NAME;
}

char const* ClipPlugin::getPluginVersion() const noexcept
{
    return kCLIP_PLUGIN_VERSION;
}

int32_t ClipPlugin::getNbOutputs() const noexcept
{
    return 1;
}

Dims ClipPlugin::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept
{
    PLUGIN_ASSERT(nbInputDims == 1);
    PLUGIN_ASSERT(index == 0);
    return *inputs;
}

int32_t ClipPlugin::initialize() noexcept
{
    return 0;
}

int32_t ClipPlugin::enqueue(
    int32_t batchSize, void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept
{
    try
    {
        void* output = outputs[0];
        int32_t status = pluginStatus_t::STATUS_FAILURE;
        status = clipInference(stream, mInputVolume * batchSize, mClipMin, mClipMax, inputs[0], output, mDataType);

        if (status != pluginStatus_t::STATUS_SUCCESS)
        {
            gLogError << "ClipPlugin Kernel failed for layer name " << mLayerName << std::endl;
        }

        return status;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return -1;
}

size_t ClipPlugin::getSerializationSize() const noexcept
{
    return 2 * sizeof(float) + sizeof(mDataType) + sizeof(mInputVolume);
}

void ClipPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;

    // Serialize plugin data
    nvinfer1::plugin::write(d, mClipMin);
    nvinfer1::plugin::write(d, mClipMax);
    nvinfer1::plugin::write(d, mDataType);
    nvinfer1::plugin::write(d, mInputVolume);

    if (d != a + getSerializationSize())
    {
        gLogError << "ClipPlugin serialize failed for layer name " << mLayerName << std::endl;
    }

    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void ClipPlugin::configureWithFormat(Dims const* inputs, int32_t nbInputs, Dims const* outputs, int32_t nbOutputs,
    DataType type, PluginFormat format, int32_t) noexcept
{
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_API_CHECK_ENUM_RANGE(DataType, type);
    PLUGIN_API_CHECK_ENUM_RANGE(PluginFormat, format);
    mDataType = type;

    size_t volume = 1;
    for (int32_t i = 0; i < inputs->nbDims; i++)
    {
        volume *= inputs->d[i];
    }
    mInputVolume = volume;
}

bool ClipPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    if (type == DataType::kINT8)
    {
        return false;
    }

    if (format != PluginFormat::kLINEAR)
    {
        return false;
    }

    PLUGIN_API_CHECK_ENUM_RANGE_RETVAL(DataType, type, false);
    PLUGIN_API_CHECK_ENUM_RANGE_RETVAL(PluginFormat, format, false);
    return true;
}

void ClipPlugin::terminate() noexcept {}

ClipPlugin::~ClipPlugin() = default;

void ClipPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2* ClipPlugin::clone() const noexcept
{
    try
    {
        ClipPlugin* ret = new ClipPlugin(mLayerName, mClipMin, mClipMax);
        ret->mInputVolume = mInputVolume;
        ret->setPluginNamespace(mNamespace.c_str());
        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

ClipPluginCreator::ClipPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("clipMin", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("clipMax", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* ClipPluginCreator::getPluginName() const noexcept
{
    return kCLIP_PLUGIN_NAME;
}

char const* ClipPluginCreator::getPluginVersion() const noexcept
{
    return kCLIP_PLUGIN_VERSION;
}

PluginFieldCollection const* ClipPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* ClipPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        gLogWarning << "ClipPlugin is deprecated since TensorRT 9.0. Use INetworkDefinition::addActivation() to add an "
                       "IActivationLayer with ActivationType::kCLIP."
                    << std::endl;
        float clipMin = 0.0, clipMax = 0.0;
        PluginField const* fields = fc->fields;

        plugin::validateRequiredAttributesExist({"clipMin", "clipMax"}, fc);
        PLUGIN_VALIDATE(fc->nbFields == 2);

        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            if (strcmp(fields[i].name, "clipMin") == 0)
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                clipMin = *(static_cast<float const*>(fields[i].data));
            }
            else if (strcmp(fields[i].name, "clipMax") == 0)
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                clipMax = *(static_cast<float const*>(fields[i].data));
            }
        }

        return new ClipPlugin(name, clipMin, clipMax);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* ClipPluginCreator::deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        gLogWarning << "ClipPlugin is deprecated since TensorRT 9.0. Use INetworkDefinition::addActivation() to add an "
                       "IActivationLayer with ActivationType::kCLIP."
                    << std::endl;
        // This object will be deleted when the network is destroyed, which will
        // call ClipPlugin::destroy()
        return new ClipPlugin(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
