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
#include "clipPlugin.h"
#include "NvInfer.h"
#include "clip.h"
#include "common/checkMacrosPlugin.h"
#include <cstring>
#include <cudnn.h>
#include <string>
#include <utility>
#include <vector>

// Remove once if common/utils is created (see TRT-17687)
#include <set>

using namespace nvinfer1;
using nvinfer1::plugin::ClipPluginCreator;
using nvinfer1::plugin::ClipPlugin;

static const char* CLIP_PLUGIN_VERSION{"1"};
static const char* CLIP_PLUGIN_NAME{"Clip_TRT"};
PluginFieldCollection ClipPluginCreator::mFC{};
std::vector<PluginField> ClipPluginCreator::mPluginAttributes;

ClipPlugin::ClipPlugin(std::string name, float clipMin, float clipMax)
    : mLayerName(std::move(name))
    , mClipMin(clipMin)
    , mClipMax(clipMax)
{
}

ClipPlugin::ClipPlugin(std::string name, const void* data, size_t length)
    : mLayerName(std::move(name))
{
    // Deserialize in the same order as serialization
    const char *d = static_cast<const char *>(data), *a = d;

    mClipMin = read<float>(d);
    mClipMax = read<float>(d);
    mDataType = read<DataType>(d);
    mInputVolume = read<size_t>(d);

    PLUGIN_VALIDATE(d == (a + length));
}

const char* ClipPlugin::getPluginType() const noexcept
{
    return CLIP_PLUGIN_NAME;
}

const char* ClipPlugin::getPluginVersion() const noexcept
{
    return CLIP_PLUGIN_VERSION;
}

int ClipPlugin::getNbOutputs() const noexcept
{
    return 1;
}

Dims ClipPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    PLUGIN_ASSERT(nbInputDims == 1);
    PLUGIN_ASSERT(index == 0);
    return *inputs;
}

int ClipPlugin::initialize() noexcept
{
    return 0;
}

int ClipPlugin::enqueue(
    int batchSize, const void* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept
{
    try
    {
        void* output = outputs[0];
        int status = pluginStatus_t::STATUS_FAILURE;
        status = clipInference(stream, mInputVolume * batchSize, mClipMin, mClipMax, inputs[0], output, mDataType);

        if (status != pluginStatus_t::STATUS_SUCCESS)
        {
            gLogError << "ClipPlugin Kernel failed for layer name " << mLayerName << std::endl;
        }

        return status;
    }
    catch (const std::exception& e)
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
    char *d = static_cast<char *>(buffer), *a = d;

    //Serialize plugin data
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

void ClipPlugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs,
    DataType type, PluginFormat format, int) noexcept
{
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_API_CHECK_ENUM_RANGE(DataType, type);
    PLUGIN_API_CHECK_ENUM_RANGE(PluginFormat, format);
    mDataType = type;

    size_t volume = 1;
    for (int i = 0; i < inputs->nbDims; i++)
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

void ClipPlugin::terminate() noexcept
{
}

ClipPlugin::~ClipPlugin() = default;

void ClipPlugin::destroy() noexcept { delete this; }

IPluginV2* ClipPlugin::clone() const noexcept
{
    try
    {
        ClipPlugin* ret = new ClipPlugin(mLayerName, mClipMin, mClipMax);
        ret->mInputVolume = mInputVolume;
        ret->setPluginNamespace(mNamespace.c_str());
        return ret;
    }
    catch (const std::exception& e)
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

const char* ClipPluginCreator::getPluginName() const noexcept
{
    return CLIP_PLUGIN_NAME;
}

const char* ClipPluginCreator::getPluginVersion() const noexcept
{
    return CLIP_PLUGIN_VERSION;
}

const PluginFieldCollection* ClipPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* ClipPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        float clipMin = 0.0, clipMax = 0.0;
        const PluginField* fields = fc->fields;

        plugin::validateRequiredAttributesExist({"clipMin", "clipMax"}, fc);
        PLUGIN_VALIDATE(fc->nbFields == 2);

        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            if (strcmp(fields[i].name, "clipMin") == 0)
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                clipMin = *(static_cast<const float*>(fields[i].data));
            }
            else if (strcmp(fields[i].name, "clipMax") == 0)
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                clipMax = *(static_cast<const float*>(fields[i].data));
            }
        }

        return new ClipPlugin(name, clipMin, clipMax);
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* ClipPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call ClipPlugin::destroy()
        return new ClipPlugin(name, serialData, serialLength);
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
