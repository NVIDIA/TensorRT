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

#include "efficientNMSExplicitTFTRTPlugin.h"
#include "efficientNMSPlugin/efficientNMSInference.h"

// This plugin provides CombinedNMS op compatibility for TF-TRT in Explicit Batch
// and Dymamic Shape modes

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using nvinfer1::plugin::EfficientNMSExplicitTFTRTPlugin;
using nvinfer1::plugin::EfficientNMSParameters;
using nvinfer1::plugin::EfficientNMSExplicitTFTRTPluginCreator;

namespace
{
const char* EFFICIENT_NMS_EXPLICIT_TFTRT_PLUGIN_VERSION{"1"};
const char* EFFICIENT_NMS_EXPLICIT_TFTRT_PLUGIN_NAME{"EfficientNMS_Explicit_TF_TRT"};
} // namespace

EfficientNMSExplicitTFTRTPlugin::EfficientNMSExplicitTFTRTPlugin(EfficientNMSParameters param)
    : EfficientNMSPlugin(param)
{
}

EfficientNMSExplicitTFTRTPlugin::EfficientNMSExplicitTFTRTPlugin(const void* data, size_t length)
    : EfficientNMSPlugin(data, length)
{
}

const char* EfficientNMSExplicitTFTRTPlugin::getPluginType() const noexcept
{
    return EFFICIENT_NMS_EXPLICIT_TFTRT_PLUGIN_NAME;
}

const char* EfficientNMSExplicitTFTRTPlugin::getPluginVersion() const noexcept
{
    return EFFICIENT_NMS_EXPLICIT_TFTRT_PLUGIN_VERSION;
}

IPluginV2DynamicExt* EfficientNMSExplicitTFTRTPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new EfficientNMSExplicitTFTRTPlugin(mParam);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

EfficientNMSExplicitTFTRTPluginCreator::EfficientNMSExplicitTFTRTPluginCreator()
    : mParam{}
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("max_output_size_per_class", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_total_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("pad_per_class", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("clip_boxes", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* EfficientNMSExplicitTFTRTPluginCreator::getPluginName() const noexcept
{
    return EFFICIENT_NMS_EXPLICIT_TFTRT_PLUGIN_NAME;
}

const char* EfficientNMSExplicitTFTRTPluginCreator::getPluginVersion() const noexcept
{
    return EFFICIENT_NMS_EXPLICIT_TFTRT_PLUGIN_VERSION;
}

const PluginFieldCollection* EfficientNMSExplicitTFTRTPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* EfficientNMSExplicitTFTRTPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "max_output_size_per_class"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.numOutputBoxesPerClass = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "max_total_size"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.numOutputBoxes = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "iou_threshold"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                mParam.iouThreshold = *(static_cast<const float*>(fields[i].data));
            }
            if (!strcmp(attrName, "score_threshold"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                mParam.scoreThreshold = *(static_cast<const float*>(fields[i].data));
            }
            if (!strcmp(attrName, "pad_per_class"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.padOutputBoxesPerClass = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "clip_boxes"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.clipBoxes = *(static_cast<const int*>(fields[i].data));
            }
        }

        auto* plugin = new EfficientNMSExplicitTFTRTPlugin(mParam);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* EfficientNMSExplicitTFTRTPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call EfficientNMSPlugin::destroy()
        auto* plugin = new EfficientNMSExplicitTFTRTPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
