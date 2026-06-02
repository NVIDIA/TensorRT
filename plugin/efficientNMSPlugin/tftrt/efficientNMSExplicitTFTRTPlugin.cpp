/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string_view>

// This plugin provides CombinedNMS op compatibility for TF-TRT in Explicit Batch
// and Dymamic Shape modes

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using nvinfer1::plugin::EfficientNMSExplicitTFTRTPlugin;
using nvinfer1::plugin::EfficientNMSExplicitTFTRTPluginCreator;

namespace
{
const char* EFFICIENT_NMS_EXPLICIT_TFTRT_PLUGIN_VERSION{"1"};
const char* EFFICIENT_NMS_EXPLICIT_TFTRT_PLUGIN_NAME{"EfficientNMS_Explicit_TF_TRT"};
} // namespace

EfficientNMSExplicitTFTRTPlugin::EfficientNMSExplicitTFTRTPlugin(EfficientNMSParameters param)
    : EfficientNMSPlugin(std::move(param))
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
        auto plugin = std::make_unique<EfficientNMSExplicitTFTRTPlugin>(mParam);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& e)
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
    char const* name, PluginFieldCollection const* fc) noexcept
{
    using namespace std::string_view_literals;
    try
    {
        PluginField const* fields = fc->fields;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            std::string_view const attrName = fields[i].name;
            if (attrName == "max_output_size_per_class"sv)
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.numOutputBoxesPerClass = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (attrName == "max_total_size"sv)
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.numOutputBoxes = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (attrName == "iou_threshold"sv)
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                mParam.iouThreshold = *(static_cast<float const*>(fields[i].data));
            }
            if (attrName == "score_threshold"sv)
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                mParam.scoreThreshold = *(static_cast<float const*>(fields[i].data));
            }
            if (attrName == "pad_per_class"sv)
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.padOutputBoxesPerClass = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (attrName == "clip_boxes"sv)
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.clipBoxes = *(static_cast<int32_t const*>(fields[i].data));
            }
        }

        auto plugin = std::make_unique<EfficientNMSExplicitTFTRTPlugin>(mParam);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& e)
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
        auto plugin = std::make_unique<EfficientNMSExplicitTFTRTPlugin>(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
