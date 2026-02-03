/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>

#include "NvInferSafePlugin.h"
#include "NvInferSafeRuntime.h"

#include "maxPoolPluginRuntimeCreator.h"

namespace nvinfer1
{
namespace plugin
{

// Plugin creator
MaxPoolRuntimeCreator::MaxPoolRuntimeCreator()
{
    // Declare the ONNX attributes that the ONNX parser will collect from the
    // ONNX model that contains the IdentityConv node.

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField("kernel_shape", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("strides", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("pads", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("pType", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* MaxPoolRuntimeCreator::getPluginNamespace() const noexcept
{
    return kMAX_POOL_PLUGIN_NAMESPACE;
}

char const* MaxPoolRuntimeCreator::getPluginName() const noexcept
{
    return kMAX_POOL_PLUGIN_NAME;
}

char const* MaxPoolRuntimeCreator::getPluginVersion() const noexcept
{
    return kMAX_POOL_PLUGIN_VERSION;
}

nvinfer1::PluginFieldCollection const* MaxPoolRuntimeCreator::getFieldNames() noexcept
{
    // This is only used in the build phase.
    return &mFC;
}

ISafeRecorder* MaxPoolRuntimeCreator::getSafeRecorder() const noexcept
{
    return mRecorder;
}

void MaxPoolRuntimeCreator::setSafeRecorder(ISafeRecorder& recorder) noexcept
{
    mRecorder = &recorder;
}

IPluginV3* MaxPoolRuntimeCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    // MaxPoolRuntimeCreator only handles RUNTIME phase.
    // BUILD phase should use MaxPoolCreator instead.
    if (phase == TensorRTPhase::kRUNTIME)
    {
        nvinfer1::PluginField const* fields{fc->fields};

        PoolParameters params{*(static_cast<PoolParameters const*>(fields[0].data))};

        MaxPoolPluginRuntime* const plugin{new MaxPoolPluginRuntime(params)};
        return plugin;
    }
    return nullptr;
}
} // namespace plugin
} // namespace nvinfer1
