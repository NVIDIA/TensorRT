/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TENSORRT_MAX_POOL_RUNTIME_PLUGIN_CREATOR_H
#define TENSORRT_MAX_POOL_RUNTIME_PLUGIN_CREATOR_H

#include <vector>

#include "NvInferSafePlugin.h"
#include "NvInferSafeRuntime.h"

#include "maxPoolPluginRuntime.h"

using namespace nvinfer2::safe;

namespace nvinfer1
{
namespace plugin
{
// Plugin factory class.
class MaxPoolRuntimeCreator : public ISafePluginCreatorV3One
{
public:
    MaxPoolRuntimeCreator();

    ~MaxPoolRuntimeCreator() override = default;

    char const* getPluginNamespace() const noexcept override;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;

    ISafeRecorder* getSafeRecorder() const noexcept override;

    void setSafeRecorder(ISafeRecorder&) noexcept override;

protected:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    ISafeRecorder* mRecorder;
};
} // namespace plugin
} // namespace nvinfer1

#ifdef GEN_PLUGIN_LIB
extern "C" IPluginCreatorInterface* getSafetyPluginCreator(char const* pluginNamespace, char const* pluginName);
#endif
#endif // TENSORRT_MAX_POOL_RUNTIME_PLUGIN_CREATOR_H
