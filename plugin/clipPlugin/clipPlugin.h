/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef TRT_CLIP_PLUGIN_H
#define TRT_CLIP_PLUGIN_H

#include "NvInferPlugin.h"
#include "common/plugin.h"
#include <cstdlib>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
class TRT_DEPRECATED ClipPlugin : public nvinfer1::pluginInternal::BasePlugin
{
public:
    ClipPlugin(std::string name, float clipMin, float clipMax);

    ClipPlugin(std::string name, void const* data, size_t length);

    ~ClipPlugin() override;

    ClipPlugin() = delete;

    int32_t getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int32_t) const noexcept override
    {
        return 0;
    };

    int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    void configureWithFormat(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
        DataType type, PluginFormat format, int32_t maxBatchSize) noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    IPluginV2* clone() const noexcept override;

private:
    std::string mLayerName;
    float mClipMin{0.0F};
    float mClipMax{0.0F};
    DataType mDataType{DataType::kFLOAT};
    size_t mInputVolume{0};
};

class TRT_DEPRECATED ClipPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    ClipPluginCreator();

    ~ClipPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_CLIP_PLUGIN_H
