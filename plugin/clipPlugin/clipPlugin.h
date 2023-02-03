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
#ifndef TRT_CLIP_PLUGIN_H
#define TRT_CLIP_PLUGIN_H

#include "NvInferPlugin.h"
#include "common/plugin.h"
#include <cstdlib>
#include <cudnn.h>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
class ClipPlugin : public nvinfer1::pluginInternal::BasePlugin
{
public:
    ClipPlugin(std::string name, float clipMin, float clipMax);

    ClipPlugin(std::string name, const void* data, size_t length);

    ~ClipPlugin() override;

    ClipPlugin() = delete;

    int getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int) const noexcept override
    {
        return 0;
    };

    int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    IPluginV2* clone() const noexcept override;

private:
    std::string mLayerName;
    float mClipMin{0.0f};
    float mClipMax{0.0f};
    DataType mDataType{DataType::kFLOAT};
    size_t mInputVolume{0};
};

class ClipPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    ClipPluginCreator();

    ~ClipPluginCreator() override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin
} //namespace nvinfer1

#endif // TRT_CLIP_PLUGIN_H
