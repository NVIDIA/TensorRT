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
#ifndef TRT_EFFICIENT_NMS_PLUGIN_H
#define TRT_EFFICIENT_NMS_PLUGIN_H

#include <vector>

#include "common/plugin.h"
#include "efficientNMSPlugin/efficientNMSParameters.h"

namespace nvinfer1
{
namespace plugin
{

class EfficientNMSPlugin : public IPluginV2DynamicExt
{
public:
    explicit EfficientNMSPlugin(EfficientNMSParameters param);
    EfficientNMSPlugin(void const* data, size_t length);
    ~EfficientNMSPlugin() override = default;

    // IPluginV2 methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* libNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputType, int32_t nbInputs) const noexcept override;

    // IPluginV2DynamicExt methods
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

protected:
    EfficientNMSParameters mParam{};
    bool initialized{false};
    std::string mNamespace;

private:
    void deserialize(int8_t const* data, size_t length);
};

// Standard NMS Plugin Operation
class EfficientNMSPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    EfficientNMSPluginCreator();
    ~EfficientNMSPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;
    IPluginV2DynamicExt* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

protected:
    PluginFieldCollection mFC;
    EfficientNMSParameters mParam;
    std::vector<PluginField> mPluginAttributes;
    std::string mPluginName;
};

// ONNX NonMaxSuppression Op Compatibility
class EfficientNMSONNXPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    EfficientNMSONNXPluginCreator();
    ~EfficientNMSONNXPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;
    IPluginV2DynamicExt* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

protected:
    PluginFieldCollection mFC;
    EfficientNMSParameters mParam;
    std::vector<PluginField> mPluginAttributes;
    std::string mPluginName;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_EFFICIENT_NMS_PLUGIN_H
