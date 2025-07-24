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

#ifndef DISENTANGLEDATTENTIONPLUGIN_LEGACY_PLUGIN_H
#define DISENTANGLEDATTENTIONPLUGIN_LEGACY_PLUGIN_H

/*
 * Legacy version of the plugin maintained for backward compatibility.
 * This implementation is based on IPluginV2 interfaces.
 */
#include "NvInferPlugin.h"
#include "common/plugin.h"
#include "common/serialize.hpp"
#include "disentangledAttentionCommon.h"
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.
namespace nvinfer1
{
namespace plugin
{

// using namespace nvinfer1;

class DisentangledAttentionPluginLegacy : public nvinfer1::IPluginV2DynamicExt
{
public:
    DisentangledAttentionPluginLegacy();

    DisentangledAttentionPluginLegacy(int32_t span, float factor);

    DisentangledAttentionPluginLegacy(void const* serialData, size_t serialLength);

    int32_t getNbOutputs() const noexcept override;

    // DynamicExt plugins returns DimsExprs class instead of Dims
    nvinfer1::DimsExprs getOutputDimensions(int32_t index, nvinfer1::DimsExprs const* inputs, int32_t nbInputDims,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override; // determine output dims based on input info

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    // This is where the plugin work is done.
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

private:
    // Helper method for enqueue()
    template <typename TDataType>
    void enqueueType(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, cudaStream_t stream, TDataType factor);

    std::string mNamespace;

    // attributes
    int32_t mSpan;
    float mFactor;

    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2::enqueue;
    using IPluginV2Ext::configurePlugin;
};

class DisentangledAttentionPluginCreatorLegacy : public nvinfer1::IPluginCreator
{
public:
    DisentangledAttentionPluginCreatorLegacy();

    ~DisentangledAttentionPluginCreatorLegacy() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2DynamicExt* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2DynamicExt* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1

#endif // DISENTANGLEDATTENTIONPLUGIN_LEGACY_PLUGIN_H
