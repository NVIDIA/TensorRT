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

#ifndef TRT_DISENTANGLED_ATTENTION_PLUGIN_H
#define TRT_DISENTANGLED_ATTENTION_PLUGIN_H

#include "NvInferPlugin.h"
#include "plugin.h"
#include "serialize.hpp"
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

#define kDISENTANGLED_VERSION 2
// Version 1: regular relative position index
// Version 2: log bucket relative position index
constexpr int32_t kDISENTANGLED_TILESIZE_V1 = 32;
constexpr int32_t kDISENTANGLED_BLOCKDIMY_V1 = 8;
constexpr int32_t kDISENTANGLED_TILESIZE_V2 = 64;
constexpr int32_t kDISENTANGLED_BLOCKDIMY_V2 = 4;

template <typename TDataType, int32_t tTileSize, int32_t tBlockDimY>
void disentangled_kernel_wrapper(TDataType const* data0, TDataType const* data1, TDataType const* data2,
    TDataType* result, dim3 dimData0, dim3 dimData1, dim3 dimData2, dim3 dimResult, TDataType factor, int32_t span,
    dim3 block, dim3 grid, cudaStream_t stream);

class DisentangledAttentionPlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    DisentangledAttentionPlugin();

    DisentangledAttentionPlugin(int32_t span, float factor);

    DisentangledAttentionPlugin(void const* serialData, size_t serialLength);

    ~DisentangledAttentionPlugin() override;

    template <typename TDataType>
    TDataType const* pointer_const_cast(void const* const p);

    template <typename TDataType>
    TDataType* pointer_cast(void* p);

    int32_t getNbOutputs() const noexcept override;

    // DynamicExt plugins returns DimsExprs class instead of Dims
    nvinfer1::DimsExprs getOutputDimensions(int32_t index, nvinfer1::DimsExprs const* inputs, int32_t nbInputDims,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override; // determine output dims based on input info

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override; // this is where the plugin work is done

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

    void attachToContext(
        cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept override;

    void detachFromContext() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

private:
    char const* mPluginNamespace;
    std::string mNamespace;

    // attributes
    int32_t mSpan;
    float mFactor;

    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2::enqueue;
    using IPluginV2Ext::configurePlugin;
};

class DisentangledAttentionPluginCreator : public nvinfer1::IPluginCreator
{
public:
    DisentangledAttentionPluginCreator();

    ~DisentangledAttentionPluginCreator() override = default;

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
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_DISENTANGLED_ATTENTION_PLUGIN_H
