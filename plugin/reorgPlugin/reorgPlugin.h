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
#ifndef TRT_REORG_PLUGIN_H
#define TRT_REORG_PLUGIN_H
#include "common/kernels/kernel.h"
#include "common/plugin.h"
#include <iostream>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

template <class TBaseClass>
class Reorg : public TBaseClass
{
public:
    Reorg(int32_t stride = 0);
    ~Reorg() override = default;

    int32_t getNbOutputs() const noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    char const* getPluginType() const noexcept override;

    void destroy() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

    void detachFromContext() noexcept override;

protected:
    int32_t stride{};
    std::string mPluginNamespace;
};

class TRT_DEPRECATED ReorgStatic : public Reorg<IPluginV2Ext>
{
public:
    ReorgStatic(int32_t stride);
    ReorgStatic(int32_t C, int32_t H, int32_t W, int32_t stride);
    ReorgStatic(void const* buffer, size_t length);

    char const* getPluginVersion() const noexcept override;
    Dims getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept override;
    size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;
    int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;
    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;
    IPluginV2Ext* clone() const noexcept override;
    bool isOutputBroadcastAcrossBatch(
        int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept override;
    bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;
    void configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
        DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
        bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

protected:
    int32_t C{};
    int32_t H{};
    int32_t W{};
};

class ReorgDynamic : public Reorg<IPluginV2DynamicExt>
{
public:
    ReorgDynamic(int32_t stride);
    ReorgDynamic(void const* buffer, size_t length);

    char const* getPluginVersion() const noexcept override;
    DimsExprs getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    IPluginV2DynamicExt* clone() const noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
};

template <class TPluginClass>
class ReorgPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    ReorgPluginCreator();

    ~ReorgPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    PluginFieldCollection mFC;
    int32_t stride{};
    std::vector<PluginField> mPluginAttributes;
};

using ReorgStaticPluginCreator = ReorgPluginCreator<ReorgStatic>;
using ReorgDynamicPluginCreator = ReorgPluginCreator<ReorgDynamic>;
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_REORG_PLUGIN_H
