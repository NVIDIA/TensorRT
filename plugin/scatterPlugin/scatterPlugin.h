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
#ifndef TRT_SCATTER_PLUGIN_H
#define TRT_SCATTER_PLUGIN_H
#include "common/kernels/kernel.h"
#include "common/plugin.h"
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class ScatterND : public IPluginV2DynamicExt
{
public:
    ScatterND();

    ~ScatterND() override = default;

    int32_t getNbOutputs() const noexcept override;

    DimsExprs getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    IPluginV2DynamicExt* clone() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

    void detachFromContext() noexcept override;

private:
    // calculate how many slices we need to scatter = reduce_mul(indexTensor.shape[:-1])
    int32_t calculateNumSlices(Dims indexTensorDims) const noexcept;
    int32_t calculateCopySize(Dims const& dataDims) const noexcept;
    void calculateTransformCoeff(Dims const& dataTensorDims, int32_t indexRank, int32_t* transformCoeff) const noexcept;
    std::string mPluginNamespace;

    static constexpr int32_t indexTensorIdx = 1;
    static constexpr int32_t updateTensorIdx = 2;
    static constexpr int32_t dataTensorIdx = 0;
};

class ScatterNDPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    ScatterNDPluginCreator();

    ~ScatterNDPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    PluginFieldCollection mFC;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_SCATTER_PLUGIN_H
