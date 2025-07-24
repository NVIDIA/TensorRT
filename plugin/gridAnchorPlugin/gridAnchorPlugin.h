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

#ifndef TRT_GRID_ANCHOR_PLUGIN_H
#define TRT_GRID_ANCHOR_PLUGIN_H
#include "common/kernels/kernel.h"
#include "common/plugin.h"
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
class GridAnchorGenerator : public IPluginV2Ext
{
public:
    GridAnchorGenerator(GridAnchorParameters const* param, int32_t numLayers, char const* version);

    GridAnchorGenerator(void const* data, size_t length, char const* version);

    ~GridAnchorGenerator() override;

    int32_t getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;

    int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    IPluginV2Ext* clone() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    bool isOutputBroadcastAcrossBatch(
        int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

    void configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
        DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
        bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept override;

    void detachFromContext() noexcept override;

protected:
    std::string mPluginName;

private:
    Weights copyToDevice(void const* hostData, size_t count) noexcept;

    void serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const noexcept;

    Weights deserializeToDevice(char const*& hostBuffer, size_t count) noexcept;

    int32_t mNumLayers;
    std::vector<GridAnchorParameters> mParam;
    int32_t* mNumPriors;
    Weights *mDeviceWidths, *mDeviceHeights;
    std::string mPluginNamespace;
};

class GridAnchorBasePluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    GridAnchorBasePluginCreator();

    ~GridAnchorBasePluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;

protected:
    std::string mPluginName;

private:
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
};

class GridAnchorPluginCreator : public GridAnchorBasePluginCreator
{
public:
    GridAnchorPluginCreator();
    ~GridAnchorPluginCreator() override = default;
};

class GridAnchorRectPluginCreator : public GridAnchorBasePluginCreator
{
public:
    GridAnchorRectPluginCreator();
    ~GridAnchorRectPluginCreator() override = default;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_GRID_ANCHOR_PLUGIN_H
