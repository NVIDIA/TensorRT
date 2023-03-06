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

#ifndef TRT_COORDCONV_PLUGIN_H
#define TRT_COORDCONV_PLUGIN_H

#include "NvInferPlugin.h"
#include "common/kernels/kernel.h"
#include "common/plugin.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class CoordConvACPlugin : public IPluginV2Ext
{
public:
    CoordConvACPlugin();

    CoordConvACPlugin(DataType iType, int iC, int iH, int iW, int oC, int oH, int oW);

    CoordConvACPlugin(void const* data, size_t length);

    ~CoordConvACPlugin() override = default;

    int getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int index, Dims const* inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int maxBatchSize) const noexcept override;

    int enqueue(int batchSize, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    void configurePlugin(Dims const* inputDims, int nbInputs, Dims const* outputDims, int nbOutputs,
        DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
        bool const* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    IPluginV2Ext* clone() const noexcept override;

    nvinfer1::DataType getOutputDataType(
        int index, nvinfer1::DataType const* inputType, int nbInputs) const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    bool isOutputBroadcastAcrossBatch(
        int outputIndex, bool const* inputIsBroadcasted, int nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

private:
    void deserialize(uint8_t const* data, size_t length);
    DataType iType{};
    int32_t iC{};
    int32_t iH{};
    int32_t iW{};
    int32_t oC{};
    int32_t oH{};
    int32_t oW{};
    char const* mPluginNamespace{};
};

class CoordConvACPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    CoordConvACPluginCreator();

    ~CoordConvACPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;

protected:
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_COORDCONV_PLUGIN_H
