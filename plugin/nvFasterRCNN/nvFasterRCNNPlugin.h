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
#ifndef TRT_NV_PLUGIN_FASTER_RCNN_H
#define TRT_NV_PLUGIN_FASTER_RCNN_H

#include "common/kernels/kernel.h"
#include "common/plugin.h"
#include "cudnn.h"
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class RPROIPlugin : public IPluginV2IOExt
{
public:
    RPROIPlugin(RPROIParams params, float const* anchorsRatios, float const* anchorsScales);

    RPROIPlugin(RPROIParams params, float const* anchorsRatios, float const* anchorsScales, int32_t A, int32_t C,
        int32_t H, int32_t W, float const* anchorsDev, size_t deviceSmemSize, DataType inFeatureType,
        DataType outFeatureType, DLayout_t inFeatureLayout);

    RPROIPlugin(void const* data, size_t length);

    ~RPROIPlugin() override;

    int getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int index, Dims const* inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int maxBatchSize) const noexcept override;

    int enqueue(int batchSize, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) const noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    IPluginV2Ext* clone() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept override;

    bool isOutputBroadcastAcrossBatch(
        int outputIndex, bool const* inputIsBroadcasted, int nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

    void configurePlugin(
        PluginTensorDesc const* in, int32_t nbInput, PluginTensorDesc const* out, int32_t nbOutput) noexcept override;

    void detachFromContext() noexcept override;

private:
    void deserialize(int8_t const* data, size_t length);
    float* copyToHost(void const* srcHostData, int count) noexcept;

    int copyFromHost(char* dstHostBuffer, void const* source, int count) const noexcept;

    size_t getSmemSize() const noexcept;

    DLayout_t convertTensorFormat(TensorFormat const& srcFormat) const noexcept;

    // These won't be serialized
    float* anchorsDev{};
    std::string mPluginNamespace;
    const int32_t PluginNbInputs{4};
    const int32_t PluginNbOutputs{2};
    // this plugin may load the whole feature map in smem. we can set different smem size according to the device.
    size_t deviceSmemSize{};

    // These need to be serialized
    RPROIParams params{};
    int32_t A{};
    int32_t C{};
    int32_t H{};
    int32_t W{};
    float* anchorsRatiosHost{};
    float* anchorsScalesHost{};
    DataType inFeatureType{};
    DataType outFeatureType{};
    DLayout_t inFeatureLayout{};
};

class RPROIPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    RPROIPluginCreator();

    ~RPROIPluginCreator() override;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    RPROIParams params;
    std::vector<float> anchorsRatios;
    std::vector<float> anchorsScales;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_NV_PLUGIN_FASTER_RCNN_H
