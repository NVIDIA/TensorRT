/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRT_NV_PLUGIN_FASTER_RCNN_H
#define TRT_NV_PLUGIN_FASTER_RCNN_H

#include "cudnn.h"
#include "kernel.h"
#include "plugin.h"
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class RPROIPlugin : public IPluginV2Ext
{
public:
    RPROIPlugin(RPROIParams params, const float* anchorsRatios, const float* anchorsScales);

    RPROIPlugin(RPROIParams params, const float* anchorsRatios, const float* anchorsScales, int A, int C, int H, int W,
        const float* anchorsDev);

    RPROIPlugin(const void* data, size_t length);

    ~RPROIPlugin() override;

    int getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int maxBatchSize) const noexcept override;

    int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    IPluginV2Ext* clone() const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override;

    void detachFromContext() noexcept override;

private:
    float* copyToHost(const void* srcHostData, int count) noexcept;

    int copyFromHost(char* dstHostBuffer, const void* source, int count) const noexcept;

    // These won't be serialized
    float* anchorsDev{nullptr};
    std::string mPluginNamespace;

    // These need to be serialized
    RPROIParams params;
    int A, C, H, W;
    float *anchorsRatiosHost{nullptr}, *anchorsScalesHost{nullptr};
};

class RPROIPluginCreator : public BaseCreator
{
public:
    RPROIPluginCreator();

    ~RPROIPluginCreator() override;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

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
