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
#ifndef TRT_PRIOR_BOX_PLUGIN_H
#define TRT_PRIOR_BOX_PLUGIN_H
#include "common/kernel.h"
#include "common/plugin.h"
#include <cstdlib>
#include <cublas_v2.h>
#include <cudnn.h>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class PriorBox : public IPluginV2Ext
{
public:
    PriorBox(PriorBoxParameters param, int32_t H = 0, int32_t W = 0);

    PriorBox(const void* buffer, size_t length);

    ~PriorBox() override = default;

    int32_t getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims) noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override {};

    size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;

    int32_t enqueue(int32_t batchSize, const void* const* inputs, void* const* outputs, void* workspace,
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

    DataType getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept override;

    bool isOutputBroadcastAcrossBatch(int32_t outputIndex, const bool* inputIsBroadcasted, int32_t nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

    void configurePlugin(const Dims* inputDims, int32_t nbInputs, const Dims* outputDims, int32_t nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept override;

    void detachFromContext() noexcept override;

private:
    void setupDeviceMemory() noexcept;

    PriorBoxParameters mParam;
    int32_t mNumPriors;
    int32_t mH;
    int32_t mW;
    Weights minSize{};      // not learnable weights
    Weights maxSize{};      // not learnable weights
    Weights aspectRatios{}; // not learnable weights
    std::string mPluginNamespace;
};

class PriorBoxPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    PriorBoxPluginCreator();

    ~PriorBoxPluginCreator() override;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_PRIOR_BOX_PLUGIN_H
