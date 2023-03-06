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
#ifndef TRT_FLATTENCONCAT_PLUGIN_H
#define TRT_FLATTENCONCAT_PLUGIN_H

#include "NvInferPlugin.h"
#include "common/plugin.h"
#include <cstdlib>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
class FlattenConcat : public IPluginV2Ext
{
public:
    FlattenConcat(int concatAxis, bool ignoreBatch);

    FlattenConcat(int concatAxis, bool ignoreBatch, int numInputs, int outputConcatAxis, int const* inputConcatAxis,
        size_t const* copySize, nvinfer1::Dims const& chwDims);

    FlattenConcat(void const* data, size_t length);

    ~FlattenConcat() override;

    FlattenConcat() = delete;

    int getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int index, Dims const* inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int) const noexcept override;

    int enqueue(int batchSize, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    DataType getOutputDataType(int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool isOutputBroadcastAcrossBatch(
        int outputIndex, bool const* inputIsBroadcasted, int nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

    void configurePlugin(Dims const* inputDims, int nbInputs, Dims const* outputDims, int nbOutputs,
        DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
        bool const* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    void detachFromContext() noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;
    IPluginV2Ext* clone() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

private:
    Weights copyToDevice(void const* hostData, size_t count) noexcept;

    void serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const noexcept;

    Weights deserializeToDevice(char const*& hostBuffer, size_t count) noexcept;

    std::vector<size_t> mCopySize;
    std::vector<int> mInputConcatAxis;
    bool mIgnoreBatch{false};
    int mConcatAxisID{0}, mOutputConcatAxis{0}, mNumInputs{0};
    nvinfer1::Dims mCHW;
    std::string mPluginNamespace;
    cublasHandle_t mCublas{nullptr};
};

class FlattenConcatPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    FlattenConcatPluginCreator();

    ~FlattenConcatPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    bool mIgnoreBatch{false};
    int mConcatAxisID;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_FLATTENCONCAT_PLUGIN_H
