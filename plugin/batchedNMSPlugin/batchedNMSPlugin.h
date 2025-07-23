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
#ifndef TRT_BATCHED_NMS_PLUGIN_H
#define TRT_BATCHED_NMS_PLUGIN_H
#include "batchedNMSPlugin/gatherNMSOutputs.h"
#include "common/kernels/kernel.h"
#include "common/nmsUtils.h"
#include "common/plugin.h"
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class TRT_DEPRECATED BatchedNMSPlugin : public IPluginV2Ext
{
public:
    BatchedNMSPlugin(NMSParameters param);
    BatchedNMSPlugin(void const* data, size_t length);
    ~BatchedNMSPlugin() override = default;

    // IPluginV2 methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    Dims getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept override;
    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;
    size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;
    int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* libNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setClipParam(bool clip) noexcept;
    void setScoreBits(int32_t scoreBits) noexcept;
    void setCaffeSemantics(bool caffeSemantics) noexcept;

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    bool isOutputBroadcastAcrossBatch(
        int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept override;
    bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;
    void configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
        DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
        bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept override;
    IPluginV2Ext* clone() const noexcept override;

private:
    NMSParameters param{};
    int32_t mBoxesSize{};
    int32_t mScoresSize{};
    int32_t mNumPriors{};
    std::string mNamespace;
    bool mClipBoxes{};
    DataType mPrecision;
    int32_t mScoreBits;
    bool mCaffeSemantics{true};
    pluginStatus_t mPluginStatus{};
};

class TRT_DEPRECATED BatchedNMSDynamicPlugin : public IPluginV2DynamicExt
{
public:
    BatchedNMSDynamicPlugin(NMSParameters param);
    BatchedNMSDynamicPlugin(void const* data, size_t length);
    ~BatchedNMSDynamicPlugin() override = default;

    // IPluginV2 methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* libNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setClipParam(bool clip) noexcept;
    void setScoreBits(int32_t scoreBits) noexcept;
    void setCaffeSemantics(bool caffeSemantics) noexcept;

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputType, int32_t nbInputs) const noexcept override;

    // IPluginV2DynamicExt methods
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

private:
    NMSParameters param{};
    int32_t mBoxesSize{};
    int32_t mScoresSize{};
    int32_t mNumPriors{};
    std::string mNamespace;
    bool mClipBoxes{};
    DataType mPrecision;
    int32_t mScoreBits;
    bool mCaffeSemantics{true};
    pluginStatus_t mPluginStatus{};
};

class TRT_DEPRECATED BatchedNMSBasePluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    BatchedNMSBasePluginCreator();
    ~BatchedNMSBasePluginCreator() override = default;

    char const* getPluginVersion() const noexcept override;
    PluginFieldCollection const* getFieldNames() noexcept override;

protected:
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
};

class TRT_DEPRECATED BatchedNMSPluginCreator : public BatchedNMSBasePluginCreator
{
public:
    char const* getPluginName() const noexcept override;
    IPluginV2Ext* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;
    IPluginV2Ext* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;
};

class TRT_DEPRECATED BatchedNMSDynamicPluginCreator : public BatchedNMSBasePluginCreator
{
public:
    char const* getPluginName() const noexcept override;
    IPluginV2DynamicExt* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;
    IPluginV2DynamicExt* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_BATCHED_NMS_PLUGIN_H
