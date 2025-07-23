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

#ifndef TRT_PROPOSAL_PLUGIN_H
#define TRT_PROPOSAL_PLUGIN_H

#include "NvInferPlugin.h"
#include "common/kernels/kernel.h"
#include "common/plugin.h"
#include <string>
#include <vector>

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2Ext and BaseCreator classes.
// For requirements for overriden functions, check TensorRT API docs.
namespace nvinfer1
{
namespace plugin
{

class TRT_DEPRECATED ProposalPlugin : public IPluginV2Ext
{
public:
    ProposalPlugin(int32_t inputHeight, int32_t inputWidth, int32_t rpnHeight, int32_t rpnWidth, float rpnStdScaling,
        int32_t rpnStride, float bboxMinSize, float nmsIouThreshold, int32_t preNmsTopN, int32_t maxBoxNum,
        float const* anchorSizes, int32_t ancSizeNum, float const* anchorRatios, int32_t ancRatioNum);

    ProposalPlugin(int32_t inputHeight, int32_t inputWidth, float rpnStdScaling, int32_t rpnStride, float bboxMinSize,
        float nmsIouThreshold, int32_t preNmsTopN, int32_t maxBoxNum, float const* anchorSizes, int32_t ancSizeNum,
        float const* anchorRatios, int32_t ancRatioNum);

    ProposalPlugin(void const* serialBuf, size_t serialSize);

    // It doesn't make sense to make ProposalPlugin without arguments, so we
    // delete default constructor.
    ProposalPlugin() noexcept = delete;

    ~ProposalPlugin() noexcept override;

    int32_t getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int32_t) const noexcept override;

    int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::IPluginV2Ext* clone() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    bool isOutputBroadcastAcrossBatch(
        int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

    void configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
        DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
        bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept override;

    void detachFromContext() noexcept override;

private:
    std::string mNamespace;
    size_t mInputHeight;
    size_t mInputWidth;
    size_t mRpnHeight;
    size_t mRpnWidth;
    float mRpnStdScaling;
    size_t mRpnStride;
    float mBboxMinSize;
    float mNmsIouThreshold;
    size_t mPreNmsTopN;
    size_t mMaxBoxNum;
    std::vector<float> mAnchorSizes;
    size_t mAnchorSizeNum;
    std::vector<float> mAnchorRatios;
    size_t mAnchorRatioNum;
};

class ProposalDynamicPlugin : public IPluginV2DynamicExt
{
public:
    ProposalDynamicPlugin(int32_t inputHeight, int32_t inputWidth, int32_t rpnHeight, int32_t rpnWidth, float rpnStdScaling,
        int32_t rpnStride, float bboxMinSize, float nmsIouThreshold, int32_t preNmsTopN, int32_t maxBoxNum,
        float const* anchorSizes, int32_t ancSizeNum, float const* anchorRatios, int32_t ancRatioNum);

    ProposalDynamicPlugin(int32_t inputHeight, int32_t inputWidth, float rpnStdScaling, int32_t rpnStride, float bboxMinSize,
        float nmsIouThreshold, int32_t preNmsTopN, int32_t maxBoxNum, float const* anchorSizes, int32_t ancSizeNum,
        float const* anchorRatios, int32_t ancRatioNum);

    ProposalDynamicPlugin(void const* serialBuf, size_t serialSize);

    // It doesn't make sense to make ProposalDynamicPlugin without arguments, so
    // we delete default constructor.
    ProposalDynamicPlugin() noexcept = delete;

    ~ProposalDynamicPlugin() noexcept override;

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

    // IPluginV2Ext methods
    DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

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
    std::string mNamespace;
    size_t mInputHeight;
    size_t mInputWidth;
    size_t mRpnHeight;
    size_t mRpnWidth;
    float mRpnStdScaling;
    size_t mRpnStride;
    float mBboxMinSize;
    float mNmsIouThreshold;
    size_t mPreNmsTopN;
    size_t mMaxBoxNum;
    std::vector<float> mAnchorSizes;
    size_t mAnchorSizeNum;
    std::vector<float> mAnchorRatios;
    size_t mAnchorRatioNum;
};

class ProposalBasePluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    ProposalBasePluginCreator() noexcept;
    ~ProposalBasePluginCreator() noexcept override = default;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    PluginFieldCollection const* getFieldNames() noexcept override;

protected:
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
    std::string mPluginName;
};

class TRT_DEPRECATED ProposalPluginCreator : public ProposalBasePluginCreator
{
public:
    ProposalPluginCreator() noexcept;
    ~ProposalPluginCreator() noexcept override = default;
    IPluginV2Ext* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;
    IPluginV2Ext* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;
};

class ProposalDynamicPluginCreator : public ProposalBasePluginCreator
{
public:
    ProposalDynamicPluginCreator() noexcept;
    ~ProposalDynamicPluginCreator() noexcept override = default;
    IPluginV2DynamicExt* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;
    IPluginV2DynamicExt* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;
};

} // namespace plugin

} // namespace nvinfer1

#endif // TRT_PROPOSAL_PLUGIN_H
