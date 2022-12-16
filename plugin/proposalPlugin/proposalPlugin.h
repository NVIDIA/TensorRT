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

#ifndef PROPOSAL_PLUGIN_H
#define PROPOSAL_PLUGIN_H

#include "NvInferPlugin.h"
#include "common/kernel.h"
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

class ProposalPlugin : public IPluginV2Ext
{
public:
    ProposalPlugin(int input_height, int input_width, int rpn_height, int rpn_width, float rpn_std_scaling,
        int rpn_stride, float bbox_min_size, float nms_iou_threshold, int pre_nms_top_n, int max_box_num,
        const float* anchor_sizes, int anc_size_num, const float* anchor_ratios, int anc_ratio_num);

    ProposalPlugin(int input_height, int input_width, float rpn_std_scaling, int rpn_stride, float bbox_min_size,
        float nms_iou_threshold, int pre_nms_top_n, int max_box_num, const float* anchor_sizes, int anc_size_num,
        const float* anchor_ratios, int anc_ratio_num);

    ProposalPlugin(const void* serial_buf, size_t serial_size);

    // It doesn't make sense to make ProposalPlugin without arguments, so we delete default constructor.
    ProposalPlugin() noexcept = delete;

    ~ProposalPlugin() noexcept override;

    int getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int) const noexcept override;

    int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::IPluginV2Ext* clone() const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
        noexcept override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override;

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
    ProposalDynamicPlugin(int input_height, int input_width, int rpn_height, int rpn_width, float rpn_std_scaling,
        int rpn_stride, float bbox_min_size, float nms_iou_threshold, int pre_nms_top_n, int max_box_num,
        const float* anchor_sizes, int anc_size_num, const float* anchor_ratios, int anc_ratio_num);

    ProposalDynamicPlugin(int input_height, int input_width, float rpn_std_scaling, int rpn_stride, float bbox_min_size,
        float nms_iou_threshold, int pre_nms_top_n, int max_box_num, const float* anchor_sizes, int anc_size_num,
        const float* anchor_ratios, int anc_ratio_num);

    ProposalDynamicPlugin(const void* serial_buf, size_t serial_size);

    // It doesn't make sense to make ProposalDynamicPlugin without arguments, so we delete default constructor.
    ProposalDynamicPlugin() noexcept = delete;

    ~ProposalDynamicPlugin() noexcept override;

    // IPluginV2 methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* libNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

    // IPluginV2Ext methods
    DataType getOutputDataType(int index, const nvinfer1::DataType* inputType, int nbInputs) const noexcept override;

    // IPluginV2DynamicExt methods
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(
        int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out,
        int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs,
        int nbOutputs) const noexcept override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
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
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;

protected:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mPluginName;
};

class ProposalPluginCreator : public ProposalBasePluginCreator
{
public:
    ProposalPluginCreator() noexcept;
    ~ProposalPluginCreator() noexcept override = default;
    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
};

class ProposalDynamicPluginCreator : public ProposalBasePluginCreator
{
public:
    ProposalDynamicPluginCreator() noexcept;
    ~ProposalDynamicPluginCreator() noexcept override = default;
    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2DynamicExt* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;
};

} // namespace plugin

} // namespace nvinfer1

#endif // PROPOSAL_PLUGIN_H
