/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef PROPOSAL_PLUGIN_H
#define PROPOSAL_PLUGIN_H

#include "NvInferPlugin.h"
#include "kernel.h"
#include "plugin.h"
#include <string>
#include <vector>

using namespace nvinfer1::plugin;

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
    ProposalPlugin(const std::string name);

    ProposalPlugin(const std::string name, int input_height, int input_width, int rpn_height, int rpn_width,
        float rpn_std_scaling, int rpn_stride, float bbox_min_size, float nms_iou_threshold, int pre_nms_top_n,
        int max_box_num, const float* anchor_sizes, int anc_size_num, const float* anchor_ratios, int anc_ratio_num);

    ProposalPlugin(const std::string name, int input_height, int input_width, float rpn_std_scaling, int rpn_stride,
        float bbox_min_size, float nms_iou_threshold, int pre_nms_top_n, int max_box_num, const float* anchor_sizes,
        int anc_size_num, const float* anchor_ratios, int anc_ratio_num);

    ProposalPlugin(const std::string name, const void* serial_buf, size_t serial_size);

    // It doesn't make sense to make ProposalPlugin without arguments, so we delete default constructor.
    ProposalPlugin() = delete;

    ~ProposalPlugin() override;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int) const override;

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    nvinfer1::IPluginV2Ext* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

    void detachFromContext() override;

private:
    const std::string mLayerName;
    std::string mNamespace;
    size_t mInputHeight;
    size_t mInputWidth;
    size_t mRpnHeight;
    size_t mRpnWidth;
    float mRpnStdScaling;
    size_t mRpnStride;
    float mBboxMinSize;
    float mNmsIouThreshold;
    size_t mMaxBoxNum;
    size_t mPreNmsTopN;
    size_t mAnchorSizeNum, mAnchorRatioNum;
    std::vector<float> mAnchorSizes;
    std::vector<float> mAnchorRatios;
};

class ProposalPluginCreator : public BaseCreator
{
public:
    ProposalPluginCreator();

    ~ProposalPluginCreator() override;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin

} // namespace nvinfer1

#endif // PROPOSAL_PLUGIN_H
