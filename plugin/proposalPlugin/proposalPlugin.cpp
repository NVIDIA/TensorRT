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

#include "proposalPlugin.h"
#include "NvInfer.h"
#include <cmath>
#include <cstring>
#include <stdio.h>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::ProposalPlugin;
using nvinfer1::plugin::ProposalDynamicPlugin;
using nvinfer1::plugin::ProposalBasePluginCreator;
using nvinfer1::plugin::ProposalPluginCreator;
using nvinfer1::plugin::ProposalDynamicPluginCreator;

// plugin specific constants
namespace
{
static const char* PROPOSAL_PLUGIN_VERSION{"1"};
static const char* PROPOSAL_PLUGIN_NAMES[] = {"Proposal", "ProposalDynamic"};
static const float RPN_STD_SCALING{1.0f};
} // namespace

// Static class fields initialization
PluginFieldCollection ProposalBasePluginCreator::mFC{};
std::vector<PluginField> ProposalBasePluginCreator::mPluginAttributes;

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

ProposalPlugin::ProposalPlugin(int input_height, int input_width, int rpn_height, int rpn_width, float rpn_std_scaling,
    int rpn_stride, float bbox_min_size, float nms_iou_threshold, int pre_nms_top_n, int max_box_num,
    const float* anchor_sizes, int anc_size_num, const float* anchor_ratios, int anc_ratio_num)
    : mInputHeight(input_height)
    , mInputWidth(input_width)
    , mRpnHeight(rpn_height)
    , mRpnWidth(rpn_width)
    , mRpnStdScaling(rpn_std_scaling)
    , mRpnStride(rpn_stride)
    , mBboxMinSize(bbox_min_size)
    , mNmsIouThreshold(nms_iou_threshold)
    , mPreNmsTopN(pre_nms_top_n)
    , mMaxBoxNum(max_box_num)
    , mAnchorSizeNum(anc_size_num)
    , mAnchorRatioNum(anc_ratio_num)
{
    for (int i = 0; i < anc_size_num; ++i)
    {
        mAnchorSizes.push_back(anchor_sizes[i]);
    }

    for (int i = 0; i < anc_ratio_num; ++i)
    {
        mAnchorRatios.push_back(anchor_ratios[i]);
    }
}

ProposalDynamicPlugin::ProposalDynamicPlugin(int input_height, int input_width, int rpn_height, int rpn_width,
    float rpn_std_scaling, int rpn_stride, float bbox_min_size, float nms_iou_threshold, int pre_nms_top_n,
    int max_box_num, const float* anchor_sizes, int anc_size_num, const float* anchor_ratios, int anc_ratio_num)
    : mInputHeight(input_height)
    , mInputWidth(input_width)
    , mRpnHeight(rpn_height)
    , mRpnWidth(rpn_width)
    , mRpnStdScaling(rpn_std_scaling)
    , mRpnStride(rpn_stride)
    , mBboxMinSize(bbox_min_size)
    , mNmsIouThreshold(nms_iou_threshold)
    , mPreNmsTopN(pre_nms_top_n)
    , mMaxBoxNum(max_box_num)
    , mAnchorSizeNum(anc_size_num)
    , mAnchorRatioNum(anc_ratio_num)
{
    for (int i = 0; i < anc_size_num; ++i)
    {
        mAnchorSizes.push_back(anchor_sizes[i]);
    }

    for (int i = 0; i < anc_ratio_num; ++i)
    {
        mAnchorRatios.push_back(anchor_ratios[i]);
    }
}

ProposalPlugin::ProposalPlugin(int input_height, int input_width, float rpn_std_scaling, int rpn_stride,
    float bbox_min_size, float nms_iou_threshold, int pre_nms_top_n, int max_box_num, const float* anchor_sizes,
    int anc_size_num, const float* anchor_ratios, int anc_ratio_num)
    : mInputHeight(input_height)
    , mInputWidth(input_width)
    , mRpnStdScaling(rpn_std_scaling)
    , mRpnStride(rpn_stride)
    , mBboxMinSize(bbox_min_size)
    , mNmsIouThreshold(nms_iou_threshold)
    , mPreNmsTopN(pre_nms_top_n)
    , mMaxBoxNum(max_box_num)
    , mAnchorSizeNum(anc_size_num)
    , mAnchorRatioNum(anc_ratio_num)
{
    for (int i = 0; i < anc_size_num; ++i)
    {
        mAnchorSizes.push_back(anchor_sizes[i]);
    }

    for (int i = 0; i < anc_ratio_num; ++i)
    {
        mAnchorRatios.push_back(anchor_ratios[i]);
    }
}

ProposalDynamicPlugin::ProposalDynamicPlugin(int input_height, int input_width, float rpn_std_scaling, int rpn_stride,
    float bbox_min_size, float nms_iou_threshold, int pre_nms_top_n, int max_box_num, const float* anchor_sizes,
    int anc_size_num, const float* anchor_ratios, int anc_ratio_num)
    : mInputHeight(input_height)
    , mInputWidth(input_width)
    , mRpnStdScaling(rpn_std_scaling)
    , mRpnStride(rpn_stride)
    , mBboxMinSize(bbox_min_size)
    , mNmsIouThreshold(nms_iou_threshold)
    , mPreNmsTopN(pre_nms_top_n)
    , mMaxBoxNum(max_box_num)
    , mAnchorSizeNum(anc_size_num)
    , mAnchorRatioNum(anc_ratio_num)
{
    for (int i = 0; i < anc_size_num; ++i)
    {
        mAnchorSizes.push_back(anchor_sizes[i]);
    }

    for (int i = 0; i < anc_ratio_num; ++i)
    {
        mAnchorRatios.push_back(anchor_ratios[i]);
    }
}

ProposalPlugin::ProposalPlugin(const void* serial_buf, size_t serial_size)
{
    const char* d = reinterpret_cast<const char*>(serial_buf);
    const char* a = d;
    mInputHeight = readFromBuffer<size_t>(a);
    mInputWidth = readFromBuffer<size_t>(a);
    mRpnHeight = readFromBuffer<size_t>(a);
    mRpnWidth = readFromBuffer<size_t>(a);
    mRpnStride = readFromBuffer<size_t>(a);
    mPreNmsTopN = readFromBuffer<size_t>(a);
    mMaxBoxNum = readFromBuffer<size_t>(a);
    mAnchorSizeNum = readFromBuffer<size_t>(a);
    mAnchorRatioNum = readFromBuffer<size_t>(a);
    mRpnStdScaling = readFromBuffer<float>(a);
    mBboxMinSize = readFromBuffer<float>(a);
    mNmsIouThreshold = readFromBuffer<float>(a);

    for (size_t i = 0; i < mAnchorSizeNum; ++i)
    {
        mAnchorSizes.push_back(readFromBuffer<float>(a));
    }

    for (size_t i = 0; i < mAnchorRatioNum; ++i)
    {
        mAnchorRatios.push_back(readFromBuffer<float>(a));
    }

    PLUGIN_VALIDATE(a == d + serial_size);
}

ProposalDynamicPlugin::ProposalDynamicPlugin(const void* serial_buf, size_t serial_size)
{
    const char* d = reinterpret_cast<const char*>(serial_buf);
    const char* a = d;
    mInputHeight = readFromBuffer<size_t>(a);
    mInputWidth = readFromBuffer<size_t>(a);
    mRpnHeight = readFromBuffer<size_t>(a);
    mRpnWidth = readFromBuffer<size_t>(a);
    mRpnStride = readFromBuffer<size_t>(a);
    mPreNmsTopN = readFromBuffer<size_t>(a);
    mMaxBoxNum = readFromBuffer<size_t>(a);
    mAnchorSizeNum = readFromBuffer<size_t>(a);
    mAnchorRatioNum = readFromBuffer<size_t>(a);
    mRpnStdScaling = readFromBuffer<float>(a);
    mBboxMinSize = readFromBuffer<float>(a);
    mNmsIouThreshold = readFromBuffer<float>(a);

    for (size_t i = 0; i < mAnchorSizeNum; ++i)
    {
        mAnchorSizes.push_back(readFromBuffer<float>(a));
    }

    for (size_t i = 0; i < mAnchorRatioNum; ++i)
    {
        mAnchorRatios.push_back(readFromBuffer<float>(a));
    }

    PLUGIN_VALIDATE(a == d + serial_size);
}

ProposalPlugin::~ProposalPlugin() noexcept {}

ProposalDynamicPlugin::~ProposalDynamicPlugin() noexcept {}

const char* ProposalPlugin::getPluginType() const noexcept
{
    return PROPOSAL_PLUGIN_NAMES[0];
}

const char* ProposalDynamicPlugin::getPluginType() const noexcept
{
    return PROPOSAL_PLUGIN_NAMES[1];
}

const char* ProposalPlugin::getPluginVersion() const noexcept
{
    return PROPOSAL_PLUGIN_VERSION;
}

const char* ProposalDynamicPlugin::getPluginVersion() const noexcept
{
    return PROPOSAL_PLUGIN_VERSION;
}

int ProposalPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int ProposalDynamicPlugin::getNbOutputs() const noexcept
{
    return 1;
}

Dims ProposalPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    // Validate input arguments
    PLUGIN_ASSERT(index == 0);
    PLUGIN_ASSERT(nbInputDims == 2);
    PLUGIN_ASSERT(inputs->nbDims == 3);
    PLUGIN_ASSERT((inputs + 1)->nbDims == 3);
    int channels = mMaxBoxNum;
    int height = 4;
    int width = 1;
    return Dims3(channels, height, width);
}

DimsExprs ProposalDynamicPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    // Validate input arguments
    PLUGIN_ASSERT(outputIndex == 0);
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(inputs[0].nbDims == 4);
    PLUGIN_ASSERT(inputs[1].nbDims == 4);
    DimsExprs out_dim;
    out_dim.nbDims = 4;
    out_dim.d[0] = inputs[0].d[0];
    out_dim.d[1] = exprBuilder.constant(mMaxBoxNum);
    out_dim.d[2] = exprBuilder.constant(4);
    out_dim.d[3] = exprBuilder.constant(1);
    return out_dim;
}

int ProposalPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

int ProposalDynamicPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

size_t ProposalPlugin::getWorkspaceSize(int max_batch_size) const noexcept
{
    return _get_workspace_size(max_batch_size, mAnchorSizeNum, mAnchorRatioNum, mRpnHeight, mRpnWidth, mMaxBoxNum);
}

size_t ProposalDynamicPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    int batch_size = inputs[0].dims.d[0];
    return _get_workspace_size(batch_size, mAnchorSizeNum, mAnchorRatioNum, mRpnHeight, mRpnWidth, mMaxBoxNum);
}

int ProposalPlugin::enqueue(
    int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int status = -1;
    // Our plugin outputs only one tensor
    void* output = outputs[0];
    status = proposalInference_gpu(stream, inputs[0], inputs[1], batchSize, mInputHeight, mInputWidth, mRpnHeight,
        mRpnWidth, mMaxBoxNum, mPreNmsTopN, &mAnchorSizes[0], mAnchorSizeNum, &mAnchorRatios[0], mAnchorRatioNum,
        mRpnStdScaling, mRpnStride, mBboxMinSize, mNmsIouThreshold, workspace, output);
    PLUGIN_ASSERT(status == STATUS_SUCCESS);
    return status;
}

int ProposalDynamicPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int status = -1;
    // Our plugin outputs only one tensor
    void* output = outputs[0];
    int batchSize = inputDesc[0].dims.d[0];
    status = proposalInference_gpu(stream, inputs[0], inputs[1], batchSize, mInputHeight, mInputWidth, mRpnHeight,
        mRpnWidth, mMaxBoxNum, mPreNmsTopN, &mAnchorSizes[0], mAnchorSizeNum, &mAnchorRatios[0], mAnchorRatioNum,
        mRpnStdScaling, mRpnStride, mBboxMinSize, mNmsIouThreshold, workspace, output);
    PLUGIN_ASSERT(status == STATUS_SUCCESS);
    return status;
}

size_t ProposalPlugin::getSerializationSize() const noexcept
{
    return sizeof(size_t) * 9 + sizeof(float) * 3 + sizeof(float) * mAnchorSizeNum + sizeof(float) * mAnchorRatioNum;
}

size_t ProposalDynamicPlugin::getSerializationSize() const noexcept
{
    return sizeof(size_t) * 9 + sizeof(float) * 3 + sizeof(float) * mAnchorSizeNum + sizeof(float) * mAnchorRatioNum;
}

void ProposalPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    char* a = d;
    writeToBuffer<size_t>(a, mInputHeight);
    writeToBuffer<size_t>(a, mInputWidth);
    writeToBuffer<size_t>(a, mRpnHeight);
    writeToBuffer<size_t>(a, mRpnWidth);
    writeToBuffer<size_t>(a, mRpnStride);
    writeToBuffer<size_t>(a, mPreNmsTopN);
    writeToBuffer<size_t>(a, mMaxBoxNum);
    writeToBuffer<size_t>(a, mAnchorSizeNum);
    writeToBuffer<size_t>(a, mAnchorRatioNum);
    writeToBuffer<float>(a, mRpnStdScaling);
    writeToBuffer<float>(a, mBboxMinSize);
    writeToBuffer<float>(a, mNmsIouThreshold);

    for (size_t i = 0; i < mAnchorSizeNum; ++i)
    {
        writeToBuffer<float>(a, mAnchorSizes[i]);
    }

    for (size_t i = 0; i < mAnchorRatioNum; ++i)
    {
        writeToBuffer<float>(a, mAnchorRatios[i]);
    }

    PLUGIN_ASSERT(a == d + getSerializationSize());
}

void ProposalDynamicPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    char* a = d;
    writeToBuffer<size_t>(a, mInputHeight);
    writeToBuffer<size_t>(a, mInputWidth);
    writeToBuffer<size_t>(a, mRpnHeight);
    writeToBuffer<size_t>(a, mRpnWidth);
    writeToBuffer<size_t>(a, mRpnStride);
    writeToBuffer<size_t>(a, mPreNmsTopN);
    writeToBuffer<size_t>(a, mMaxBoxNum);
    writeToBuffer<size_t>(a, mAnchorSizeNum);
    writeToBuffer<size_t>(a, mAnchorRatioNum);
    writeToBuffer<float>(a, mRpnStdScaling);
    writeToBuffer<float>(a, mBboxMinSize);
    writeToBuffer<float>(a, mNmsIouThreshold);

    for (size_t i = 0; i < mAnchorSizeNum; ++i)
    {
        writeToBuffer<float>(a, mAnchorSizes[i]);
    }

    for (size_t i = 0; i < mAnchorRatioNum; ++i)
    {
        writeToBuffer<float>(a, mAnchorRatios[i]);
    }

    PLUGIN_ASSERT(a == d + getSerializationSize());
}

bool ProposalPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    // This plugin only supports ordinary floats, and NCHW input format
    if (type == DataType::kFLOAT && format == PluginFormat::kLINEAR)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool ProposalDynamicPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // 2 inputs, 1 outputs, so 3 input/output in total
    PLUGIN_ASSERT(0 <= pos && pos < 3);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    const bool consistentFloatPrecision = (in[0].type == in[pos].type);
    switch (pos)
    {
    case 0: return in[0].type == DataType::kFLOAT && in[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 1: return in[1].type == DataType::kFLOAT && in[1].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 2:
        return out[0].type == DataType::kFLOAT && out[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    }
    return false;
}

void ProposalPlugin::terminate() noexcept {}

void ProposalDynamicPlugin::terminate() noexcept {}

void ProposalPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void ProposalDynamicPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2Ext* ProposalPlugin::clone() const noexcept
{
    try
    {
        IPluginV2Ext* plugin = new ProposalPlugin(mInputHeight, mInputWidth, mRpnHeight, mRpnWidth, mRpnStdScaling,
            mRpnStride, mBboxMinSize, mNmsIouThreshold, mPreNmsTopN, mMaxBoxNum, &mAnchorSizes[0], mAnchorSizeNum,
            &mAnchorRatios[0], mAnchorRatioNum);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* ProposalDynamicPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new ProposalDynamicPlugin(mInputHeight, mInputWidth, mRpnHeight, mRpnWidth, mRpnStdScaling,
            mRpnStride, mBboxMinSize, mNmsIouThreshold, mPreNmsTopN, mMaxBoxNum, &mAnchorSizes[0], mAnchorSizeNum,
            &mAnchorRatios[0], mAnchorRatioNum);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void ProposalPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

void ProposalDynamicPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* ProposalPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

const char* ProposalDynamicPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index.
DataType ProposalPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // one outputs
    PLUGIN_ASSERT(index == 0);
    return DataType::kFLOAT;
}

DataType ProposalDynamicPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    noexcept
{
    // one outputs
    PLUGIN_ASSERT(index == 0);
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool ProposalPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool ProposalPlugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

void ProposalPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(nbOutputs == 1);
    mRpnHeight = inputDims->d[1];
    mRpnWidth = inputDims->d[2];
}

void ProposalDynamicPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(nbOutputs == 1);
    mRpnHeight = in[0].desc.dims.d[2];
    mRpnWidth = in[0].desc.dims.d[3];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void ProposalPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void ProposalPlugin::detachFromContext() noexcept {}

ProposalBasePluginCreator::ProposalBasePluginCreator() noexcept
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("input_height", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("input_width", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("rpn_stride", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("roi_min_size", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("nms_iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("pre_nms_top_n", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("post_nms_top_n", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("anchor_sizes", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("anchor_ratios", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

ProposalPluginCreator::ProposalPluginCreator() noexcept
{
    mPluginName = PROPOSAL_PLUGIN_NAMES[0];
}

ProposalDynamicPluginCreator::ProposalDynamicPluginCreator() noexcept
{
    mPluginName = PROPOSAL_PLUGIN_NAMES[1];
}

const char* ProposalBasePluginCreator::getPluginName() const noexcept
{
    return mPluginName.c_str();
}

const char* ProposalBasePluginCreator::getPluginVersion() const noexcept
{
    return PROPOSAL_PLUGIN_VERSION;
}

const PluginFieldCollection* ProposalBasePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* ProposalPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        int nbFields = fc->nbFields;
        int input_height = 0, input_width = 0, rpn_stride = 0, pre_nms_top_n = 0, post_nms_top_n = 0;
        float roi_min_size = 0.0f, nms_iou_threshold = 0.0f;
        std::vector<float> anchor_sizes;
        std::vector<float> anchor_ratios;

        for (int i = 0; i < nbFields; ++i)
        {
            const char* attr_name = fields[i].name;

            if (!strcmp(attr_name, "input_height"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                input_height = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attr_name, "input_width"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                input_width = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attr_name, "rpn_stride"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                rpn_stride = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attr_name, "roi_min_size"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                roi_min_size = *(static_cast<const float*>(fields[i].data));
            }
            else if (!strcmp(attr_name, "nms_iou_threshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                nms_iou_threshold = *(static_cast<const float*>(fields[i].data));
            }
            else if (!strcmp(attr_name, "pre_nms_top_n"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                pre_nms_top_n = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attr_name, "post_nms_top_n"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                post_nms_top_n = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attr_name, "anchor_sizes"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                const float* as = static_cast<const float*>(fields[i].data);

                for (int j = 0; j < fields[i].length; ++j)
                {
                    PLUGIN_VALIDATE(*as > 0.0f);
                    anchor_sizes.push_back(*as);
                    ++as;
                }
            }
            else if (!strcmp(attr_name, "anchor_ratios"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                const float* ar = static_cast<const float*>(fields[i].data);

                // take the square root.
                for (int j = 0; j < fields[i].length; ++j)
                {
                    PLUGIN_VALIDATE(*ar > 0.0f);
                    anchor_ratios.push_back(std::sqrt(*ar));
                    ++ar;
                }
            }
        }

        PLUGIN_VALIDATE(input_height > 0 && input_width > 0 && rpn_stride > 0 && pre_nms_top_n > 0 && post_nms_top_n
            && roi_min_size >= 0.0f && nms_iou_threshold > 0.0f);

        IPluginV2Ext* plugin = new ProposalPlugin(input_height, input_width, RPN_STD_SCALING, rpn_stride, roi_min_size,
            nms_iou_threshold, pre_nms_top_n, post_nms_top_n, &anchor_sizes[0], anchor_sizes.size(), &anchor_ratios[0],
            anchor_ratios.size());
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* ProposalDynamicPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        int nbFields = fc->nbFields;
        int input_height = 0, input_width = 0, rpn_stride = 0, pre_nms_top_n = 0, post_nms_top_n = 0;
        float roi_min_size = 0.0f, nms_iou_threshold = 0.0f;
        std::vector<float> anchor_sizes;
        std::vector<float> anchor_ratios;

        for (int i = 0; i < nbFields; ++i)
        {
            const char* attr_name = fields[i].name;

            if (!strcmp(attr_name, "input_height"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                input_height = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attr_name, "input_width"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                input_width = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attr_name, "rpn_stride"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                rpn_stride = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attr_name, "roi_min_size"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                roi_min_size = *(static_cast<const float*>(fields[i].data));
            }
            else if (!strcmp(attr_name, "nms_iou_threshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                nms_iou_threshold = *(static_cast<const float*>(fields[i].data));
            }
            else if (!strcmp(attr_name, "pre_nms_top_n"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                pre_nms_top_n = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attr_name, "post_nms_top_n"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                post_nms_top_n = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attr_name, "anchor_sizes"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                const float* as = static_cast<const float*>(fields[i].data);

                for (int j = 0; j < fields[i].length; ++j)
                {
                    PLUGIN_VALIDATE(*as > 0.0f);
                    anchor_sizes.push_back(*as);
                    ++as;
                }
            }
            else if (!strcmp(attr_name, "anchor_ratios"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                const float* ar = static_cast<const float*>(fields[i].data);

                // take the square root.
                for (int j = 0; j < fields[i].length; ++j)
                {
                    PLUGIN_VALIDATE(*ar > 0.0f);
                    anchor_ratios.push_back(std::sqrt(*ar));
                    ++ar;
                }
            }
        }

        PLUGIN_VALIDATE(input_height > 0 && input_width > 0 && rpn_stride > 0 && pre_nms_top_n > 0 && post_nms_top_n
            && roi_min_size >= 0.0f && nms_iou_threshold > 0.0f);

        IPluginV2DynamicExt* plugin = new ProposalDynamicPlugin(input_height, input_width, RPN_STD_SCALING, rpn_stride,
            roi_min_size, nms_iou_threshold, pre_nms_top_n, post_nms_top_n, &anchor_sizes[0], anchor_sizes.size(),
            &anchor_ratios[0], anchor_ratios.size());
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* ProposalPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed,
        IPluginV2Ext* plugin = new ProposalPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* ProposalDynamicPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed,
        IPluginV2DynamicExt* plugin = new ProposalDynamicPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
