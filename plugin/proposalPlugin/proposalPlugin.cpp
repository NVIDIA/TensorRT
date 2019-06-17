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

#include "proposalPlugin.h"
#include "NvInfer.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdio.h>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::ProposalPlugin;
using nvinfer1::plugin::ProposalPluginCreator;

// plugin specific constants
namespace
{
static const char* PROPOSAL_PLUGIN_VERSION{"1"};
static const char* PROPOSAL_PLUGIN_NAME{"Proposal"};
static const float RPN_STD_SCALING{1.0f};
}

// Static class fields initialization
PluginFieldCollection ProposalPluginCreator::mFC{};
std::vector<PluginField> ProposalPluginCreator::mPluginAttributes;

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

ProposalPlugin::ProposalPlugin(const std::string name)
    : mLayerName(name)
{
}

ProposalPlugin::ProposalPlugin(const std::string name, int input_height, int input_width, int rpn_height, int rpn_width,
    float rpn_std_scaling, int rpn_stride, float bbox_min_size, float nms_iou_threshold, int pre_nms_top_n,
    int max_box_num, const float* anchor_sizes, int anc_size_num, const float* anchor_ratios, int anc_ratio_num)
    : mLayerName(name)
    , mInputHeight(input_height)
    , mInputWidth(input_width)
    , mRpnHeight(rpn_height)
    , mRpnWidth(rpn_width)
    , mRpnStdScaling(rpn_std_scaling)
    , mRpnStride(rpn_stride)
    , mBboxMinSize(bbox_min_size)
    , mNmsIouThreshold(nms_iou_threshold)
    , mPreNmsTopN(pre_nms_top_n)
    , mAnchorSizeNum(anc_size_num)
    , mAnchorRatioNum(anc_ratio_num)
    , mMaxBoxNum(max_box_num)
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

ProposalPlugin::ProposalPlugin(const std::string name, int input_height, int input_width, float rpn_std_scaling,
    int rpn_stride, float bbox_min_size, float nms_iou_threshold, int pre_nms_top_n, int max_box_num,
    const float* anchor_sizes, int anc_size_num, const float* anchor_ratios, int anc_ratio_num)
    : mLayerName(name)
    , mInputHeight(input_height)
    , mInputWidth(input_width)
    , mRpnStdScaling(rpn_std_scaling)
    , mRpnStride(rpn_stride)
    , mBboxMinSize(bbox_min_size)
    , mNmsIouThreshold(nms_iou_threshold)
    , mPreNmsTopN(pre_nms_top_n)
    , mAnchorSizeNum(anc_size_num)
    , mAnchorRatioNum(anc_ratio_num)
    , mMaxBoxNum(max_box_num)
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

ProposalPlugin::ProposalPlugin(const std::string name, const void* serial_buf, size_t serial_size)
    : mLayerName(name)
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

    for (int i = 0; i < mAnchorSizeNum; ++i)
    {
        mAnchorSizes.push_back(readFromBuffer<float>(a));
    }

    for (int i = 0; i < mAnchorRatioNum; ++i)
    {
        mAnchorRatios.push_back(readFromBuffer<float>(a));
    }

    ASSERT(a == d + serial_size);
}

ProposalPlugin::~ProposalPlugin()
{
}

const char* ProposalPlugin::getPluginType() const
{
    return PROPOSAL_PLUGIN_NAME;
}

const char* ProposalPlugin::getPluginVersion() const
{
    return PROPOSAL_PLUGIN_VERSION;
}

int ProposalPlugin::getNbOutputs() const
{
    return 1;
}

Dims ProposalPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Validate input arguments
    ASSERT(index == 0);
    ASSERT(nbInputDims == 2);
    ASSERT(inputs->nbDims == 3);
    ASSERT((inputs + 1)->nbDims == 3);
    int channels = mMaxBoxNum;
    int height = 4;
    int width = 1;
    return DimsCHW(channels, height, width);
}

int ProposalPlugin::initialize()
{
    return 0;
}

size_t ProposalPlugin::getWorkspaceSize(int max_batch_size) const
{
    return _get_workspace_size(max_batch_size, mAnchorSizeNum, mAnchorRatioNum, mRpnHeight, mRpnWidth, mMaxBoxNum);
}

int ProposalPlugin::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    int status = -1;
    // Our plugin outputs only one tensor
    void* output = outputs[0];
    status = proposalInference_gpu(stream, inputs[0], inputs[1], batchSize, mInputHeight, mInputWidth, mRpnHeight,
        mRpnWidth, mMaxBoxNum, mPreNmsTopN, &mAnchorSizes[0], mAnchorSizeNum, &mAnchorRatios[0], mAnchorRatioNum,
        mRpnStdScaling, mRpnStride, mBboxMinSize, mNmsIouThreshold, workspace, output);
    return status;
}

size_t ProposalPlugin::getSerializationSize() const
{
    return sizeof(size_t) * 9 + sizeof(float) * 3 + sizeof(float) * mAnchorSizeNum + sizeof(float) * mAnchorRatioNum;
}

void ProposalPlugin::serialize(void* buffer) const
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

    for (int i = 0; i < mAnchorSizeNum; ++i)
    {
        writeToBuffer<float>(a, mAnchorSizes[i]);
    }

    for (int i = 0; i < mAnchorRatioNum; ++i)
    {
        writeToBuffer<float>(a, mAnchorRatios[i]);
    }

    ASSERT(a == d + getSerializationSize());
}

bool ProposalPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    // This plugin only supports ordinary floats, and NCHW input format
    if (type == DataType::kFLOAT && format == PluginFormat::kNCHW)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void ProposalPlugin::terminate()
{
}

void ProposalPlugin::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2Ext* ProposalPlugin::clone() const
{
    IPluginV2Ext* plugin = new ProposalPlugin(mLayerName, mInputHeight, mInputWidth, mRpnHeight, mRpnWidth,
        mRpnStdScaling, mRpnStride, mBboxMinSize, mNmsIouThreshold, mPreNmsTopN, mMaxBoxNum, &mAnchorSizes[0],
        mAnchorSizeNum, &mAnchorRatios[0], mAnchorRatioNum);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void ProposalPlugin::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* ProposalPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index.
DataType ProposalPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // one outputs
    ASSERT(index == 0);
    return DataType::kFLOAT;
}
// Return true if output tensor is broadcast across a batch.
bool ProposalPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool ProposalPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

void ProposalPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    ASSERT(
        inputTypes[0] == DataType::kFLOAT && inputTypes[1] == DataType::kFLOAT && floatFormat == PluginFormat::kNCHW);

    ASSERT(nbInputs == 2);
    ASSERT(nbOutputs == 1);

    mRpnHeight = inputDims->d[1];
    mRpnWidth = inputDims->d[2];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void ProposalPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void ProposalPlugin::detachFromContext()
{
}

ProposalPluginCreator::ProposalPluginCreator()
{
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

ProposalPluginCreator::~ProposalPluginCreator()
{
}

const char* ProposalPluginCreator::getPluginName() const
{
    return PROPOSAL_PLUGIN_NAME;
}

const char* ProposalPluginCreator::getPluginVersion() const
{
    return PROPOSAL_PLUGIN_VERSION;
}

const PluginFieldCollection* ProposalPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* ProposalPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int input_height, input_width, rpn_stride, pre_nms_top_n, post_nms_top_n;
    float roi_min_size, nms_iou_threshold;
    std::vector<float> anchor_sizes;
    std::vector<float> anchor_ratios;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;

        if (!strcmp(attr_name, "input_height"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            input_height = *(static_cast<const int*>(fields[i].data));
            ASSERT(input_height > 0);
        }
        else if (!strcmp(attr_name, "input_width"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            input_width = *(static_cast<const int*>(fields[i].data));
            ASSERT(input_width > 0);
        }
        else if (!strcmp(attr_name, "rpn_stride"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            rpn_stride = *(static_cast<const int*>(fields[i].data));
            ASSERT(rpn_stride > 0);
        }
        else if (!strcmp(attr_name, "roi_min_size"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            roi_min_size = *(static_cast<const float*>(fields[i].data));
            ASSERT(roi_min_size >= 0.0f);
        }
        else if (!strcmp(attr_name, "nms_iou_threshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            nms_iou_threshold = *(static_cast<const float*>(fields[i].data));
            ASSERT(nms_iou_threshold > 0.0f);
        }
        else if (!strcmp(attr_name, "pre_nms_top_n"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            pre_nms_top_n = *(static_cast<const int*>(fields[i].data));
            ASSERT(pre_nms_top_n > 0);
        }
        else if (!strcmp(attr_name, "post_nms_top_n"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            post_nms_top_n = *(static_cast<const int*>(fields[i].data));
            ASSERT(post_nms_top_n > 0);
        }
        else if (!strcmp(attr_name, "anchor_sizes"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            const float* as = static_cast<const float*>(fields[i].data);

            for (int j = 0; j < fields[i].length; ++j)
            {
                ASSERT(*as > 0.0f);
                anchor_sizes.push_back(*as);
                ++as;
            }
        }
        else if (!strcmp(attr_name, "anchor_ratios"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            const float* ar = static_cast<const float*>(fields[i].data);

            // take the square root.
            for (int j = 0; j < fields[i].length; ++j)
            {
                ASSERT(*ar > 0.0f);
                anchor_ratios.push_back(std::sqrt(*ar));
                ++ar;
            }
        }
    }

    IPluginV2Ext* plugin = new ProposalPlugin(name, input_height, input_width, RPN_STD_SCALING, rpn_stride,
        roi_min_size, nms_iou_threshold, pre_nms_top_n, post_nms_top_n, &anchor_sizes[0], anchor_sizes.size(),
        &anchor_ratios[0], anchor_ratios.size());
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* ProposalPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed,
    IPluginV2Ext* plugin = new ProposalPlugin(name, serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
