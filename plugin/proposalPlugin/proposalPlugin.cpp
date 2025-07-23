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

#include "proposalPlugin.h"
#include "NvInfer.h"
#include "common/templates.h"
#include <cmath>
#include <cstring>
#include <stdio.h>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::ProposalBasePluginCreator;
using nvinfer1::plugin::ProposalDynamicPlugin;
using nvinfer1::plugin::ProposalDynamicPluginCreator;
using nvinfer1::plugin::ProposalPlugin;
using nvinfer1::plugin::ProposalPluginCreator;

// plugin specific constants
namespace
{
static char const* const kPROPOSAL_PLUGIN_VERSION{"1"};
static char const* const kPROPOSAL_PLUGIN_NAMES[] = {"Proposal", "ProposalDynamic"};
static constexpr float kRPN_STD_SCALING{1.0F};
} // namespace

ProposalPlugin::ProposalPlugin(int32_t inputHeight, int32_t inputWidth, int32_t rpnHeight, int32_t rpnWidth,
    float rpnStdScaling, int32_t rpnStride, float bboxMinSize, float nmsIouThreshold, int32_t preNmsTopN,
    int32_t maxBoxNum, float const* anchorSizes, int32_t ancSizeNum, float const* anchorRatios, int32_t ancRatioNum)
    : mInputHeight(inputHeight)
    , mInputWidth(inputWidth)
    , mRpnHeight(rpnHeight)
    , mRpnWidth(rpnWidth)
    , mRpnStdScaling(rpnStdScaling)
    , mRpnStride(rpnStride)
    , mBboxMinSize(bboxMinSize)
    , mNmsIouThreshold(nmsIouThreshold)
    , mPreNmsTopN(preNmsTopN)
    , mMaxBoxNum(maxBoxNum)
    , mAnchorSizeNum(ancSizeNum)
    , mAnchorRatioNum(ancRatioNum)
{
    for (int32_t i = 0; i < ancSizeNum; ++i)
    {
        mAnchorSizes.push_back(anchorSizes[i]);
    }

    for (int32_t i = 0; i < ancRatioNum; ++i)
    {
        mAnchorRatios.push_back(anchorRatios[i]);
    }
}

ProposalDynamicPlugin::ProposalDynamicPlugin(int32_t inputHeight, int32_t inputWidth, int32_t rpnHeight,
    int32_t rpnWidth, float rpnStdScaling, int32_t rpnStride, float bboxMinSize, float nmsIouThreshold,
    int32_t preNmsTopN, int32_t maxBoxNum, float const* anchorSizes, int32_t ancSizeNum, float const* anchorRatios,
    int32_t ancRatioNum)
    : mInputHeight(inputHeight)
    , mInputWidth(inputWidth)
    , mRpnHeight(rpnHeight)
    , mRpnWidth(rpnWidth)
    , mRpnStdScaling(rpnStdScaling)
    , mRpnStride(rpnStride)
    , mBboxMinSize(bboxMinSize)
    , mNmsIouThreshold(nmsIouThreshold)
    , mPreNmsTopN(preNmsTopN)
    , mMaxBoxNum(maxBoxNum)
    , mAnchorSizeNum(ancSizeNum)
    , mAnchorRatioNum(ancRatioNum)
{
    for (int32_t i = 0; i < ancSizeNum; ++i)
    {
        mAnchorSizes.push_back(anchorSizes[i]);
    }

    for (int32_t i = 0; i < ancRatioNum; ++i)
    {
        mAnchorRatios.push_back(anchorRatios[i]);
    }
}

ProposalPlugin::ProposalPlugin(int32_t inputHeight, int32_t inputWidth, float rpnStdScaling, int32_t rpnStride,
    float bboxMinSize, float nmsIouThreshold, int32_t preNmsTopN, int32_t maxBoxNum, float const* anchorSizes,
    int32_t ancSizeNum, float const* anchorRatios, int32_t ancRatioNum)
    : mInputHeight(inputHeight)
    , mInputWidth(inputWidth)
    , mRpnStdScaling(rpnStdScaling)
    , mRpnStride(rpnStride)
    , mBboxMinSize(bboxMinSize)
    , mNmsIouThreshold(nmsIouThreshold)
    , mPreNmsTopN(preNmsTopN)
    , mMaxBoxNum(maxBoxNum)
    , mAnchorSizeNum(ancSizeNum)
    , mAnchorRatioNum(ancRatioNum)
{
    for (int32_t i = 0; i < ancSizeNum; ++i)
    {
        mAnchorSizes.push_back(anchorSizes[i]);
    }

    for (int32_t i = 0; i < ancRatioNum; ++i)
    {
        mAnchorRatios.push_back(anchorRatios[i]);
    }
}

ProposalDynamicPlugin::ProposalDynamicPlugin(int32_t inputHeight, int32_t inputWidth, float rpnStdScaling,
    int32_t rpnStride, float bboxMinSize, float nmsIouThreshold, int32_t preNmsTopN, int32_t maxBoxNum,
    float const* anchorSizes, int32_t ancSizeNum, float const* anchorRatios, int32_t ancRatioNum)
    : mInputHeight(inputHeight)
    , mInputWidth(inputWidth)
    , mRpnStdScaling(rpnStdScaling)
    , mRpnStride(rpnStride)
    , mBboxMinSize(bboxMinSize)
    , mNmsIouThreshold(nmsIouThreshold)
    , mPreNmsTopN(preNmsTopN)
    , mMaxBoxNum(maxBoxNum)
    , mAnchorSizeNum(ancSizeNum)
    , mAnchorRatioNum(ancRatioNum)
{
    for (int32_t i = 0; i < ancSizeNum; ++i)
    {
        mAnchorSizes.push_back(anchorSizes[i]);
    }

    for (int32_t i = 0; i < ancRatioNum; ++i)
    {
        mAnchorRatios.push_back(anchorRatios[i]);
    }
}

ProposalPlugin::ProposalPlugin(void const* serialBuf, size_t serialSize)
{
    PLUGIN_ASSERT(serialBuf != nullptr);
    uint8_t const* d = reinterpret_cast<uint8_t const*>(serialBuf);
    uint8_t const* a = d;
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

    PLUGIN_ASSERT(a == d + serialSize);
}

ProposalDynamicPlugin::ProposalDynamicPlugin(void const* serialBuf, size_t serialSize)
{
    PLUGIN_ASSERT(serialBuf != nullptr);
    uint8_t const* d = reinterpret_cast<uint8_t const*>(serialBuf);
    uint8_t const* a = d;
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

    PLUGIN_ASSERT(a == d + serialSize);
}

ProposalPlugin::~ProposalPlugin() noexcept {}

ProposalDynamicPlugin::~ProposalDynamicPlugin() noexcept {}

char const* ProposalPlugin::getPluginType() const noexcept
{
    return kPROPOSAL_PLUGIN_NAMES[0];
}

char const* ProposalDynamicPlugin::getPluginType() const noexcept
{
    return kPROPOSAL_PLUGIN_NAMES[1];
}

char const* ProposalPlugin::getPluginVersion() const noexcept
{
    return kPROPOSAL_PLUGIN_VERSION;
}

char const* ProposalDynamicPlugin::getPluginVersion() const noexcept
{
    return kPROPOSAL_PLUGIN_VERSION;
}

int32_t ProposalPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t ProposalDynamicPlugin::getNbOutputs() const noexcept
{
    return 1;
}

Dims ProposalPlugin::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept
{
    try
    {
        // Validate input arguments
        PLUGIN_VALIDATE(index == 0);
        PLUGIN_VALIDATE(nbInputDims == 2);
        PLUGIN_VALIDATE(inputs->nbDims == 3);
        PLUGIN_VALIDATE(inputs[1].nbDims == 3);
        int32_t channels = mMaxBoxNum;
        int32_t height = 4;
        int32_t width = 1;
        return Dims3(channels, height, width);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return Dims{};
}

DimsExprs ProposalDynamicPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        // Validate input arguments
        PLUGIN_VALIDATE(outputIndex == 0);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(inputs[0].nbDims == 4);
        PLUGIN_VALIDATE(inputs[1].nbDims == 4);
        DimsExprs outDim;
        outDim.nbDims = 4;
        outDim.d[0] = inputs[0].d[0];
        outDim.d[1] = exprBuilder.constant(mMaxBoxNum);
        outDim.d[2] = exprBuilder.constant(4);
        outDim.d[3] = exprBuilder.constant(1);
        return outDim;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

int32_t ProposalPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

int32_t ProposalDynamicPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

size_t ProposalPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept
{
    return _get_workspace_size(maxBatchSize, mAnchorSizeNum, mAnchorRatioNum, mRpnHeight, mRpnWidth, mMaxBoxNum);
}

size_t ProposalDynamicPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    int32_t batchSize = inputs[0].dims.d[0];
    return _get_workspace_size(batchSize, mAnchorSizeNum, mAnchorRatioNum, mRpnHeight, mRpnWidth, mMaxBoxNum);
}

int32_t ProposalPlugin::enqueue(
    int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        int32_t status = STATUS_FAILURE;
        // Our plugin outputs only one tensor
        void* output = outputs[0];
        status = proposalInference_gpu(stream, inputs[0], inputs[1], batchSize, mInputHeight, mInputWidth, mRpnHeight,
            mRpnWidth, mMaxBoxNum, mPreNmsTopN, &mAnchorSizes[0], mAnchorSizeNum, &mAnchorRatios[0], mAnchorRatioNum,
            mRpnStdScaling, mRpnStride, mBboxMinSize, mNmsIouThreshold, workspace, output);
        return status;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

int32_t ProposalDynamicPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr && inputs != nullptr && outputs != nullptr && workspace != nullptr);

        int32_t status = STATUS_FAILURE;
        // Our plugin outputs only one tensor
        void* output = outputs[0];
        int32_t batchSize = inputDesc[0].dims.d[0];
        status = proposalInference_gpu(stream, inputs[0], inputs[1], batchSize, mInputHeight, mInputWidth, mRpnHeight,
            mRpnWidth, mMaxBoxNum, mPreNmsTopN, &mAnchorSizes[0], mAnchorSizeNum, &mAnchorRatios[0], mAnchorRatioNum,
            mRpnStdScaling, mRpnStride, mBboxMinSize, mNmsIouThreshold, workspace, output);
        return status;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
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
    PLUGIN_ASSERT(buffer != nullptr);
    uint8_t* d = reinterpret_cast<uint8_t*>(buffer);
    uint8_t* a = d;
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
    PLUGIN_ASSERT(buffer != nullptr);
    uint8_t* d = reinterpret_cast<uint8_t*>(buffer);
    uint8_t* a = d;
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
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

bool ProposalDynamicPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        // 2 inputs, 1 outputs, so 3 input/output in total
        PLUGIN_VALIDATE(0 <= pos && pos < 3);
        auto const* in = inOut;
        auto const* out = inOut + nbInputs;
        bool const consistentFloatPrecision = (in[0].type == in[pos].type);
        switch (pos)
        {
        case 0:
            return in[0].type == DataType::kFLOAT && in[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
        case 1:
            return in[1].type == DataType::kFLOAT && in[1].format == PluginFormat::kLINEAR && consistentFloatPrecision;
        case 2:
            return out[0].type == DataType::kFLOAT && out[0].format == PluginFormat::kLINEAR
                && consistentFloatPrecision;
        }
        return false;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
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

void ProposalPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

void ProposalDynamicPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* ProposalPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* ProposalDynamicPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index.
DataType ProposalPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        // one outputs
        PLUGIN_VALIDATE(index == 0);
        return DataType::kFLOAT;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DataType{};
}

DataType ProposalDynamicPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        // one outputs
        PLUGIN_VALIDATE(index == 0);
        return DataType::kFLOAT;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DataType{};
}

// Return true if output tensor is broadcast across a batch.
bool ProposalPlugin::isOutputBroadcastAcrossBatch(
    int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without
// replication.
bool ProposalPlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{
    return false;
}

void ProposalPlugin::configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
    DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
    bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept
{
    try
    {
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);
        mRpnHeight = inputDims->d[1];
        mRpnWidth = inputDims->d[2];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

void ProposalDynamicPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);
        mRpnHeight = in[0].desc.dims.d[2];
        mRpnWidth = in[0].desc.dims.d[3];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

// Attach the plugin object to an execution context and grant the plugin the
// access to some context resource.
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
    mPluginName = kPROPOSAL_PLUGIN_NAMES[0];
}

ProposalDynamicPluginCreator::ProposalDynamicPluginCreator() noexcept
{
    mPluginName = kPROPOSAL_PLUGIN_NAMES[1];
}

char const* ProposalBasePluginCreator::getPluginName() const noexcept
{
    return mPluginName.c_str();
}

char const* ProposalBasePluginCreator::getPluginVersion() const noexcept
{
    return kPROPOSAL_PLUGIN_VERSION;
}

PluginFieldCollection const* ProposalBasePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* ProposalPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        gLogWarning << "Proposal plugin (implementing IPluginV2Ext) is deprecated since TensorRT 9.0. Use "
                       "ProposalDynamic plugin."
                    << std::endl;
        PluginField const* fields = fc->fields;
        int32_t nbFields = fc->nbFields;
        int32_t inputHeight = 0;
        int32_t inputWidth = 0;
        int32_t rpnStride = 0;
        int32_t preNmsTopN = 0;
        int32_t postNmsTopN = 0;
        float roiMinSize = 0.0F;
        float nmsIouThreshold = 0.0F;
        std::vector<float> anchorSizes;
        std::vector<float> anchorRatios;

        for (int32_t i = 0; i < nbFields; ++i)
        {
            char const* attrName = fields[i].name;

            if (!strcmp(attrName, "input_height"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                inputHeight = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "input_width"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                inputWidth = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "rpn_stride"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                rpnStride = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "roi_min_size"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                roiMinSize = *(static_cast<float const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "nms_iou_threshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                nmsIouThreshold = *(static_cast<float const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "pre_nms_top_n"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                preNmsTopN = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "post_nms_top_n"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                postNmsTopN = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "anchor_sizes"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                float const* as = static_cast<float const*>(fields[i].data);

                for (int32_t j = 0; j < fields[i].length; ++j)
                {
                    PLUGIN_VALIDATE(*as > 0.0F);
                    anchorSizes.push_back(*as);
                    ++as;
                }
            }
            else if (!strcmp(attrName, "anchor_ratios"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                float const* ar = static_cast<float const*>(fields[i].data);

                // take the square root.
                for (int32_t j = 0; j < fields[i].length; ++j)
                {
                    PLUGIN_VALIDATE(*ar > 0.0F);
                    anchorRatios.push_back(std::sqrt(*ar));
                    ++ar;
                }
            }
        }

        PLUGIN_VALIDATE(inputHeight > 0);
        PLUGIN_VALIDATE(inputWidth > 0);
        PLUGIN_VALIDATE(rpnStride > 0);
        PLUGIN_VALIDATE(preNmsTopN > 0);
        PLUGIN_VALIDATE(postNmsTopN);
        PLUGIN_VALIDATE(roiMinSize >= 0.0F);
        PLUGIN_VALIDATE(nmsIouThreshold > 0.0F);

        IPluginV2Ext* plugin
            = new ProposalPlugin(inputHeight, inputWidth, kRPN_STD_SCALING, rpnStride, roiMinSize, nmsIouThreshold,
                preNmsTopN, postNmsTopN, &anchorSizes[0], anchorSizes.size(), &anchorRatios[0], anchorRatios.size());
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
    char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PluginField const* fields = fc->fields;
        int32_t nbFields = fc->nbFields;
        int32_t inputHeight = 0;
        int32_t inputWidth = 0;
        int32_t rpnStride = 0;
        int32_t preNmsTopN = 0;
        int32_t postNmsTopN = 0;
        float roiMinSize = 0.0F;
        float nmsIouThreshold = 0.0F;
        std::vector<float> anchorSizes;
        std::vector<float> anchorRatios;

        for (int32_t i = 0; i < nbFields; ++i)
        {
            char const* attrName = fields[i].name;

            if (!strcmp(attrName, "input_height"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                inputHeight = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "input_width"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                inputWidth = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "rpn_stride"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                rpnStride = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "roi_min_size"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                roiMinSize = *(static_cast<float const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "nms_iou_threshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                nmsIouThreshold = *(static_cast<float const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "pre_nms_top_n"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                preNmsTopN = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "post_nms_top_n"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                postNmsTopN = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "anchor_sizes"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                float const* as = static_cast<float const*>(fields[i].data);

                for (int32_t j = 0; j < fields[i].length; ++j)
                {
                    PLUGIN_VALIDATE(*as > 0.0F);
                    anchorSizes.push_back(*as);
                    ++as;
                }
            }
            else if (!strcmp(attrName, "anchor_ratios"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                float const* ar = static_cast<float const*>(fields[i].data);

                // take the square root.
                for (int32_t j = 0; j < fields[i].length; ++j)
                {
                    PLUGIN_VALIDATE(*ar > 0.0F);
                    anchorRatios.push_back(std::sqrt(*ar));
                    ++ar;
                }
            }
        }

        PLUGIN_VALIDATE(inputHeight > 0);
        PLUGIN_VALIDATE(inputWidth > 0);
        PLUGIN_VALIDATE(rpnStride > 0);
        PLUGIN_VALIDATE(preNmsTopN > 0);
        PLUGIN_VALIDATE(postNmsTopN);
        PLUGIN_VALIDATE(roiMinSize >= 0.0F);
        PLUGIN_VALIDATE(nmsIouThreshold > 0.0F);

        IPluginV2DynamicExt* plugin = new ProposalDynamicPlugin(inputHeight, inputWidth, kRPN_STD_SCALING, rpnStride,
            roiMinSize, nmsIouThreshold, preNmsTopN, postNmsTopN, &anchorSizes[0], anchorSizes.size(), &anchorRatios[0],
            anchorRatios.size());
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
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        gLogWarning << "Proposal plugin (implementing IPluginV2Ext) is deprecated since TensorRT 9.0. Use "
                       "ProposalDynamic plugin."
                    << std::endl;
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
    char const* name, void const* serialData, size_t serialLength) noexcept
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
