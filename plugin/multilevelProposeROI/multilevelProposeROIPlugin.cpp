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

#include "multilevelProposeROIPlugin.h"
#include "common/plugin.h"
#include "multilevelProposeROI/tlt_mrcnn_config.h"
#include <algorithm>
#include <cuda_runtime_api.h>
#include <iostream>
#include <math.h>

#include <fstream>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::MultilevelProposeROI;
using nvinfer1::plugin::MultilevelProposeROIPluginCreator;

namespace
{
char const* const kMULTILEVELPROPOSEROI_PLUGIN_VERSION{"1"};
char const* const kMULTILEVELPROPOSEROI_PLUGIN_NAME{"MultilevelProposeROI_TRT"};
} // namespace

MultilevelProposeROIPluginCreator::MultilevelProposeROIPluginCreator() noexcept
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("prenms_topk", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("keep_topk", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("fg_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("image_size", nullptr, PluginFieldType::kINT32, 3));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* MultilevelProposeROIPluginCreator::getPluginName() const noexcept
{
    return kMULTILEVELPROPOSEROI_PLUGIN_NAME;
}

char const* MultilevelProposeROIPluginCreator::getPluginVersion() const noexcept
{
    return kMULTILEVELPROPOSEROI_PLUGIN_VERSION;
}

PluginFieldCollection const* MultilevelProposeROIPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* MultilevelProposeROIPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        plugin::validateRequiredAttributesExist({"prenms_topk", "keep_topk", "fg_threshold", "iou_threshold"}, fc);
        auto imageSize = TLTMaskRCNNConfig::IMAGE_SHAPE;
        PluginField const* fields = fc->fields;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "prenms_topk"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                mPreNMSTopK = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "keep_topk"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                mKeepTopK = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "fg_threshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                mFGThreshold = *(static_cast<float const*>(fields[i].data));
            }
            if (!strcmp(attrName, "iou_threshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                mIOUThreshold = *(static_cast<float const*>(fields[i].data));
            }
            if (!strcmp(attrName, "image_size"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                auto const dims = static_cast<int32_t const*>(fields[i].data);
                std::copy_n(dims, 3, imageSize.d);
            }
        }
        return new MultilevelProposeROI(mPreNMSTopK, mKeepTopK, mFGThreshold, mIOUThreshold, imageSize);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* MultilevelProposeROIPluginCreator::deserializePlugin(
    char const* name, void const* data, size_t length) noexcept
{
    try
    {
        return new MultilevelProposeROI(data, length);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

MultilevelProposeROI::MultilevelProposeROI(
    int32_t prenms_topk, int32_t keep_topk, float fg_threshold, float iou_threshold, const nvinfer1::Dims imageSize)
    : mPreNMSTopK(prenms_topk)
    , mKeepTopK(keep_topk)
    , mFGThreshold(fg_threshold)
    , mIOUThreshold(iou_threshold)
    , mImageSize(imageSize)
{
    mBackgroundLabel = -1;
    PLUGIN_VALIDATE(mPreNMSTopK > 0);
    PLUGIN_VALIDATE(mPreNMSTopK <= 4096);
    PLUGIN_VALIDATE(mKeepTopK > 0);
    PLUGIN_VALIDATE(mIOUThreshold >= 0.0F);
    PLUGIN_VALIDATE(mFGThreshold >= 0.0F);
    PLUGIN_VALIDATE(mImageSize.nbDims == 3);
    PLUGIN_VALIDATE(mImageSize.d[0] > 0 && mImageSize.d[1] > 0 && mImageSize.d[2] > 0);

    mParam.backgroundLabelId = -1;
    mParam.numClasses = 1;
    mParam.keepTopK = mKeepTopK;
    mParam.scoreThreshold = mFGThreshold;
    mParam.iouThreshold = mIOUThreshold;

    mType = DataType::kFLOAT;

    mFeatureCnt = TLTMaskRCNNConfig::MAX_LEVEL - TLTMaskRCNNConfig::MIN_LEVEL + 1;

    generate_pyramid_anchors(mImageSize);
}

int32_t MultilevelProposeROI::getNbOutputs() const noexcept
{
    return 1;
}

int32_t MultilevelProposeROI::initialize() noexcept
{
    // Init the regWeight [1, 1, 1, 1]
    mRegWeightDevice = std::make_shared<CudaBind<float>>(4);
    std::vector<float> reg_weight(4, 1);
    PLUGIN_CUASSERT(cudaMemcpy(static_cast<void*>(mRegWeightDevice->mPtr), static_cast<void*>(reg_weight.data()),
        sizeof(float) * 4, cudaMemcpyHostToDevice));

    // Init the mValidCnt of max batch size
    std::vector<int32_t> tempValidCnt(mMaxBatchSize, mPreNMSTopK);

    mValidCnt = std::make_shared<CudaBind<int32_t>>(mMaxBatchSize);

    PLUGIN_CUASSERT(cudaMemcpy(mValidCnt->mPtr, static_cast<void*>(tempValidCnt.data()),
        sizeof(int32_t) * mMaxBatchSize, cudaMemcpyHostToDevice));

    // Init the anchors for batch size:
    for (int32_t i = 0; i < mFeatureCnt; i++)
    {
        int32_t i_anchors_cnt = mAnchorsCnt[i];
        auto i_anchors_host = mAnchorBoxesHost[i].data();
        auto i_anchors_device = std::make_shared<CudaBind<float>>(i_anchors_cnt * 4 * mMaxBatchSize);
        int32_t batch_offset = sizeof(float) * i_anchors_cnt * 4;
        uint8_t* device_ptr = static_cast<uint8_t*>(i_anchors_device->mPtr);
        for (int32_t i = 0; i < mMaxBatchSize; i++)
        {
            PLUGIN_CUASSERT(cudaMemcpy(static_cast<void*>(device_ptr + i * batch_offset),
                static_cast<void*>(i_anchors_host), batch_offset, cudaMemcpyHostToDevice));
        }
        mAnchorBoxesDevice.push_back(i_anchors_device);
    }

    // Init the temp storage for proposals from feature maps before concat
    std::vector<void*> score_tp;
    std::vector<void*> box_tp;
    for (int32_t i = 0; i < mFeatureCnt; i++)
    {
        if (mType == DataType::kFLOAT)
        {
            auto i_scores_device = std::make_shared<CudaBind<float>>(mKeepTopK * mMaxBatchSize);
            auto i_bboxes_device = std::make_shared<CudaBind<float>>(mKeepTopK * 4 * mMaxBatchSize);
            mTempScores_float.push_back(i_scores_device);
            score_tp.push_back(static_cast<void*>(i_scores_device->mPtr));
            mTempBboxes_float.push_back(i_bboxes_device);
            box_tp.push_back(static_cast<void*>(i_bboxes_device->mPtr));
        }
        else if (mType == DataType::kHALF)
        {
            auto i_scores_device = std::make_shared<CudaBind<uint16_t>>(mKeepTopK * mMaxBatchSize);
            auto i_bboxes_device = std::make_shared<CudaBind<uint16_t>>(mKeepTopK * 4 * mMaxBatchSize);
            mTempScores_half.push_back(i_scores_device);
            score_tp.push_back(static_cast<void*>(i_scores_device->mPtr));
            mTempBboxes_half.push_back(i_bboxes_device);
            box_tp.push_back(static_cast<void*>(i_bboxes_device->mPtr));
        }
    }

    // Init the temp storage for pointer arrays of score and box:
    PLUGIN_CUASSERT(cudaMalloc(&mDeviceScores, sizeof(void*) * mFeatureCnt));
    PLUGIN_CUASSERT(cudaMalloc(&mDeviceBboxes, sizeof(void*) * mFeatureCnt));

    PLUGIN_CUASSERT(cudaMemcpy(mDeviceScores, score_tp.data(), sizeof(void*) * mFeatureCnt, cudaMemcpyHostToDevice));
    PLUGIN_CUASSERT(cudaMemcpy(mDeviceBboxes, box_tp.data(), sizeof(void*) * mFeatureCnt, cudaMemcpyHostToDevice));

    return 0;
}

void MultilevelProposeROI::terminate() noexcept {}

void MultilevelProposeROI::destroy() noexcept
{
    delete this;
}

bool MultilevelProposeROI::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return ((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kLINEAR);
}

char const* MultilevelProposeROI::getPluginType() const noexcept
{
    return "MultilevelProposeROI_TRT";
}

char const* MultilevelProposeROI::getPluginVersion() const noexcept
{
    return "1";
}

IPluginV2Ext* MultilevelProposeROI::clone() const noexcept
{
    try
    {
        return new MultilevelProposeROI(*this);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void MultilevelProposeROI::setPluginNamespace(char const* libNamespace) noexcept
{
    mNameSpace = libNamespace;
}

char const* MultilevelProposeROI::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

size_t MultilevelProposeROI::getSerializationSize() const noexcept
{
    return sizeof(int32_t) * 2 + sizeof(float) * 2 + sizeof(int32_t) * (mFeatureCnt + 1) + sizeof(nvinfer1::Dims)
        + sizeof(DataType);
}

void MultilevelProposeROI::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mPreNMSTopK);
    write(d, mKeepTopK);
    write(d, mFGThreshold);
    write(d, mIOUThreshold);
    write(d, mMaxBatchSize);
    for (int32_t i = 0; i < mFeatureCnt; i++)
    {
        write(d, mAnchorsCnt[i]);
    }
    write(d, mImageSize);
    write(d, mType);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

MultilevelProposeROI::MultilevelProposeROI(void const* data, size_t length)
{
    mFeatureCnt = TLTMaskRCNNConfig::MAX_LEVEL - TLTMaskRCNNConfig::MIN_LEVEL + 1;

    char const *d = reinterpret_cast<char const*>(data), *a = d;
    int32_t prenms_topk = read<int32_t>(d);
    int32_t keep_topk = read<int32_t>(d);
    float fg_threshold = read<float>(d);
    float iou_threshold = read<float>(d);
    mMaxBatchSize = read<int32_t>(d);
    PLUGIN_VALIDATE(mAnchorsCnt.size() == 0);
    for (int32_t i = 0; i < mFeatureCnt; i++)
    {
        mAnchorsCnt.push_back(read<int32_t>(d));
    }
    mImageSize = read<nvinfer1::Dims3>(d);
    mType = read<DataType>(d);
    PLUGIN_VALIDATE(d == a + length);

    mBackgroundLabel = -1;
    mPreNMSTopK = prenms_topk;
    mKeepTopK = keep_topk;
    mFGThreshold = fg_threshold;
    mIOUThreshold = iou_threshold;

    mParam.backgroundLabelId = -1;
    mParam.numClasses = 1;
    mParam.keepTopK = mKeepTopK;
    mParam.scoreThreshold = mFGThreshold;
    mParam.iouThreshold = mIOUThreshold;

    generate_pyramid_anchors(mImageSize);
}

void MultilevelProposeROI::check_valid_inputs(nvinfer1::Dims const* inputs, int32_t nbInputDims) noexcept
{
    // x=2,3,4,5,6
    // foreground_delta_px [N, h_x * w_x * anchors_per_location, 4, 1],
    // foreground_score_px [N, h_x * w_x * anchors_per_location, 1, 1],
    // anchors should be generated inside
    PLUGIN_ASSERT(nbInputDims == 2 * mFeatureCnt);
    for (int32_t i = 0; i < 2 * mFeatureCnt; i += 2)
    {
        // foreground_delta
        PLUGIN_ASSERT(inputs[i].nbDims == 3 && inputs[i].d[1] == 4);
        // foreground_score
        PLUGIN_ASSERT(inputs[i + 1].nbDims == 3 && inputs[i + 1].d[1] == 1);
    }
}

size_t MultilevelProposeROI::getWorkspaceSize(int32_t batch_size) const noexcept
{
    size_t total_size = 0;
    PLUGIN_ASSERT(mAnchorsCnt.size() == static_cast<size_t>(mFeatureCnt));

    // workspace for propose on each feature map
    for (int32_t i = 0; i < mFeatureCnt; i++)
    {

        MultilevelProposeROIWorkSpace proposal(batch_size, mAnchorsCnt[i], mPreNMSTopK, mParam, mType);
        total_size += proposal.totalSize;
    }

    // workspace for Concat and TopK
    ConcatTopKWorkSpace ct(batch_size, mFeatureCnt, mKeepTopK, mType);
    total_size += ct.totalSize;

    return total_size;
}

Dims MultilevelProposeROI::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept
{

    check_valid_inputs(inputs, nbInputDims);
    PLUGIN_ASSERT(index == 0);

    return {2, {mKeepTopK, 4}};
}

void MultilevelProposeROI::generate_pyramid_anchors(nvinfer1::Dims const& imageSize)
{
    auto const image_dims = imageSize;

    auto const& anchor_scale = TLTMaskRCNNConfig::RPN_ANCHOR_SCALE;
    auto const& min_level = TLTMaskRCNNConfig::MIN_LEVEL;
    auto const& max_level = TLTMaskRCNNConfig::MAX_LEVEL;
    auto const& aspect_ratios = TLTMaskRCNNConfig::ANCHOR_RATIOS;

    // Generate anchors strides and scales
    std::vector<float> anchor_scales;
    std::vector<int32_t> anchor_strides;
    for (int32_t i = min_level; i < max_level + 1; i++)
    {
        int32_t stride = static_cast<int32_t>(pow(2.0, i));
        anchor_strides.push_back(stride);
        anchor_scales.push_back(stride * anchor_scale);
    }

    auto& anchors = mAnchorBoxesHost;
    PLUGIN_VALIDATE(anchors.size() == 0);

    PLUGIN_VALIDATE(anchor_scales.size() == anchor_strides.size());
    for (size_t s = 0; s < anchor_scales.size(); ++s)
    {
        float scale = anchor_scales[s];
        int32_t stride = anchor_strides[s];

        std::vector<float> s_anchors;
        for (int32_t y = stride / 2; y < image_dims.d[1]; y += stride)
            for (int32_t x = stride / 2; x < image_dims.d[2]; x += stride)
                for (auto r : aspect_ratios)
                {
                    float h = scale * r.second;
                    float w = scale * r.first;

                    // Using y+h/2 instead of y+h/2-1 for alignment with TLT implementation
                    s_anchors.insert(s_anchors.end(), {(y - h / 2), (x - w / 2), (y + h / 2), (x + w / 2)});
                }

        anchors.push_back(s_anchors);
    }

    PLUGIN_VALIDATE(anchors.size() == static_cast<size_t>(max_level - min_level + 1));
}

int32_t MultilevelProposeROI::enqueue(
    int32_t batch_size, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{

    void* final_proposals = outputs[0];
    size_t kernel_workspace_offset = 0;
    cudaError_t status;

    std::vector<void*> mTempScores;
    std::vector<void*> mTempBboxes;

    for (int32_t i = 0; i < mFeatureCnt; i++)
    {
        if (mType == DataType::kFLOAT)
        {
            mTempScores.push_back(mTempScores_float[i]->mPtr);
            mTempBboxes.push_back(mTempBboxes_float[i]->mPtr);
        }
        else if (mType == DataType::kHALF)
        {
            mTempScores.push_back(mTempScores_half[i]->mPtr);
            mTempBboxes.push_back(mTempBboxes_half[i]->mPtr);
        }
    }

    for (int32_t i = 0; i < mFeatureCnt; i++)
    {
        MultilevelProposeROIWorkSpace proposal_ws(batch_size, mAnchorsCnt[i], mPreNMSTopK, mParam, mType);
        status = MultilevelPropose(stream, batch_size, mAnchorsCnt[i], mPreNMSTopK,
            static_cast<float*>(mRegWeightDevice->mPtr),
            static_cast<float>(mImageSize.d[1]), // Input Height
            static_cast<float>(mImageSize.d[2]),
            mType, // mType,
            mParam, proposal_ws, static_cast<uint8_t*>(workspace) + kernel_workspace_offset,
            inputs[2 * i + 1], // inputs[object_score],
            inputs[2 * i],     // inputs[bbox_delta]
            mValidCnt->mPtr,
            mAnchorBoxesDevice[i]->mPtr, // inputs[anchors]
            mTempScores[i],              // temp scores [batch_size, topk, 1]
            mTempBboxes[i]);             // temp
        PLUGIN_ASSERT(status == cudaSuccess);
        kernel_workspace_offset += proposal_ws.totalSize;
    }

    ConcatTopKWorkSpace ctopk_ws(batch_size, mFeatureCnt, mKeepTopK, mType);
    status = ConcatTopK(stream, batch_size, mFeatureCnt, mKeepTopK, mType,
        static_cast<uint8_t*>(workspace) + kernel_workspace_offset, ctopk_ws, reinterpret_cast<void**>(mDeviceScores),
        reinterpret_cast<void**>(mDeviceBboxes), final_proposals);

    PLUGIN_ASSERT(status == cudaSuccess);
    return status;
}

// Return the DataType of the plugin output at the requested index
DataType MultilevelProposeROI::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    // Only DataType::kFLOAT is acceptable by the plugin layer
    if ((inputTypes[0] == DataType::kFLOAT) || (inputTypes[0] == DataType::kHALF))
        return inputTypes[0];
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool MultilevelProposeROI::isOutputBroadcastAcrossBatch(
    int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool MultilevelProposeROI::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void MultilevelProposeROI::configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims,
    int32_t nbOutputs, DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
    bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept
{
    check_valid_inputs(inputDims, nbInputs);

    mAnchorsCnt.clear();
    for (int32_t i = 0; i < mFeatureCnt; i++)
    {
        mAnchorsCnt.push_back(inputDims[2 * i].d[0]);
        PLUGIN_ASSERT(mAnchorsCnt[i] == (int32_t) (mAnchorBoxesHost[i].size() / 4));
    }

    mMaxBatchSize = maxBatchSize;

    mType = inputTypes[0];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void MultilevelProposeROI::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void MultilevelProposeROI::detachFromContext() noexcept {}
