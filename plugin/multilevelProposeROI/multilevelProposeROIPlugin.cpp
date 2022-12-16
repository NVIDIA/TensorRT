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
const char* MULTILEVELPROPOSEROI_PLUGIN_VERSION{"1"};
const char* MULTILEVELPROPOSEROI_PLUGIN_NAME{"MultilevelProposeROI_TRT"};
} // namespace

PluginFieldCollection MultilevelProposeROIPluginCreator::mFC{};
std::vector<PluginField> MultilevelProposeROIPluginCreator::mPluginAttributes;

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

const char* MultilevelProposeROIPluginCreator::getPluginName() const noexcept
{
    return MULTILEVELPROPOSEROI_PLUGIN_NAME;
}

const char* MultilevelProposeROIPluginCreator::getPluginVersion() const noexcept
{
    return MULTILEVELPROPOSEROI_PLUGIN_VERSION;
}

const PluginFieldCollection* MultilevelProposeROIPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* MultilevelProposeROIPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        plugin::validateRequiredAttributesExist({"prenms_topk", "keep_topk", "fg_threshold", "iou_threshold"}, fc);
        auto imageSize = TLTMaskRCNNConfig::IMAGE_SHAPE;
        PluginField const* fields = fc->fields;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
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
                const auto dims = static_cast<int32_t const*>(fields[i].data);
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
    const char* name, const void* data, size_t length) noexcept
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
    int prenms_topk, int keep_topk, float fg_threshold, float iou_threshold, const nvinfer1::Dims imageSize)
    : mPreNMSTopK(prenms_topk)
    , mKeepTopK(keep_topk)
    , mFGThreshold(fg_threshold)
    , mIOUThreshold(iou_threshold)
    , mImageSize(imageSize)
{
    mBackgroundLabel = -1;
    PLUGIN_VALIDATE(mPreNMSTopK > 0 && mPreNMSTopK <= 4096);
    PLUGIN_VALIDATE(mKeepTopK > 0);
    PLUGIN_VALIDATE(mIOUThreshold >= 0.0f);
    PLUGIN_VALIDATE(mFGThreshold >= 0.0f);
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

int MultilevelProposeROI::getNbOutputs() const noexcept
{
    return 1;
}

int MultilevelProposeROI::initialize() noexcept
{
    // Init the regWeight [1, 1, 1, 1]
    mRegWeightDevice = std::make_shared<CudaBind<float>>(4);
    std::vector<float> reg_weight(4, 1);
    PLUGIN_CUASSERT(cudaMemcpy(static_cast<void*>(mRegWeightDevice->mPtr), static_cast<void*>(reg_weight.data()),
        sizeof(float) * 4, cudaMemcpyHostToDevice));

    // Init the mValidCnt of max batch size
    std::vector<int> tempValidCnt(mMaxBatchSize, mPreNMSTopK);

    mValidCnt = std::make_shared<CudaBind<int>>(mMaxBatchSize);

    PLUGIN_CUASSERT(cudaMemcpy(
        mValidCnt->mPtr, static_cast<void*>(tempValidCnt.data()), sizeof(int) * mMaxBatchSize, cudaMemcpyHostToDevice));

    // Init the anchors for batch size:
    for (int i = 0; i < mFeatureCnt; i++)
    {
        int i_anchors_cnt = mAnchorsCnt[i];
        auto i_anchors_host = mAnchorBoxesHost[i].data();
        auto i_anchors_device = std::make_shared<CudaBind<float>>(i_anchors_cnt * 4 * mMaxBatchSize);
        int batch_offset = sizeof(float) * i_anchors_cnt * 4;
        uint8_t* device_ptr = static_cast<uint8_t*>(i_anchors_device->mPtr);
        for (int i = 0; i < mMaxBatchSize; i++)
        {
            PLUGIN_CUASSERT(cudaMemcpy(static_cast<void*>(device_ptr + i * batch_offset), static_cast<void*>(i_anchors_host),
                batch_offset, cudaMemcpyHostToDevice));
        }
        mAnchorBoxesDevice.push_back(i_anchors_device);
    }

    // Init the temp storage for proposals from feature maps before concat
    std::vector<void*> score_tp;
    std::vector<void*> box_tp;
    for (int i = 0; i < mFeatureCnt; i++)
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

void MultilevelProposeROI::terminate() noexcept
{
}

void MultilevelProposeROI::destroy() noexcept
{
    delete this;
}

bool MultilevelProposeROI::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return ((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kLINEAR);
}

const char* MultilevelProposeROI::getPluginType() const noexcept
{
    return "MultilevelProposeROI_TRT";
}

const char* MultilevelProposeROI::getPluginVersion() const noexcept
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

void MultilevelProposeROI::setPluginNamespace(const char* libNamespace) noexcept
{
    mNameSpace = libNamespace;
}

const char* MultilevelProposeROI::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

size_t MultilevelProposeROI::getSerializationSize() const noexcept
{
    return sizeof(int) * 2 + sizeof(float) * 2 + sizeof(int) * (mFeatureCnt + 1) + sizeof(nvinfer1::Dims) + sizeof(DataType);
}

void MultilevelProposeROI::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mPreNMSTopK);
    write(d, mKeepTopK);
    write(d, mFGThreshold);
    write(d, mIOUThreshold);
    write(d, mMaxBatchSize);
    for (int i = 0; i < mFeatureCnt; i++)
    {
        write(d, mAnchorsCnt[i]);
    }
    write(d, mImageSize);
    write(d, mType);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

MultilevelProposeROI::MultilevelProposeROI(const void* data, size_t length)
{
    mFeatureCnt = TLTMaskRCNNConfig::MAX_LEVEL - TLTMaskRCNNConfig::MIN_LEVEL + 1;

    const char *d = reinterpret_cast<const char*>(data), *a = d;
    int prenms_topk = read<int>(d);
    int keep_topk = read<int>(d);
    float fg_threshold = read<float>(d);
    float iou_threshold = read<float>(d);
    mMaxBatchSize = read<int>(d);
    PLUGIN_VALIDATE(mAnchorsCnt.size() == 0);
    for (int i = 0; i < mFeatureCnt; i++)
    {
        mAnchorsCnt.push_back(read<int>(d));
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

void MultilevelProposeROI::check_valid_inputs(const nvinfer1::Dims* inputs, int nbInputDims) noexcept
{
    // x=2,3,4,5,6
    // foreground_delta_px [N, h_x * w_x * anchors_per_location, 4, 1],
    // foreground_score_px [N, h_x * w_x * anchors_per_location, 1, 1],
    // anchors should be generated inside
    PLUGIN_ASSERT(nbInputDims == 2 * mFeatureCnt);
    for (int i = 0; i < 2 * mFeatureCnt; i += 2)
    {
        // foreground_delta
        PLUGIN_ASSERT(inputs[i].nbDims == 3 && inputs[i].d[1] == 4);
        // foreground_score
        PLUGIN_ASSERT(inputs[i + 1].nbDims == 3 && inputs[i + 1].d[1] == 1);
    }
}

size_t MultilevelProposeROI::getWorkspaceSize(int batch_size) const noexcept
{
    size_t total_size = 0;
    PLUGIN_ASSERT(mAnchorsCnt.size() == static_cast<size_t>(mFeatureCnt));

    // workspace for propose on each feature map
    for (int i = 0; i < mFeatureCnt; i++)
    {

        MultilevelProposeROIWorkSpace proposal(batch_size, mAnchorsCnt[i], mPreNMSTopK, mParam, mType);
        total_size += proposal.totalSize;
    }

    // workspace for Concat and TopK
    ConcatTopKWorkSpace ct(batch_size, mFeatureCnt, mKeepTopK, mType);
    total_size += ct.totalSize;

    return total_size;
}

Dims MultilevelProposeROI::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{

    check_valid_inputs(inputs, nbInputDims);
    PLUGIN_ASSERT(index == 0);

    // [N, anchors, (y1, x1, y2, x2)]
    nvinfer1::Dims proposals;

    proposals.nbDims = 2;
    // number of keeping anchors
    proposals.d[0] = mKeepTopK;
    proposals.d[1] = 4;

    return proposals;
}

void MultilevelProposeROI::generate_pyramid_anchors(nvinfer1::Dims const& imageSize)
{
    const auto image_dims = imageSize;

    const auto& anchor_scale = TLTMaskRCNNConfig::RPN_ANCHOR_SCALE;
    const auto& min_level = TLTMaskRCNNConfig::MIN_LEVEL;
    const auto& max_level = TLTMaskRCNNConfig::MAX_LEVEL;
    const auto& aspect_ratios = TLTMaskRCNNConfig::ANCHOR_RATIOS;

    // Generate anchors strides and scales
    std::vector<float> anchor_scales;
    std::vector<int> anchor_strides;
    for (int i = min_level; i < max_level + 1; i++)
    {
        int stride = static_cast<int>(pow(2.0, i));
        anchor_strides.push_back(stride);
        anchor_scales.push_back(stride * anchor_scale);
    }

    auto& anchors = mAnchorBoxesHost;
    PLUGIN_VALIDATE(anchors.size() == 0);

    PLUGIN_VALIDATE(anchor_scales.size() == anchor_strides.size());
    for (size_t s = 0; s < anchor_scales.size(); ++s)
    {
        float scale = anchor_scales[s];
        int stride = anchor_strides[s];

        std::vector<float> s_anchors;
        for (int y = stride / 2; y < image_dims.d[1]; y += stride)
            for (int x = stride / 2; x < image_dims.d[2]; x += stride)
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
    int32_t batch_size, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{

    void* final_proposals = outputs[0];
    size_t kernel_workspace_offset = 0;
    cudaError_t status;

    std::vector<void*> mTempScores;
    std::vector<void*> mTempBboxes;

    for (int i = 0; i < mFeatureCnt; i++)
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

    for (int i = 0; i < mFeatureCnt; i++)
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
            mTempScores[i],        // temp scores [batch_size, topk, 1]
            mTempBboxes[i]);       // temp
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
DataType MultilevelProposeROI::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // Only DataType::kFLOAT is acceptable by the plugin layer
    if ((inputTypes[0] == DataType::kFLOAT) || (inputTypes[0] == DataType::kHALF))
        return inputTypes[0];
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool MultilevelProposeROI::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool MultilevelProposeROI::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void MultilevelProposeROI::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    check_valid_inputs(inputDims, nbInputs);

    mAnchorsCnt.clear();
    for (int i = 0; i < mFeatureCnt; i++)
    {
        mAnchorsCnt.push_back(inputDims[2 * i].d[0]);
        PLUGIN_ASSERT(mAnchorsCnt[i] == (int) (mAnchorBoxesHost[i].size() / 4));
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
void MultilevelProposeROI::detachFromContext() noexcept
{
}
