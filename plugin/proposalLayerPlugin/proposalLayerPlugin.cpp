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
#include "proposalLayerPlugin.h"
#include "common/mrcnn_config.h"
#include "common/plugin.h"
#include <algorithm>
#include <cuda_runtime_api.h>
#include <iostream>
#include <math.h>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::ProposalLayer;
using nvinfer1::plugin::ProposalLayerPluginCreator;

namespace
{
char const* const kPROPOSALLAYER_PLUGIN_VERSION{"1"};
char const* const kPROPOSALLAYER_PLUGIN_NAME{"ProposalLayer_TRT"};
} // namespace

ProposalLayerPluginCreator::ProposalLayerPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("prenms_topk", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("keep_topk", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("image_size", nullptr, PluginFieldType::kINT32, 3));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* ProposalLayerPluginCreator::getPluginName() const noexcept
{
    return kPROPOSALLAYER_PLUGIN_NAME;
}

char const* ProposalLayerPluginCreator::getPluginVersion() const noexcept
{
    return kPROPOSALLAYER_PLUGIN_VERSION;
}

PluginFieldCollection const* ProposalLayerPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* ProposalLayerPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        auto imageSize = MaskRCNNConfig::IMAGE_SHAPE;
        PluginField const* fields = fc->fields;
        plugin::validateRequiredAttributesExist({"prenms_topk", "keep_topk", "iou_threshold"}, fc);
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
            if (!strcmp(attrName, "iou_threshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                mIOUThreshold = *(static_cast<float const*>(fields[i].data));
            }
            if (!strcmp(attrName, "image_size"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                auto const* const dims = static_cast<int32_t const*>(fields[i].data);
                std::copy_n(dims, 3, imageSize.d);
            }
        }
        return new ProposalLayer(mPreNMSTopK, mKeepTopK, mIOUThreshold, imageSize);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* ProposalLayerPluginCreator::deserializePlugin(char const* name, void const* data, size_t length) noexcept
{
    try
    {
        return new ProposalLayer(data, length);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

ProposalLayer::ProposalLayer(
    int32_t prenms_topk, int32_t keep_topk, float iou_threshold, nvinfer1::Dims const& imageSize)
    : mPreNMSTopK(prenms_topk)
    , mKeepTopK(keep_topk)
    , mIOUThreshold(iou_threshold)
    , mImageSize(imageSize)
{
    mBackgroundLabel = -1;
    PLUGIN_VALIDATE(mPreNMSTopK > 0);
    PLUGIN_VALIDATE(mPreNMSTopK <= 1024);
    PLUGIN_VALIDATE(mKeepTopK > 0);
    PLUGIN_VALIDATE(iou_threshold > 0.0F);
    PLUGIN_VALIDATE(mImageSize.nbDims == 3);
    PLUGIN_VALIDATE(mImageSize.d[0] == 3 && mImageSize.d[1] > 0 && mImageSize.d[2] > 0);

    mParam.backgroundLabelId = -1;
    mParam.numClasses = 1;
    mParam.keepTopK = mKeepTopK;
    mParam.scoreThreshold = 0.0;
    mParam.iouThreshold = mIOUThreshold;

    mType = DataType::kFLOAT;

    generate_pyramid_anchors(imageSize);
}

int32_t ProposalLayer::getNbOutputs() const noexcept
{
    return 1;
}

int32_t ProposalLayer::initialize() noexcept
{
    // Init the mValidCnt of max batch size
    std::vector<int32_t> tempValidCnt(mMaxBatchSize, mPreNMSTopK);

    mValidCnt = std::make_shared<CudaBind<int32_t>>(mMaxBatchSize);

    PLUGIN_CUASSERT(cudaMemcpy(mValidCnt->mPtr, static_cast<void*>(tempValidCnt.data()),
        sizeof(int32_t) * mMaxBatchSize, cudaMemcpyHostToDevice));

    // Init the anchors for batch size:
    mAnchorBoxesDevice = std::make_shared<CudaBind<float>>(mAnchorsCnt * 4 * mMaxBatchSize);
    int32_t batch_offset = sizeof(float) * mAnchorsCnt * 4;
    uint8_t* device_ptr = static_cast<uint8_t*>(mAnchorBoxesDevice->mPtr);
    for (int32_t i = 0; i < mMaxBatchSize; i++)
    {
        PLUGIN_CUASSERT(cudaMemcpy(static_cast<void*>(device_ptr + i * batch_offset),
            static_cast<void*>(mAnchorBoxesHost.data()), batch_offset, cudaMemcpyHostToDevice));
    }

    return 0;
}

void ProposalLayer::terminate() noexcept {}

void ProposalLayer::destroy() noexcept
{
    delete this;
}

bool ProposalLayer::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

char const* ProposalLayer::getPluginType() const noexcept
{
    return kPROPOSALLAYER_PLUGIN_NAME;
}

char const* ProposalLayer::getPluginVersion() const noexcept
{
    return kPROPOSALLAYER_PLUGIN_VERSION;
}

IPluginV2Ext* ProposalLayer::clone() const noexcept
{
    try
    {
        auto* plugin = new ProposalLayer(*this);
        plugin->setPluginNamespace(mNameSpace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void ProposalLayer::setPluginNamespace(char const* libNamespace) noexcept
{
    mNameSpace = libNamespace;
}

char const* ProposalLayer::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

size_t ProposalLayer::getSerializationSize() const noexcept
{
    return sizeof(int32_t) * 2 + sizeof(float) + sizeof(int32_t) * 2 + sizeof(nvinfer1::Dims);
}

void ProposalLayer::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mPreNMSTopK);
    write(d, mKeepTopK);
    write(d, mIOUThreshold);
    write(d, mMaxBatchSize);
    write(d, mAnchorsCnt);
    write(d, mImageSize);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

ProposalLayer::ProposalLayer(void const* data, size_t length)
{
    deserialize(static_cast<int8_t const*>(data), length);
}

void ProposalLayer::deserialize(int8_t const* data, size_t length)
{
    auto const* d{data};
    int32_t prenms_topk = read<int32_t>(d);
    int32_t keep_topk = read<int32_t>(d);
    float iou_threshold = read<float>(d);
    mMaxBatchSize = read<int32_t>(d);
    mAnchorsCnt = read<int32_t>(d);
    mImageSize = read<nvinfer1::Dims3>(d);
    PLUGIN_VALIDATE(d == data + length);

    mBackgroundLabel = -1;
    mPreNMSTopK = prenms_topk;
    mKeepTopK = keep_topk;
    mIOUThreshold = iou_threshold;

    mParam.backgroundLabelId = -1;
    mParam.numClasses = 1;
    mParam.keepTopK = mKeepTopK;
    mParam.scoreThreshold = 0.0;
    mParam.iouThreshold = mIOUThreshold;

    mType = DataType::kFLOAT;

    generate_pyramid_anchors(mImageSize);
}

void ProposalLayer::check_valid_inputs(nvinfer1::Dims const* inputs, int32_t nbInputDims)
{
    // object_score[N, anchors, 2, 1],
    // foreground_delta[N, anchors, 4, 1],
    // anchors should be generated inside
    PLUGIN_ASSERT(nbInputDims == 2);
    // foreground_score
    PLUGIN_ASSERT(inputs[0].nbDims == 3 && inputs[0].d[1] == 2);
    // foreground_delta
    PLUGIN_ASSERT(inputs[1].nbDims == 3 && inputs[1].d[1] == 4);
}

size_t ProposalLayer::getWorkspaceSize(int32_t batch_size) const noexcept
{

    ProposalWorkSpace proposal(batch_size, mAnchorsCnt, mPreNMSTopK, mParam, mType);
    return proposal.totalSize;
}

Dims ProposalLayer::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept
{

    check_valid_inputs(inputs, nbInputDims);
    PLUGIN_ASSERT(index == 0);

    return {2, {mKeepTopK, 4}};
}

void ProposalLayer::generate_pyramid_anchors(nvinfer1::Dims const& imageDims)
{
    PLUGIN_VALIDATE(imageDims.nbDims == 3 && imageDims.d[0] == 3);

    auto const& scales = MaskRCNNConfig::RPN_ANCHOR_SCALES;
    auto const& ratios = MaskRCNNConfig::RPN_ANCHOR_RATIOS;
    auto const& strides = MaskRCNNConfig::BACKBONE_STRIDES;
    auto anchor_stride = MaskRCNNConfig::RPN_ANCHOR_STRIDE;

    float const cy = imageDims.d[1] - 1;
    float const cx = imageDims.d[2] - 1;

    auto& anchors = mAnchorBoxesHost;
    PLUGIN_VALIDATE(anchors.empty());

    PLUGIN_VALIDATE(scales.size() == strides.size());
    for (size_t s = 0; s < scales.size(); ++s)
    {
        float scale = scales[s];
        int32_t stride = strides[s];

        for (int32_t y = 0; y < imageDims.d[1]; y += anchor_stride * stride)
            for (int32_t x = 0; x < imageDims.d[2]; x += anchor_stride * stride)
                for (float r : ratios)
                {
                    float sqrt_r = sqrt(r);
                    float h = scale / sqrt_r;
                    float w = scale * sqrt_r;

                    anchors.insert(anchors.end(),
                        {(y - h / 2) / cy, (x - w / 2) / cx, (y + h / 2 - 1) / cy, (x + w / 2 - 1) / cx});
                }
    }

    PLUGIN_VALIDATE(anchors.size() % 4 == 0);
}

int32_t ProposalLayer::enqueue(
    int32_t batch_size, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{

    void* proposals = outputs[0];

    // proposal
    ProposalWorkSpace proposalWorkspace(batch_size, mAnchorsCnt, mPreNMSTopK, mParam, mType);
    cudaError_t status = proposalRefineBatchClassNMS(stream, batch_size, mAnchorsCnt, mPreNMSTopK,
        DataType::kFLOAT, // mType,
        mParam, proposalWorkspace, workspace,
        inputs[0], // inputs[object_score]
        inputs[1], // inputs[bbox_delta],
        mValidCnt->mPtr,
        mAnchorBoxesDevice->mPtr, // inputs[anchors]
        proposals);

    PLUGIN_ASSERT(status == cudaSuccess);
    return status;
}

// Return the DataType of the plugin output at the requested index
DataType ProposalLayer::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool ProposalLayer::isOutputBroadcastAcrossBatch(
    int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool ProposalLayer::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void ProposalLayer::configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
    DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
    bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept
{
    check_valid_inputs(inputDims, nbInputs);
    PLUGIN_ASSERT(inputDims[0].d[0] == inputDims[1].d[0]);

    mAnchorsCnt = inputDims[0].d[0];
    PLUGIN_ASSERT(mAnchorsCnt == (int32_t) (mAnchorBoxesHost.size() / 4));
    mMaxBatchSize = maxBatchSize;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void ProposalLayer::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void ProposalLayer::detachFromContext() noexcept {}
