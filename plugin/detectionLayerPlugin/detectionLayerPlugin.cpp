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
#include "detectionLayerPlugin.h"
#include "common/plugin.h"

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::DetectionLayer;
using nvinfer1::plugin::DetectionLayerPluginCreator;

namespace
{
char const* const kDETECTIONLAYER_PLUGIN_VERSION{"1"};
char const* const kDETECTIONLAYER_PLUGIN_NAME{"DetectionLayer_TRT"};
} // namespace

DetectionLayerPluginCreator::DetectionLayerPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_classes", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("keep_topk", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* DetectionLayerPluginCreator::getPluginName() const noexcept
{
    return kDETECTIONLAYER_PLUGIN_NAME;
}

char const* DetectionLayerPluginCreator::getPluginVersion() const noexcept
{
    return kDETECTIONLAYER_PLUGIN_VERSION;
}

PluginFieldCollection const* DetectionLayerPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* DetectionLayerPluginCreator::createPlugin(char const* /*name*/, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);
        plugin::validateRequiredAttributesExist({"num_classes", "keep_topk", "score_threshold", "iou_threshold"}, fc);
        PluginField const* fields = fc->fields;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "num_classes"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                mNbClasses = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "keep_topk"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                mKeepTopK = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "score_threshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                mScoreThreshold = *(static_cast<float const*>(fields[i].data));
            }
            if (!strcmp(attrName, "iou_threshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                mIOUThreshold = *(static_cast<float const*>(fields[i].data));
            }
        }
        return new DetectionLayer(mNbClasses, mKeepTopK, mScoreThreshold, mIOUThreshold);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* DetectionLayerPluginCreator::deserializePlugin(
    char const* /*name*/, void const* data, size_t length) noexcept
{
    try
    {
        return new DetectionLayer(data, length);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

DetectionLayer::DetectionLayer(int32_t numClasses, int32_t keepTopk, float scoreThreshold, float iouThreshold)
    : mNbClasses(numClasses)
    , mKeepTopK(keepTopk)
    , mScoreThreshold(scoreThreshold)
    , mIOUThreshold(iouThreshold)
{
    mBackgroundLabel = 0;
    PLUGIN_VALIDATE(mNbClasses > 0);
    PLUGIN_VALIDATE(mKeepTopK > 0);
    PLUGIN_VALIDATE(mScoreThreshold >= 0.F);
    PLUGIN_VALIDATE(mIOUThreshold > 0.F);

    mParam.backgroundLabelId = 0;
    mParam.numClasses = mNbClasses;
    mParam.keepTopK = mKeepTopK;
    mParam.scoreThreshold = mScoreThreshold;
    mParam.iouThreshold = mIOUThreshold;

    mType = DataType::kFLOAT;
}

int32_t DetectionLayer::getNbOutputs() const noexcept
{
    return 1;
}

int32_t DetectionLayer::initialize() noexcept
{
    try
    {
        // Init the mValidCnt and mDecodedBboxes for max batch size.
        std::vector<int32_t> tempValidCnt(mMaxBatchSize, mAnchorsCnt);

        mValidCnt = std::make_shared<CudaBind<int32_t>>(mMaxBatchSize);

        PLUGIN_CUASSERT(cudaMemcpy(mValidCnt->mPtr, static_cast<void*>(tempValidCnt.data()),
            sizeof(int32_t) * mMaxBatchSize, cudaMemcpyHostToDevice));

        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

void DetectionLayer::terminate() noexcept {}

void DetectionLayer::destroy() noexcept
{
    delete this;
}

bool DetectionLayer::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

char const* DetectionLayer::getPluginType() const noexcept
{
    return kDETECTIONLAYER_PLUGIN_NAME;
}

char const* DetectionLayer::getPluginVersion() const noexcept
{
    return kDETECTIONLAYER_PLUGIN_VERSION;
}

IPluginV2Ext* DetectionLayer::clone() const noexcept
{
    try
    {
        DetectionLayer* plugin = new DetectionLayer(*this);
        plugin->setPluginNamespace(mNameSpace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void DetectionLayer::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNameSpace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* DetectionLayer::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

size_t DetectionLayer::getSerializationSize() const noexcept
{
    return sizeof(int32_t) * 2 + sizeof(float) * 2 + sizeof(int32_t) * 2;
}

void DetectionLayer::serialize(void* buffer) const noexcept
{
    auto* d = reinterpret_cast<uint8_t*>(buffer);
    auto* const a = d;
    write(d, mNbClasses);
    write(d, mKeepTopK);
    write(d, mScoreThreshold);
    write(d, mIOUThreshold);
    write(d, mMaxBatchSize);
    write(d, mAnchorsCnt);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

DetectionLayer::DetectionLayer(void const* data, size_t length)
{
    auto const* d = reinterpret_cast<uint8_t const*>(data);
    auto const* const a = d;
    mNbClasses = read<int32_t>(d);
    mKeepTopK = read<int32_t>(d);
    mScoreThreshold = read<float>(d);
    mIOUThreshold = read<float>(d);
    mMaxBatchSize = read<int32_t>(d);
    mAnchorsCnt = read<int32_t>(d);
    PLUGIN_VALIDATE(d == a + length);

    mParam.backgroundLabelId = 0;
    mParam.numClasses = mNbClasses;
    mParam.keepTopK = mKeepTopK;
    mParam.scoreThreshold = mScoreThreshold;
    mParam.iouThreshold = mIOUThreshold;

    mType = DataType::kFLOAT;
}

void DetectionLayer::checkValidInputs(nvinfer1::Dims const* inputs, int32_t nbInputDims)
{
    // classifier_delta_bbox[N, anchors, num_classes*4, 1, 1]
    // classifier_class[N, anchors, num_classes, 1, 1]
    // rpn_rois[N, anchors, 4]
    PLUGIN_VALIDATE(nbInputDims == 3);
    // delta_bbox
    PLUGIN_VALIDATE(inputs[0].nbDims == 4 && inputs[0].d[1] == mNbClasses * 4);
    // score
    PLUGIN_VALIDATE(inputs[1].nbDims == 4 && inputs[1].d[1] == mNbClasses);
    // roi
    PLUGIN_VALIDATE(inputs[2].nbDims == 2 && inputs[2].d[1] == 4);
}

size_t DetectionLayer::getWorkspaceSize(int32_t batchSize) const noexcept
{
    RefineDetectionWorkSpace refine(batchSize, mAnchorsCnt, mParam, mType);
    return refine.totalSize;
}

Dims DetectionLayer::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept
{
    try
    {
        checkValidInputs(inputs, nbInputDims);
        PLUGIN_VALIDATE(index == 0);
        // [N, anchors, (y1, x1, y2, x2, class_id, score)]
        return {2, {mKeepTopK, 6}};
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return Dims{};
}

int32_t DetectionLayer::enqueue(
    int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);
        void* detections = outputs[0];

        // refine detection
        RefineDetectionWorkSpace refDetcWorkspace(batchSize, mAnchorsCnt, mParam, mType);
        cudaError_t status = RefineBatchClassNMS(stream, batchSize, mAnchorsCnt,
            DataType::kFLOAT, // mType,
            mParam, refDetcWorkspace, workspace,
            inputs[1],       // inputs[InScore]
            inputs[0],       // inputs[InDelta],
            mValidCnt->mPtr, // inputs[InCountValid],
            inputs[2],       // inputs[ROI]
            detections);

        return status;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

DataType DetectionLayer::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    // Only DataType::kFLOAT is acceptable by the plugin layer.
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool DetectionLayer::isOutputBroadcastAcrossBatch(
    int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool DetectionLayer::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void DetectionLayer::configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
    DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
    bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept
{
    try
    {
        checkValidInputs(inputDims, nbInputs);
        PLUGIN_VALIDATE(inputDims[0].d[0] == inputDims[1].d[0] && inputDims[1].d[0] == inputDims[2].d[0]);

        mAnchorsCnt = inputDims[2].d[0];
        mType = inputTypes[0];
        mMaxBatchSize = maxBatchSize;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void DetectionLayer::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void DetectionLayer::detachFromContext() noexcept {}
