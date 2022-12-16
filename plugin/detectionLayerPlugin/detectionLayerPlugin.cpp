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
#include "detectionLayerPlugin.h"
#include "common/plugin.h"
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::DetectionLayer;
using nvinfer1::plugin::DetectionLayerPluginCreator;

namespace
{
const char* DETECTIONLAYER_PLUGIN_VERSION{"1"};
const char* DETECTIONLAYER_PLUGIN_NAME{"DetectionLayer_TRT"};
} // namespace

PluginFieldCollection DetectionLayerPluginCreator::mFC{};
std::vector<PluginField> DetectionLayerPluginCreator::mPluginAttributes;

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

const char* DetectionLayerPluginCreator::getPluginName() const noexcept
{
    return DETECTIONLAYER_PLUGIN_NAME;
}

const char* DetectionLayerPluginCreator::getPluginVersion() const noexcept
{
    return DETECTIONLAYER_PLUGIN_VERSION;
}

const PluginFieldCollection* DetectionLayerPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* DetectionLayerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        plugin::validateRequiredAttributesExist({"num_classes", "keep_topk", "score_threshold", "iou_threshold"}, fc);
        PluginField const* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "num_classes"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                mNbClasses = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "keep_topk"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                mKeepTopK = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "score_threshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                mScoreThreshold = *(static_cast<const float*>(fields[i].data));
            }
            if (!strcmp(attrName, "iou_threshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                mIOUThreshold = *(static_cast<const float*>(fields[i].data));
            }
        }
        return new DetectionLayer(mNbClasses, mKeepTopK, mScoreThreshold, mIOUThreshold);
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* DetectionLayerPluginCreator::deserializePlugin(const char* name, const void* data, size_t length) noexcept
{
    try
    {
        return new DetectionLayer(data, length);
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

DetectionLayer::DetectionLayer(int num_classes, int keep_topk, float score_threshold, float iou_threshold)
    : mNbClasses(num_classes)
    , mKeepTopK(keep_topk)
    , mScoreThreshold(score_threshold)
    , mIOUThreshold(iou_threshold)
{
    mBackgroundLabel = 0;
    PLUGIN_VALIDATE(mNbClasses > 0);
    PLUGIN_VALIDATE(mKeepTopK > 0);
    PLUGIN_VALIDATE(score_threshold >= 0.0f);
    PLUGIN_VALIDATE(iou_threshold > 0.0f);

    mParam.backgroundLabelId = 0;
    mParam.numClasses = mNbClasses;
    mParam.keepTopK = mKeepTopK;
    mParam.scoreThreshold = mScoreThreshold;
    mParam.iouThreshold = mIOUThreshold;

    mType = DataType::kFLOAT;
}

int DetectionLayer::getNbOutputs() const noexcept
{
    return 1;
}

int DetectionLayer::initialize() noexcept
{
    //@Init the mValidCnt and mDecodedBboxes for max batch size
    std::vector<int> tempValidCnt(mMaxBatchSize, mAnchorsCnt);

    mValidCnt = std::make_shared<CudaBind<int>>(mMaxBatchSize);

    PLUGIN_CUASSERT(cudaMemcpy(
        mValidCnt->mPtr, static_cast<void*>(tempValidCnt.data()), sizeof(int) * mMaxBatchSize, cudaMemcpyHostToDevice));

    return 0;
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

const char* DetectionLayer::getPluginType() const noexcept
{
    return "DetectionLayer_TRT";
}

const char* DetectionLayer::getPluginVersion() const noexcept
{
    return "1";
}

IPluginV2Ext* DetectionLayer::clone() const noexcept
{
    try
    {
        DetectionLayer* plugin = new DetectionLayer(*this);
        plugin->setPluginNamespace(mNameSpace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void DetectionLayer::setPluginNamespace(const char* libNamespace) noexcept
{
    try
    {
        mNameSpace = libNamespace;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

const char* DetectionLayer::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

size_t DetectionLayer::getSerializationSize() const noexcept
{
    return sizeof(int) * 2 + sizeof(float) * 2 + sizeof(int) * 2;
}

void DetectionLayer::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mNbClasses);
    write(d, mKeepTopK);
    write(d, mScoreThreshold);
    write(d, mIOUThreshold);
    write(d, mMaxBatchSize);
    write(d, mAnchorsCnt);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

DetectionLayer::DetectionLayer(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    int num_classes = read<int>(d);
    int keep_topk = read<int>(d);
    float score_threshold = read<float>(d);
    float iou_threshold = read<float>(d);
    mMaxBatchSize = read<int>(d);
    mAnchorsCnt = read<int>(d);
    PLUGIN_VALIDATE(d == a + length);

    mNbClasses = num_classes;
    mKeepTopK = keep_topk;
    mScoreThreshold = score_threshold;
    mIOUThreshold = iou_threshold;

    mParam.backgroundLabelId = 0;
    mParam.numClasses = mNbClasses;
    mParam.keepTopK = mKeepTopK;
    mParam.scoreThreshold = mScoreThreshold;
    mParam.iouThreshold = mIOUThreshold;

    mType = DataType::kFLOAT;
}

void DetectionLayer::check_valid_inputs(const nvinfer1::Dims* inputs, int nbInputDims)
{
    // classifier_delta_bbox[N, anchors, num_classes*4, 1, 1]
    // classifier_class[N, anchors, num_classes, 1, 1]
    // rpn_rois[N, anchors, 4]
    PLUGIN_ASSERT(nbInputDims == 3);
    // delta_bbox
    PLUGIN_ASSERT(inputs[0].nbDims == 4 && inputs[0].d[1] == mNbClasses * 4);
    // score
    PLUGIN_ASSERT(inputs[1].nbDims == 4 && inputs[1].d[1] == mNbClasses);
    // roi
    PLUGIN_ASSERT(inputs[2].nbDims == 2 && inputs[2].d[1] == 4);
}

size_t DetectionLayer::getWorkspaceSize(int batch_size) const noexcept
{
    RefineDetectionWorkSpace refine(batch_size, mAnchorsCnt, mParam, mType);
    return refine.totalSize;
}

Dims DetectionLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{

    check_valid_inputs(inputs, nbInputDims);
    PLUGIN_ASSERT(index == 0);

    // [N, anchors, (y1, x1, y2, x2, class_id, score)]
    nvinfer1::Dims detections;

    detections.nbDims = 2;
    // number of anchors
    detections.d[0] = mKeepTopK;
    detections.d[1] = 6;

    return detections;
}

int DetectionLayer::enqueue(
    int batch_size, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        void* detections = outputs[0];

        // refine detection
        RefineDetectionWorkSpace refDetcWorkspace(batch_size, mAnchorsCnt, mParam, mType);
        cudaError_t status = RefineBatchClassNMS(stream, batch_size, mAnchorsCnt,
            DataType::kFLOAT, // mType,
            mParam, refDetcWorkspace, workspace,
            inputs[1],       // inputs[InScore]
            inputs[0],       // inputs[InDelta],
            mValidCnt->mPtr, // inputs[InCountValid],
            inputs[2],       // inputs[ROI]
            detections);

        return status;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return -1;
}

DataType DetectionLayer::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool DetectionLayer::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool DetectionLayer::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void DetectionLayer::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    check_valid_inputs(inputDims, nbInputs);
    PLUGIN_ASSERT(inputDims[0].d[0] == inputDims[1].d[0] && inputDims[1].d[0] == inputDims[2].d[0]);

    mAnchorsCnt = inputDims[2].d[0];
    mType = inputTypes[0];
    mMaxBatchSize = maxBatchSize;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void DetectionLayer::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void DetectionLayer::detachFromContext() noexcept {}
