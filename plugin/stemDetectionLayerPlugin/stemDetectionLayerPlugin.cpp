/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include "stemDetectionLayerPlugin.h"
#include "plugin.h"
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::StemDetectionLayer;
using nvinfer1::plugin::StemDetectionLayerPluginCreator;

namespace
{
const char* STEMDETECTIONLAYER_PLUGIN_VERSION{"1"};
const char* STEMDETECTIONLAYER_PLUGIN_NAME{"StemDetectionLayer_TRT"};
} // namespace

PluginFieldCollection StemDetectionLayerPluginCreator::mFC{};
std::vector<PluginField> StemDetectionLayerPluginCreator::mPluginAttributes;

StemDetectionLayerPluginCreator::StemDetectionLayerPluginCreator()
{

    mPluginAttributes.emplace_back(PluginField("num_classes", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("keep_topk", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* StemDetectionLayerPluginCreator::getPluginName() const noexcept
{
    return STEMDETECTIONLAYER_PLUGIN_NAME;
}

const char* StemDetectionLayerPluginCreator::getPluginVersion() const noexcept
{
    return STEMDETECTIONLAYER_PLUGIN_VERSION;
}

const PluginFieldCollection* StemDetectionLayerPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* StemDetectionLayerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "num_classes"))
            {
                assert(fields[i].type == PluginFieldType::kINT32);
                mNbClasses = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "keep_topk"))
            {
                assert(fields[i].type == PluginFieldType::kINT32);
                mKeepTopK = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "score_threshold"))
            {
                assert(fields[i].type == PluginFieldType::kFLOAT32);
                mScoreThreshold = *(static_cast<const float*>(fields[i].data));
            }
            if (!strcmp(attrName, "iou_threshold"))
            {
                assert(fields[i].type == PluginFieldType::kFLOAT32);
                mIOUThreshold = *(static_cast<const float*>(fields[i].data));
            }
        }
        return new StemDetectionLayer(mNbClasses, mKeepTopK, mScoreThreshold, mIOUThreshold);
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* StemDetectionLayerPluginCreator::deserializePlugin(const char* name, const void* data, size_t length) noexcept
{
    try
    {
        return new StemDetectionLayer(data, length);
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

StemDetectionLayer::StemDetectionLayer(int num_classes, int keep_topk, float score_threshold, float iou_threshold)
    : mNbClasses(num_classes)
    , mKeepTopK(keep_topk)
    , mScoreThreshold(score_threshold)
    , mIOUThreshold(iou_threshold)
{
    mBackgroundLabel = 0;
    assert(mNbClasses > 0);
    assert(mKeepTopK > 0);
    assert(score_threshold >= 0.0f);
    assert(iou_threshold > 0.0f);

    mParam.backgroundLabelId = 0;
    mParam.numClasses = mNbClasses;
    mParam.keepTopK = mKeepTopK;
    mParam.scoreThreshold = mScoreThreshold;
    mParam.iouThreshold = mIOUThreshold;

    mType = DataType::kFLOAT;
}

int StemDetectionLayer::getNbOutputs() const noexcept
{
    return 1;
}

int StemDetectionLayer::initialize() noexcept
{
    //@Init the mValidCnt and mDecodedBboxes for max batch size
    std::vector<int> tempValidCnt(mMaxBatchSize, mAnchorsCnt);

    mValidCnt = std::make_shared<CudaBind<int>>(mMaxBatchSize);

    CUASSERT(cudaMemcpy(
        mValidCnt->mPtr, static_cast<void*>(tempValidCnt.data()), sizeof(int) * mMaxBatchSize, cudaMemcpyHostToDevice));

    return 0;
}

void StemDetectionLayer::terminate() noexcept {}

void StemDetectionLayer::destroy() noexcept
{
    delete this;
}

bool StemDetectionLayer::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

const char* StemDetectionLayer::getPluginType() const noexcept
{
    return "StemDetectionLayer_TRT";
}

const char* StemDetectionLayer::getPluginVersion() const noexcept
{
    return "1";
}

IPluginV2Ext* StemDetectionLayer::clone() const noexcept
{
    try
    {
        StemDetectionLayer* plugin = new StemDetectionLayer(*this);
        plugin->setPluginNamespace(mNameSpace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void StemDetectionLayer::setPluginNamespace(const char* libNamespace) noexcept
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

const char* StemDetectionLayer::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

size_t StemDetectionLayer::getSerializationSize() const noexcept
{
    return sizeof(int) * 2 + sizeof(float) * 2 + sizeof(int) * 2;
}

void StemDetectionLayer::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mNbClasses);
    write(d, mKeepTopK);
    write(d, mScoreThreshold);
    write(d, mIOUThreshold);
    write(d, mMaxBatchSize);
    write(d, mAnchorsCnt);
    ASSERT(d == a + getSerializationSize());
}

StemDetectionLayer::StemDetectionLayer(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    int num_classes = read<int>(d);
    int keep_topk = read<int>(d);
    float score_threshold = read<float>(d);
    float iou_threshold = read<float>(d);
    mMaxBatchSize = read<int>(d);
    mAnchorsCnt = read<int>(d);
    ASSERT(d == a + length);

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

void StemDetectionLayer::check_valid_inputs(const nvinfer1::Dims* inputs, int nbInputDims)
{
    // classifier_delta_bbox[N, anchors, num_classes*4, 1, 1]
    // classifier_class[N, anchors, num_classes, 1, 1]
    // rpn_rois[N, anchors, 4]
    // stem_fc[N, anchors, num_classes*2]
    assert(nbInputDims == 4);
    // delta_bbox
    assert(inputs[0].nbDims == 4 && inputs[0].d[1] == mNbClasses * 4);
    // score
    assert(inputs[1].nbDims == 4 && inputs[1].d[1] == mNbClasses);
    // roi
    assert(inputs[2].nbDims == 2 && inputs[2].d[1] == 4);
    // stems
    assert(inputs[3].nbDims == 2 && inputs[2].d[1] == mNbClasses * 2);
}

size_t StemDetectionLayer::getWorkspaceSize(int batch_size) const noexcept
{
    RefineStemDetectionWorkSpace refine(batch_size, mAnchorsCnt, mParam, mType);
    return refine.totalSize;
}

Dims StemDetectionLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{

    check_valid_inputs(inputs, nbInputDims);
    assert(index == 0);

    // [N, anchors, (y1, x1, y2, x2, class_id, score, stem_y, stem_x)]
    nvinfer1::Dims detections;

    detections.nbDims = 2;
    // number of anchors
    detections.d[0] = mKeepTopK;
    detections.d[1] = 8;

    return detections;
}

int StemDetectionLayer::enqueue(
    int batch_size, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        void* detections = outputs[0];

        // refine detection
        RefineStemDetectionWorkSpace refDetcWorkspace(batch_size, mAnchorsCnt, mParam, mType);
        cudaError_t status = RefineStemBatchClassNMS(stream, batch_size, mAnchorsCnt,
            DataType::kFLOAT, // mType,
            mParam, refDetcWorkspace, workspace,
            inputs[1],       // inputs[InScore]
            inputs[0],       // inputs[InDelta],
            mValidCnt->mPtr, // inputs[InCountValid],
            inputs[2],       // inputs[ROI]
            inputs[3],       // inputs[InStemsDelta]
            detections);

        return status;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return -1;
}

DataType StemDetectionLayer::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool StemDetectionLayer::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool StemDetectionLayer::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void StemDetectionLayer::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    check_valid_inputs(inputDims, nbInputs);
    assert(inputDims[0].d[0] == inputDims[1].d[0] && inputDims[1].d[0] == inputDims[2].d[0] && inputDims[1].d[0] == inputDims[3].d[0]);

    mAnchorsCnt = inputDims[2].d[0];
    mType = inputTypes[0];
    mMaxBatchSize = maxBatchSize;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void StemDetectionLayer::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void StemDetectionLayer::detachFromContext() noexcept {}
