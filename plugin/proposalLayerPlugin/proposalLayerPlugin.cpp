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
#include "proposalLayerPlugin.h"
#include "mrcnn_config.h"
#include "plugin.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <iostream>
#include <math.h>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::ProposalLayer;
using nvinfer1::plugin::ProposalLayerPluginCreator;

namespace
{
const char* PROPOSALLAYER_PLUGIN_VERSION{"1"};
const char* PROPOSALLAYER_PLUGIN_NAME{"ProposalLayer_TRT"};
} // namespace

PluginFieldCollection ProposalLayerPluginCreator::mFC{};
std::vector<PluginField> ProposalLayerPluginCreator::mPluginAttributes;

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

const char* ProposalLayerPluginCreator::getPluginName() const noexcept
{
    return PROPOSALLAYER_PLUGIN_NAME;
}

const char* ProposalLayerPluginCreator::getPluginVersion() const noexcept
{
    return PROPOSALLAYER_PLUGIN_VERSION;
}

const PluginFieldCollection* ProposalLayerPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* ProposalLayerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    auto image_size = MaskRCNNConfig::IMAGE_SHAPE;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "prenms_topk"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            mPreNMSTopK = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "keep_topk"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            mKeepTopK = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "iou_threshold"))
        {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            mIOUThreshold = *(static_cast<const float*>(fields[i].data));
        }
        if (!strcmp(attrName, "image_size"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            const auto* const dims = static_cast<const int32_t*>(fields[i].data);
            std::copy_n(dims, 3, image_size.d);
        }
    }
    return new ProposalLayer(mPreNMSTopK, mKeepTopK, mIOUThreshold, image_size);
}

IPluginV2Ext* ProposalLayerPluginCreator::deserializePlugin(const char* name, const void* data, size_t length) noexcept
{
    return new ProposalLayer(data, length);
}

ProposalLayer::ProposalLayer(int prenms_topk, int keep_topk, float iou_threshold, const nvinfer1::Dims& image_size)
    : mPreNMSTopK(prenms_topk)
    , mKeepTopK(keep_topk)
    , mIOUThreshold(iou_threshold)
    , mImageSize(image_size)
{
    mBackgroundLabel = -1;
    assert(mPreNMSTopK > 0);
    assert(mKeepTopK > 0);
    assert(iou_threshold > 0.0F);

    mParam.backgroundLabelId = -1;
    mParam.numClasses = 1;
    mParam.keepTopK = mKeepTopK;
    mParam.scoreThreshold = 0.0;
    mParam.iouThreshold = mIOUThreshold;

    mType = DataType::kFLOAT;

    generate_pyramid_anchors(image_size);
}

int ProposalLayer::getNbOutputs() const noexcept
{
    return 1;
}

int ProposalLayer::initialize() noexcept
{
    // Init the mValidCnt of max batch size
    std::vector<int> tempValidCnt(mMaxBatchSize, mPreNMSTopK);

    mValidCnt = std::make_shared<CudaBind<int>>(mMaxBatchSize);

    CUASSERT(cudaMemcpy(
        mValidCnt->mPtr, static_cast<void*>(tempValidCnt.data()), sizeof(int) * mMaxBatchSize, cudaMemcpyHostToDevice));

    // Init the anchors for batch size:
    mAnchorBoxesDevice = std::make_shared<CudaBind<float>>(mAnchorsCnt * 4 * mMaxBatchSize);
    int batch_offset = sizeof(float) * mAnchorsCnt * 4;
    uint8_t* device_ptr = static_cast<uint8_t*>(mAnchorBoxesDevice->mPtr);
    for (int i = 0; i < mMaxBatchSize; i++)
    {
        CUASSERT(cudaMemcpy(static_cast<void*>(device_ptr + i * batch_offset),
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

const char* ProposalLayer::getPluginType() const noexcept
{
    return PROPOSALLAYER_PLUGIN_NAME;
}

const char* ProposalLayer::getPluginVersion() const noexcept
{
    return PROPOSALLAYER_PLUGIN_VERSION;
}

IPluginV2Ext* ProposalLayer::clone() const noexcept
{
    auto* plugin = new ProposalLayer(*this);
    plugin->setPluginNamespace(mNameSpace.c_str());
    return plugin;
}

void ProposalLayer::setPluginNamespace(const char* libNamespace) noexcept
{
    mNameSpace = libNamespace;
}

const char* ProposalLayer::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

size_t ProposalLayer::getSerializationSize() const noexcept
{
    return sizeof(int) * 2 + sizeof(float) + sizeof(int) * 2 + sizeof(nvinfer1::Dims);
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
    ASSERT(d == a + getSerializationSize());
}

ProposalLayer::ProposalLayer(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    int prenms_topk = read<int>(d);
    int keep_topk = read<int>(d);
    float iou_threshold = read<float>(d);
    mMaxBatchSize = read<int>(d);
    mAnchorsCnt = read<int>(d);
    mImageSize = read<nvinfer1::Dims3>(d);
    ASSERT(d == a + length);

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

void ProposalLayer::check_valid_inputs(const nvinfer1::Dims* inputs, int nbInputDims)
{
    // object_score[N, anchors, 2, 1],
    // foreground_delta[N, anchors, 4, 1],
    // anchors should be generated inside
    assert(nbInputDims == 2);
    // foreground_score
    assert(inputs[0].nbDims == 3 && inputs[0].d[1] == 2);
    // foreground_delta
    assert(inputs[1].nbDims == 3 && inputs[1].d[1] == 4);
}

size_t ProposalLayer::getWorkspaceSize(int batch_size) const noexcept
{

    ProposalWorkSpace proposal(batch_size, mAnchorsCnt, mPreNMSTopK, mParam, mType);
    return proposal.totalSize;
}

Dims ProposalLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{

    check_valid_inputs(inputs, nbInputDims);
    assert(index == 0);

    // [N, anchors, (y1, x1, y2, x2)]
    nvinfer1::Dims proposals;

    proposals.nbDims = 2;
    // number of keeping anchors
    proposals.d[0] = mKeepTopK;
    proposals.d[1] = 4;

    return proposals;
}

void ProposalLayer::generate_pyramid_anchors(const nvinfer1::Dims& image_dims) noexcept
{
    assert(image_dims.nbDims == 3 && image_dims.d[0] == 3);

    const auto& scales = MaskRCNNConfig::RPN_ANCHOR_SCALES;
    const auto& ratios = MaskRCNNConfig::RPN_ANCHOR_RATIOS;
    const auto& strides = MaskRCNNConfig::BACKBONE_STRIDES;
    auto anchor_stride = MaskRCNNConfig::RPN_ANCHOR_STRIDE;

    const float cy = image_dims.d[1] - 1;
    const float cx = image_dims.d[2] - 1;

    auto& anchors = mAnchorBoxesHost;
    assert(anchors.empty());

    assert(scales.size() == strides.size());
    for (size_t s = 0; s < scales.size(); ++s)
    {
        float scale = scales[s];
        int stride = strides[s];

        for (int y = 0; y < image_dims.d[1]; y += anchor_stride * stride)
            for (int x = 0; x < image_dims.d[2]; x += anchor_stride * stride)
                for (float r : ratios)
                {
                    float sqrt_r = sqrt(r);
                    float h = scale / sqrt_r;
                    float w = scale * sqrt_r;

                    anchors.insert(anchors.end(),
                        {(y - h / 2) / cy, (x - w / 2) / cx, (y + h / 2 - 1) / cy, (x + w / 2 - 1) / cx});
                }
    }

    assert(anchors.size() % 4 == 0);
}

int ProposalLayer::enqueue(
    int batch_size, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
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

    assert(status == cudaSuccess);
    return status;
}

// Return the DataType of the plugin output at the requested index
DataType ProposalLayer::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool ProposalLayer::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool ProposalLayer::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void ProposalLayer::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    check_valid_inputs(inputDims, nbInputs);
    assert(inputDims[0].d[0] == inputDims[1].d[0]);

    mAnchorsCnt = inputDims[0].d[0];
    assert(mAnchorsCnt == (int) (mAnchorBoxesHost.size() / 4));
    mMaxBatchSize = maxBatchSize;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void ProposalLayer::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void ProposalLayer::detachFromContext() noexcept {}
