/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "multilevelProposeROIPlugin.h"
#include "tlt_mrcnn_config.h"
#include "plugin.h"
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

MultilevelProposeROIPluginCreator::MultilevelProposeROIPluginCreator()
{

    mPluginAttributes.emplace_back(PluginField("prenms_topk", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("keep_topk", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("fg_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* MultilevelProposeROIPluginCreator::getPluginName() const
{
    return MULTILEVELPROPOSEROI_PLUGIN_NAME;
};

const char* MultilevelProposeROIPluginCreator::getPluginVersion() const
{
    return MULTILEVELPROPOSEROI_PLUGIN_VERSION;
};

const PluginFieldCollection* MultilevelProposeROIPluginCreator::getFieldNames()
{
    return &mFC;
};

IPluginV2Ext* MultilevelProposeROIPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
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
        if (!strcmp(attrName, "fg_threshold"))
        {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            mFGThreshold = *(static_cast<const float*>(fields[i].data));
        }
        if (!strcmp(attrName, "iou_threshold"))
        {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            mIOUThreshold = *(static_cast<const float*>(fields[i].data));
        }
    }
    return new MultilevelProposeROI(mPreNMSTopK, mKeepTopK, mFGThreshold, mIOUThreshold);
};

IPluginV2Ext* MultilevelProposeROIPluginCreator::deserializePlugin(const char* name, const void* data, size_t length)
{
    return new MultilevelProposeROI(data, length);
};

MultilevelProposeROI::MultilevelProposeROI(int prenms_topk, int keep_topk, float fg_threshold, float iou_threshold)
    : mPreNMSTopK(prenms_topk)
    , mKeepTopK(keep_topk)
    , mFGThreshold(fg_threshold)
    , mIOUThreshold(iou_threshold)
{
    mBackgroundLabel = -1;
    assert(mPreNMSTopK > 0);
    assert(mKeepTopK > 0);
    assert(mIOUThreshold >= 0.0f);
    assert(mFGThreshold >= 0.0f);

    mPreNMSTopK = 4096;
    mParam.backgroundLabelId = -1;
    mParam.numClasses = 1;
    mParam.keepTopK = mKeepTopK;
    mParam.scoreThreshold = mFGThreshold;
    mParam.iouThreshold = mIOUThreshold;

    mType = DataType::kFLOAT;

    mFeatureCnt = TLTMaskRCNNConfig::MAX_LEVEL - TLTMaskRCNNConfig::MIN_LEVEL + 1;

    generate_pyramid_anchors();
};

int MultilevelProposeROI::getNbOutputs() const
{
    return 1;
};

int MultilevelProposeROI::initialize()
{
    // Init the regWeight [1, 1, 1, 1]
    mRegWeightDevice = std::make_shared<CudaBind<float>>(4);
    std::vector<float> reg_weight(4, 1);
    CUASSERT(cudaMemcpy(static_cast<void*>(mRegWeightDevice->mPtr),
    static_cast<void*>(reg_weight.data()), sizeof(float) * 4, cudaMemcpyHostToDevice));

    // Init the mValidCnt of max batch size
    std::vector<int> tempValidCnt(mMaxBatchSize, mPreNMSTopK);

    mValidCnt = std::make_shared<CudaBind<int>>(mMaxBatchSize);

    CUASSERT(cudaMemcpy(
        mValidCnt->mPtr, static_cast<void*>(tempValidCnt.data()), sizeof(int) * mMaxBatchSize, cudaMemcpyHostToDevice));

    // Init the anchors for batch size:
    for(int i = 0; i < mFeatureCnt; i++)
    {
        int i_anchors_cnt = mAnchorsCnt[i];
        auto i_anchors_host = mAnchorBoxesHost[i].data();
        auto i_anchors_device = std::make_shared<CudaBind<float>>(i_anchors_cnt * 4 * mMaxBatchSize);
        int batch_offset = sizeof(float) * i_anchors_cnt * 4;
        uint8_t* device_ptr = static_cast<uint8_t*>(i_anchors_device->mPtr);
        for (int i = 0; i < mMaxBatchSize; i++)
        {
            CUASSERT(cudaMemcpy(static_cast<void*>(device_ptr + i * batch_offset),
                static_cast<void*>(i_anchors_host), batch_offset, cudaMemcpyHostToDevice));
        }
        mAnchorBoxesDevice.push_back(i_anchors_device);
    }

    // Init the temp storage for proposals from feature maps before concat
    std::vector<float*> score_tp;
    std::vector<float*> box_tp;
    for(int i = 0; i < mFeatureCnt; i++)
    {
        auto i_scores_device = std::make_shared<CudaBind<float>>(mKeepTopK * mMaxBatchSize);
        mTempScores.push_back(i_scores_device);
        score_tp.push_back(static_cast<float*>(i_scores_device->mPtr));

        auto i_bboxes_device = std::make_shared<CudaBind<float>>(mKeepTopK * 4 * mMaxBatchSize);
        mTempBboxes.push_back(i_bboxes_device);
        box_tp.push_back(static_cast<float*>(i_bboxes_device->mPtr));
    }

    // Init the temp storage for pointer arrays of score and box:
    CUASSERT(cudaMalloc(&mDeviceScores, sizeof(float*)*mFeatureCnt));
    CUASSERT(cudaMalloc(&mDeviceBboxes, sizeof(float*)*mFeatureCnt));

    CUASSERT(cudaMemcpy(mDeviceScores, score_tp.data(), sizeof(float*)*mFeatureCnt, cudaMemcpyHostToDevice));
    CUASSERT(cudaMemcpy(mDeviceBboxes, box_tp.data(), sizeof(float*)*mFeatureCnt, cudaMemcpyHostToDevice));

    return 0;
};

void MultilevelProposeROI::terminate(){};

void MultilevelProposeROI::destroy()
{
    delete this;
};

bool MultilevelProposeROI::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
};

const char* MultilevelProposeROI::getPluginType() const
{
    return "MultilevelProposeROI_TRT";
};

const char* MultilevelProposeROI::getPluginVersion() const
{
    return "1";
};

IPluginV2Ext* MultilevelProposeROI::clone() const
{
    return new MultilevelProposeROI(*this);
};

void MultilevelProposeROI::setPluginNamespace(const char* libNamespace)
{
    mNameSpace = libNamespace;
};

const char* MultilevelProposeROI::getPluginNamespace() const
{
    return mNameSpace.c_str();
};

size_t MultilevelProposeROI::getSerializationSize() const
{
    return sizeof(int) * 2 + sizeof(float) * 2 + sizeof(int) * (mFeatureCnt + 1);
};

void MultilevelProposeROI::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mPreNMSTopK);
    write(d, mKeepTopK);
    write(d, mFGThreshold);
    write(d, mIOUThreshold);
    write(d, mMaxBatchSize);
    for(int i = 0; i < mFeatureCnt; i++)
    {
        write(d, mAnchorsCnt[i]);
    }
    ASSERT(d == a + getSerializationSize());
};

MultilevelProposeROI::MultilevelProposeROI(const void* data, size_t length)
{
    mFeatureCnt = TLTMaskRCNNConfig::MAX_LEVEL - TLTMaskRCNNConfig::MIN_LEVEL + 1;

    const char *d = reinterpret_cast<const char*>(data), *a = d;
    int prenms_topk = read<int>(d);
    int keep_topk = read<int>(d);
    float fg_threshold = read<float>(d);
    float iou_threshold = read<float>(d);
    mMaxBatchSize = read<int>(d);
    assert(mAnchorsCnt.size() == 0);
    for(int i = 0; i < mFeatureCnt; i++)
    {
        mAnchorsCnt.push_back(read<int>(d));
    }
    ASSERT(d == a + length);

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

    mType = DataType::kFLOAT;

    generate_pyramid_anchors();
};

void MultilevelProposeROI::check_valid_inputs(const nvinfer1::Dims* inputs, int nbInputDims)
{
    // x=2,3,4,5,6
    // foreground_delta_px [N, h_x * w_x * anchors_per_location, 4, 1],
    // foreground_score_px [N, h_x * w_x * anchors_per_location, 1, 1],
    // anchors should be generated inside
    assert(nbInputDims == 2 * mFeatureCnt);
    for(int i = 0; i < 2 * mFeatureCnt; i += 2)
    {
        // foreground_delta
        assert(inputs[i].nbDims == 3 && inputs[i].d[1] == 4);
        // foreground_score
        assert(inputs[i+1].nbDims == 3 && inputs[i+1].d[1] == 1);
    }
};

size_t MultilevelProposeROI::getWorkspaceSize(int batch_size) const
{
    size_t total_size = 0; 
    assert(mAnchorsCnt.size() == mFeatureCnt);

    //workspace for propose on each feature map 
    for(int i = 0; i < mFeatureCnt; i++)
    {
        
        MultilevelProposeROIWorkSpace proposal(batch_size, mAnchorsCnt[i], mPreNMSTopK, mParam, mType);
        total_size += proposal.totalSize;
    }
    
    //workspace for Concat and TopK
    ConcatTopKWorkSpace ct(batch_size, mFeatureCnt, mKeepTopK, mType);
    total_size += ct.totalSize;

    return total_size;
};

Dims MultilevelProposeROI::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
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

void MultilevelProposeROI::generate_pyramid_anchors()
{
    const auto image_dims = TLTMaskRCNNConfig::IMAGE_SHAPE;

    const auto& anchor_scale = TLTMaskRCNNConfig::RPN_ANCHOR_SCALE;
    const auto& min_level = TLTMaskRCNNConfig::MIN_LEVEL;
    const auto& max_level = TLTMaskRCNNConfig::MAX_LEVEL; 
    const auto& aspect_ratios = TLTMaskRCNNConfig::ANCHOR_RATIOS;
    
    //Generate anchors strides and scales
    std::vector<float> anchor_scales;
    std::vector<int> anchor_strides;
    for(int i = min_level; i < max_level + 1; i++)
    {
        int stride = static_cast<int>(pow(2.0, i));
        anchor_strides.push_back(stride);
        anchor_scales.push_back(stride*anchor_scale);
    }

    auto& anchors = mAnchorBoxesHost;
    assert(anchors.size() == 0);

    assert(anchor_scales.size() == anchor_strides.size());
    for (size_t s = 0; s < anchor_scales.size(); ++s)
    {
        float scale = anchor_scales[s];
        int stride = anchor_strides[s];

        std::vector<float> s_anchors;
        for (int y = stride / 2 ; y < image_dims.d[1]; y += stride)
            for (int x = stride / 2; x < image_dims.d[2]; x += stride)
                for (auto r : aspect_ratios)
                {
                    float h = scale * r.second;
                    float w = scale * r.first;
                    
                    // Using y+h/2 instead of y+h/2-1 for alignment with TLT implementation
                    s_anchors.insert(s_anchors.end(),
                        {(y - h / 2), (x - w / 2), (y + h / 2 ), (x + w / 2 )});
                }

        anchors.push_back(s_anchors);
    }

    assert(anchors.size() == (max_level - min_level + 1));
}

int MultilevelProposeROI::enqueue(
    int batch_size, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{

    void* final_proposals = outputs[0];
    size_t kernel_workspace_offset = 0; 
    cudaError_t status;

    for(int i = 0; i < mFeatureCnt; i++)
    {

        MultilevelProposeROIWorkSpace proposal_ws(batch_size, mAnchorsCnt[i], mPreNMSTopK, mParam, mType);
        status = MultilevelPropose(stream, 
                    batch_size, 
                    mAnchorsCnt[i], 
                    mPreNMSTopK,
                    static_cast<float*>(mRegWeightDevice->mPtr), 
                    static_cast<float>(TLTMaskRCNNConfig::IMAGE_SHAPE.d[1]), //Input Height
                    static_cast<float>(TLTMaskRCNNConfig::IMAGE_SHAPE.d[2]),
                    DataType::kFLOAT, // mType,
                    mParam, 
                    proposal_ws, 
                    workspace + kernel_workspace_offset,
                    inputs[2*i + 1], // inputs[object_score],
                    inputs[2*i], // inputs[bbox_delta]
                    mValidCnt->mPtr,
                    mAnchorBoxesDevice[i]->mPtr, // inputs[anchors]
                    mTempScores[i]->mPtr, //temp scores [batch_size, topk, 1]
                    mTempBboxes[i]->mPtr); //temp
        assert(status == cudaSuccess);
        kernel_workspace_offset += proposal_ws.totalSize;
    }

    ConcatTopKWorkSpace ctopk_ws(batch_size, mFeatureCnt, mKeepTopK, mType);
    status = ConcatTopK(stream, 
                batch_size, 
                mFeatureCnt, 
                mKeepTopK, 
                DataType::kFLOAT,
                workspace + kernel_workspace_offset,
                ctopk_ws, 
                reinterpret_cast<void**>(mDeviceScores),
                reinterpret_cast<void**>(mDeviceBboxes), 
                final_proposals);

    assert(status == cudaSuccess);
    return status;
};

// Return the DataType of the plugin output at the requested index
DataType MultilevelProposeROI::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool MultilevelProposeROI::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool MultilevelProposeROI::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Configure the layer with input and output data types.
void MultilevelProposeROI::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    check_valid_inputs(inputDims, nbInputs);

    mAnchorsCnt.clear();
    for(int i = 0; i < mFeatureCnt; i++)
    {
        mAnchorsCnt.push_back(inputDims[2*i].d[0]);
        assert(mAnchorsCnt[i] == (int) (mAnchorBoxesHost[i].size() / 4));
    }

    mMaxBatchSize = maxBatchSize;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void MultilevelProposeROI::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void MultilevelProposeROI::detachFromContext() {}
