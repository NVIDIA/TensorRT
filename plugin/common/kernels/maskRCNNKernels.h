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

#ifndef TRT_MASKRCNN_UTILS_H
#define TRT_MASKRCNN_UTILS_H

#include "NvInfer.h"
#include "common/plugin.h"

inline size_t nAlignUp(size_t x, size_t align)
{
    size_t mask = align - 1;
    PLUGIN_ASSERT((align & mask) == 0); // power of 2
    return (x + mask) & (~mask);
}

inline size_t nAlignDown(size_t x, size_t align)
{
    size_t mask = align - 1;
    PLUGIN_ASSERT((align & mask) == 0); // power of 2
    return (x) & (~mask);
}

inline size_t dimVolume(const nvinfer1::Dims& dims)
{
    size_t volume = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
        volume *= dims.d[i];

    return volume;
}

inline size_t typeSize(const nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: return sizeof(float);
    case nvinfer1::DataType::kBF16: return sizeof(uint16_t);
    case nvinfer1::DataType::kHALF: return sizeof(uint16_t);
    case nvinfer1::DataType::kINT8: return sizeof(uint8_t);
    case nvinfer1::DataType::kINT32: return sizeof(int32_t);
    case nvinfer1::DataType::kINT64: return sizeof(int64_t);
    case nvinfer1::DataType::kBOOL: return sizeof(bool);
    case nvinfer1::DataType::kUINT8: return sizeof(uint8_t);
    case nvinfer1::DataType::kFP8:
    case nvinfer1::DataType::kINT4:
    case nvinfer1::DataType::kFP4:
    case nvinfer1::DataType::kE8M0: PLUGIN_FAIL("Unsupported data type");
    }
    return 0;
}

#define AlignMem(x) nAlignUp(x, 256)

struct RefineNMSParameters
{
    int32_t backgroundLabelId, numClasses, keepTopK;
    float scoreThreshold, iouThreshold;
};

struct RefineDetectionWorkSpace
{
    RefineDetectionWorkSpace(const int32_t batchSize, const int32_t sampleCount, const RefineNMSParameters& param,
        const nvinfer1::DataType type);

    RefineDetectionWorkSpace() = default;

    nvinfer1::DimsHW argMaxScoreDims;
    nvinfer1::DimsHW argMaxBboxDims;
    nvinfer1::DimsHW argMaxLabelDims;
    nvinfer1::DimsHW sortClassScoreDims;
    nvinfer1::DimsHW sortClassLabelDims;
    nvinfer1::DimsHW sortClassSampleIdxDims;
    nvinfer1::Dims sortClassValidCountDims = {1, {1, 0}};
    nvinfer1::DimsHW sortClassPosDims;
    nvinfer1::DimsHW sortNMSMarkDims;

    size_t argMaxScoreOffset = 0;
    size_t argMaxBboxOffset = 0;
    size_t argMaxLabelOffset = 0;
    size_t sortClassScoreOffset = 0;
    size_t sortClassLabelOffset = 0;
    size_t sortClassSampleIdxOffset = 0;
    size_t sortClassValidCountOffset = 0;
    size_t sortClassPosOffset = 0;
    size_t sortNMSMarkOffset = 0;
    size_t totalSize = 0;
};

struct ProposalWorkSpace
{
    ProposalWorkSpace(const int32_t batchSize, const int32_t inputCnt, const int32_t sampleCount,
        const RefineNMSParameters& param, const nvinfer1::DataType type);

    ProposalWorkSpace() = default;

    nvinfer1::DimsHW preRefineScoreDims;
    nvinfer1::DimsHW preRefineSortedScoreDims;
    nvinfer1::DimsHW preRefineBboxDims;
    nvinfer1::DimsHW argMaxScoreDims;
    nvinfer1::DimsHW argMaxBboxDims;
    nvinfer1::DimsHW argMaxLabelDims;
    nvinfer1::DimsHW sortClassScoreDims;
    nvinfer1::DimsHW sortClassLabelDims;
    nvinfer1::DimsHW sortClassSampleIdxDims;
    nvinfer1::Dims sortClassValidCountDims = {1, {1, 0}};
    nvinfer1::DimsHW sortClassPosDims;
    nvinfer1::DimsHW sortNMSMarkDims;

    size_t tempStorageOffset = 0;
    size_t preRefineScoreOffset = 0;
    size_t preRefineSortedScoreOffset = 0;
    size_t preRefineBboxOffset = 0;
    size_t argMaxScoreOffset = 0;
    size_t argMaxBboxOffset = 0;
    size_t argMaxLabelOffset = 0;
    size_t sortClassScoreOffset = 0;
    size_t sortClassLabelOffset = 0;
    size_t sortClassSampleIdxOffset = 0;
    size_t sortClassValidCountOffset = 0;
    size_t sortClassPosOffset = 0;
    size_t sortNMSMarkOffset = 0;
    size_t totalSize = 0;
};

struct MultilevelProposeROIWorkSpace
{
    MultilevelProposeROIWorkSpace(const int32_t batchSize, const int32_t inputCnt, const int32_t sampleCount,
        const RefineNMSParameters& param, const nvinfer1::DataType type);

    MultilevelProposeROIWorkSpace() = default;

    nvinfer1::DimsHW preRefineSortedScoreDims;
    nvinfer1::DimsHW preRefineBboxDims;
    nvinfer1::DimsHW argMaxScoreDims;
    nvinfer1::DimsHW argMaxBboxDims;
    nvinfer1::DimsHW argMaxLabelDims;
    nvinfer1::DimsHW sortClassScoreDims;
    nvinfer1::DimsHW sortClassLabelDims;
    nvinfer1::DimsHW sortClassSampleIdxDims;
    nvinfer1::Dims sortClassValidCountDims = {1, {1, 0}};
    nvinfer1::DimsHW sortClassPosDims;
    nvinfer1::DimsHW sortNMSMarkDims;

    size_t tempStorageOffset = 0;
    size_t preRefineSortedScoreOffset = 0;
    size_t preRefineBboxOffset = 0;
    size_t argMaxScoreOffset = 0;
    size_t argMaxBboxOffset = 0;
    size_t argMaxLabelOffset = 0;
    size_t sortClassScoreOffset = 0;
    size_t sortClassLabelOffset = 0;
    size_t sortClassSampleIdxOffset = 0;
    size_t sortClassValidCountOffset = 0;
    size_t sortClassPosOffset = 0;
    size_t sortNMSMarkOffset = 0;
    size_t totalSize = 0;
};

struct ConcatTopKWorkSpace
{
    ConcatTopKWorkSpace(
        const int32_t batchSize, const int32_t concatCnt, const int32_t topK, const nvinfer1::DataType inType);

    ConcatTopKWorkSpace() = default;

    nvinfer1::DimsHW concatedScoreDims;
    nvinfer1::DimsHW concatedBBoxDims;
    nvinfer1::DimsHW sortedScoreDims;
    nvinfer1::DimsHW sortedBBoxDims;

    size_t tempStorageOffset = 0;
    size_t concatedScoreOffset = 0;
    size_t concatedBBoxOffset = 0;
    size_t sortedScoreOffset = 0;
    size_t sortedBBoxOffset = 0;
    size_t totalSize = 0;
};

cudaError_t RefineBatchClassNMS(cudaStream_t stream, int32_t N, int32_t samples, nvinfer1::DataType dtype,
    const RefineNMSParameters& param, const RefineDetectionWorkSpace& refineOffset, void* workspace,
    const void* inScores, const void* inDelta, const void* inCountValid, const void* inROI, void* outDetections);

cudaError_t DetectionPostProcess(cudaStream_t stream, int32_t N, int32_t samples, const float* regWeight,
    const float inputHeight, const float inputWidth, nvinfer1::DataType dtype, const RefineNMSParameters& param,
    const RefineDetectionWorkSpace& refineOffset, void* workspace, const void* inScores, const void* inDelta,
    const void* inCountValid, const void* inROI, void* outDetections);

cudaError_t proposalRefineBatchClassNMS(cudaStream_t stream, int32_t N,
    int32_t inputCnt, // candidate anchors
    int32_t samples,  // preNMS_topK
    nvinfer1::DataType dtype, const RefineNMSParameters& param, const ProposalWorkSpace& proposalOffset,
    void* workspace, const void* inScores, const void* inDelta, const void* inCountValid, const void* inAnchors,
    void* outProposals);

// inScores: [N, anchorsCnt, 1]
// inDelta: [N, anchorsCnt, 4]
// outScores: [N, topK, 1]
// outBbox: [N, topK, 4]
cudaError_t MultilevelPropose(cudaStream_t stream, int32_t N,
    int32_t inputCnt, // candidate anchors number among feature map
    int32_t samples,  // pre nms cnt
    const float* regWeight, const float inputHeight, const float inputWidth, nvinfer1::DataType dtype,
    const RefineNMSParameters& param, const MultilevelProposeROIWorkSpace& proposalOffset, void* workspace,
    const void* inScore, const void* inDelta, void* inCountValid, const void* inAnchors, void* outScores,
    void* outBbox);

// inScores: [N, topK, 1] * featureCnt
// inBboxes: [N, topK, 4] * featureCnt
// outProposals: [N, topK, 4]
cudaError_t ConcatTopK(cudaStream_t stream, int32_t N, int32_t featureCnt, int32_t topK, nvinfer1::DataType dtype,
    void* workspace, const ConcatTopKWorkSpace& spaceOffset, void** inScores, void** inBBox, void* outProposals);

cudaError_t DecodeBBoxes(cudaStream_t stream, int32_t N,
    int32_t samples, // number of anchors per image
    const float* regWeight, const float inputHeight, const float inputWidth,
    const void* anchors, // [N, anchors, (y1, x1, y2, x2)]
    const void* delta,   //[N, anchors, (dy, dx, log(dh), log(dw)]
    void* outputBbox, nvinfer1::DataType dtype);

cudaError_t ApplyDelta2Bboxes(cudaStream_t stream, int32_t N,
    int32_t samples,     // number of anchors per image
    const void* anchors, // [N, anchors, (y1, x1, y2, x2)]
    const void* delta,   //[N, anchors, (dy, dx, log(dh), log(dw)]
    void* outputBbox);

struct xy_t
{
    int32_t y;
    int32_t x;

    xy_t()
        : y(0)
        , x(0)
    {
    }
    xy_t(int32_t y_, int32_t x_)
        : y(y_)
        , x(x_)
    {
    }
};
// PYRAMID ROIALIGN
cudaError_t roiAlign(cudaStream_t const stream, int32_t const batchSize, xy_t const imageSize,
    int32_t const featureCount, int32_t const roiCount, float const firstThreshold, int32_t const transformCoords,
    bool const absCoords, bool const swapCoords, bool const plusOneCoords, int32_t const samplingRatio,
    void const* rois, void const* const layers[], xy_t const* layerDims, void* pooled, xy_t const poolDims);

cudaError_t roiAlignHalfCenter(cudaStream_t stream, int32_t batchSize, int32_t featureCount, int32_t roiCount,
    float firstThreshold,

    int32_t inputHeight, int32_t inputWidth, const void* rois, const void* const layers[], const xy_t* layerDims,

    void* pooled, const xy_t poolDims, const nvinfer1::DataType dtype);

// RESIZE NEAREST
void resizeNearest(dim3 grid, dim3 block, cudaStream_t stream, int32_t nbatch, float scale, int2 osize,
    float const* idata, int32_t istride, int32_t ibatchstride, float* odata, int32_t ostride, int32_t obatchstride);
// SPECIAL SLICE
void specialSlice(cudaStream_t stream, int32_t batch_size, int32_t boxes_cnt, const void* idata, void* odata);

#endif // TRT_MASKRCNN_UTILS_H
