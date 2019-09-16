/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef TRT_MASKRCNN_UTILS_H
#define TRT_MASKRCNN_UTILS_H

#include "NvInfer.h"
#include "plugin.h"

using namespace nvinfer1;

inline size_t nAlignUp(size_t x, size_t align)
{
    size_t mask = align - 1;
    assert((align & mask) == 0); // power of 2
    return (x + mask) & (~mask);
}

inline size_t nAlignDown(size_t x, size_t align)
{
    size_t mask = align - 1;
    assert((align & mask) == 0); // power of 2
    return (x) & (~mask);
}

inline size_t dimVolume(const nvinfer1::Dims& dims)
{
    size_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i)
        volume *= dims.d[i];

    return volume;
}

inline size_t typeSize(const nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: return sizeof(float);
    case nvinfer1::DataType::kHALF: return sizeof(uint16_t);
    case nvinfer1::DataType::kINT8: return sizeof(uint8_t);
    case nvinfer1::DataType::kINT32: return sizeof(uint32_t);
    default: return 0;
    }
}

#define AlignMem(x) nAlignUp(x, 256)

template <typename Dtype>
struct CudaBind
{
    size_t mSize;
    void* mPtr;

    CudaBind(size_t size)
    {
        mSize = size;
        CUASSERT(cudaMalloc(&mPtr, sizeof(Dtype) * mSize));
    }

    ~CudaBind()
    {
        if (mPtr != nullptr)
        {
            CUASSERT(cudaFree(mPtr));
            mPtr = nullptr;
        }
    }
};

struct RefineNMSParameters
{
    int backgroundLabelId, numClasses, keepTopK;
    float scoreThreshold, iouThreshold;
};

struct RefineDetectionWorkSpace
{
    RefineDetectionWorkSpace(
        const int batchSize, const int sampleCount, const RefineNMSParameters& param, const nvinfer1::DataType type);

    RefineDetectionWorkSpace() = default;

    nvinfer1::DimsHW argMaxScoreDims;
    nvinfer1::DimsHW argMaxBboxDims;
    nvinfer1::DimsHW argMaxLabelDims;
    nvinfer1::DimsHW sortClassScoreDims;
    nvinfer1::DimsHW sortClassLabelDims;
    nvinfer1::DimsHW sortClassSampleIdxDims;
    nvinfer1::Dims sortClassValidCountDims = {1, {1, 0}, {nvinfer1::DimensionType::kINDEX}};
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
    ProposalWorkSpace(const int batchSize, const int inputCnt, const int sampleCount, const RefineNMSParameters& param,
        const nvinfer1::DataType type);

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
    nvinfer1::Dims sortClassValidCountDims = {1, {1, 0}, {nvinfer1::DimensionType::kINDEX}};
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

cudaError_t RefineBatchClassNMS(cudaStream_t stream, int N, int samples, nvinfer1::DataType dtype,
    const RefineNMSParameters& param, const RefineDetectionWorkSpace& refineOffset, void* workspace,
    const void* inScores, const void* inDelta, const void* inCountValid, const void* inROI, void* outDetections);

cudaError_t proposalRefineBatchClassNMS(cudaStream_t stream, int N,
    int inputCnt, // candidate anchors
    int samples,  // preNMS_topK
    nvinfer1::DataType dtype, const RefineNMSParameters& param, const ProposalWorkSpace& proposalOffset,
    void* workspace, const void* inScores, const void* inDelta, const void* inCountValid, const void* inAnchors,
    void* outProposals);

cudaError_t ApplyDelta2Bboxes(cudaStream_t stream, int N,
    int samples,         // number of anchors per image
    const void* anchors, // [N, anchors, (y1, x1, y2, x2)]
    const void* delta,   //[N, anchors, (dy, dx, log(dh), log(dw)]
    void* outputBbox);

struct xy_t
{
    int y;
    int x;

    xy_t()
        : y(0)
        , x(0)
    {
    }
    xy_t(int y_, int x_)
        : y(y_)
        , x(x_)
    {
    }
};
// PYRAMID ROIALIGN
cudaError_t roiAlign(cudaStream_t stream, int batchSize, int featureCount, int roiCount, float firstThreshold,

    const void* rois, const void* const layers[], const xy_t* layerDims,

    void* pooled, const xy_t poolDims);

// RESIZE NEAREST
void resizeNearest(dim3 grid, dim3 block, cudaStream_t stream, int nbatch, float scale, int2 osize, float const* idata,
    int istride, int ibatchstride, float* odata, int ostride, int obatchstride);
// SPECIAL SLICE
void specialSlice(cudaStream_t stream, int batch_size, int boxes_cnt, const void* idata, void* odata);

#endif // TRT_MASKRCNN_UTILS_H
