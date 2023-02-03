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
#include "common/kernel.h"
#include "common/plugin.h"
#include "cuda_fp16.h"
#include <array>

using namespace nvinfer1;

inline __device__ __half minus_fb(const __half& a, const __half& b)
{
#if __CUDA_ARCH__ >= 530
    return a - b;
#else
    return __float2half(__half2float(a) - __half2float(b));
#endif
}

inline __device__ float minus_fb(const float & a, const float & b) {
    return a - b;
}

template <typename T_BBOX, typename T_SCORE, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta)
    __global__ void gatherTopDetections_kernel(
        const bool shareLocation,
        const int numImages,
        const int numPredsPerClass,
        const int numClasses,
        const int topK,
        const int keepTopK,
        const int* indices,
        const T_SCORE* scores,
        const T_BBOX* bboxData,
        int* keepCount,
        T_BBOX* topDetections,
        const T_SCORE score_shift)
{
    if (keepTopK > topK)
        return;
    for (int i = blockIdx.x * nthds_per_cta + threadIdx.x;
         i < numImages * keepTopK;
         i += gridDim.x * nthds_per_cta)
    {
        const int imgId = i / keepTopK;
        const int detId = i % keepTopK;
        const int offset = imgId * numClasses * topK;
        const int index = indices[offset + detId];
        const T_SCORE score = scores[offset + detId];
        /*
         * It is also likely that there is "bad bounding boxes" in the keepTopK bounding boxes.
         * We set the bounding boxes parameters as the parameters shown below.
         * These data will only show up at the end of keepTopK bounding boxes since the bounding boxes were sorted previously.
         * It is also not going to affect the count of valid bounding boxes (keepCount).
         * These data will probably never be used (because we have keepCount).
         */
        if (index == -1)
        {
            topDetections[i * 7] = imgId;  // image id
            topDetections[i * 7 + 1] = -1; // label
            topDetections[i * 7 + 2] = 0;  // confidence score
            // score==0 will not pass the VisualizeBBox check
            topDetections[i * 7 + 3] = 0;   // bbox xmin
            topDetections[i * 7 + 4] = 0;   // bbox ymin
            topDetections[i * 7 + 5] = 0;   // bbox xmax
            topDetections[i * 7 + 6] = 0;   // bbox ymax
        }
        else
        {
            const int bboxOffset = imgId * (shareLocation ? numPredsPerClass : (numClasses * numPredsPerClass));
            const int bboxId = ((shareLocation ? (index % numPredsPerClass)
                        : index % (numClasses * numPredsPerClass)) + bboxOffset) * 4;
            topDetections[i * 7] = imgId;                                                            // image id
            topDetections[i * 7 + 1] = (index % (numClasses * numPredsPerClass)) / numPredsPerClass; // label
            topDetections[i * 7 + 2] = score;                                                        // confidence score
            // subtract 1.0 score shift we added in sortScorePerClass
            topDetections[i * 7 + 2] = minus_fb(topDetections[i * 7 + 2], score_shift);
            const T_BBOX xMin = bboxData[bboxId];
            const T_BBOX yMin = bboxData[bboxId + 1];
            const T_BBOX xMax = bboxData[bboxId + 2];
            const T_BBOX yMax = bboxData[bboxId + 3];
            // clipped bbox xmin
            topDetections[i * 7 + 3] = __saturatef(xMin);
            // clipped bbox ymin
            topDetections[i * 7 + 4] = __saturatef(yMin);
            // clipped bbox xmax
            topDetections[i * 7 + 5] = __saturatef(xMax);
            // clipped bbox ymax
            topDetections[i * 7 + 6] = __saturatef(yMax);
            // Atomic add to increase the count of valid keepTopK bounding boxes
            // Without having to do manual sync.
            atomicAdd(&keepCount[i / keepTopK], 1);
        }
    }
}

template <typename T_BBOX, typename T_SCORE>
pluginStatus_t gatherTopDetections_gpu(
    cudaStream_t stream,
    const bool shareLocation,
    const int numImages,
    const int numPredsPerClass,
    const int numClasses,
    const int topK,
    const int keepTopK,
    const void* indices,
    const void* scores,
    const void* bboxData,
    void* keepCount,
    void* topDetections,
    const float score_shift
)
{
    CSC(cudaMemsetAsync(keepCount, 0, numImages * sizeof(int), stream), STATUS_FAILURE);
    const int BS = 32;
    const int GS = 32;
    gatherTopDetections_kernel<T_BBOX, T_SCORE, BS><<<GS, BS, 0, stream>>>(shareLocation, numImages, numPredsPerClass,
                                                                           numClasses, topK, keepTopK,
                                                                           (int*) indices, (T_SCORE*) scores, (T_BBOX*) bboxData,
                                                                           (int*) keepCount, (T_BBOX*) topDetections,
                                                                           T_SCORE(score_shift));

    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// gatherTopDetections LAUNCH CONFIG
typedef pluginStatus_t (*gtdFunc)(cudaStream_t,
                               const bool,
                               const int,
                               const int,
                               const int,
                               const int,
                               const int,
                               const void*,
                               const void*,
                               const void*,
                               void*,
                               void*,
                               const float);
struct gtdLaunchConfig
{
    DataType t_bbox;
    DataType t_score;
    gtdFunc function;

    gtdLaunchConfig(DataType t_bbox, DataType t_score)
        : t_bbox(t_bbox)
        , t_score(t_score)
        , function(nullptr)
    {
    }
    gtdLaunchConfig(DataType t_bbox, DataType t_score, gtdFunc function)
        : t_bbox(t_bbox)
        , t_score(t_score)
        , function(function)
    {
    }
    bool operator==(const gtdLaunchConfig& other)
    {
        return t_bbox == other.t_bbox && t_score == other.t_score;
    }
};

using nvinfer1::DataType;

static std::array<gtdLaunchConfig, 2> gtdLCOptions = {
    gtdLaunchConfig(DataType::kFLOAT, DataType::kFLOAT, gatherTopDetections_gpu<float, float>),
    gtdLaunchConfig(DataType::kHALF, DataType::kHALF, gatherTopDetections_gpu<__half, __half>)
};

pluginStatus_t gatherTopDetections(
    cudaStream_t stream,
    const bool shareLocation,
    const int numImages,
    const int numPredsPerClass,
    const int numClasses,
    const int topK,
    const int keepTopK,
    const DataType DT_BBOX,
    const DataType DT_SCORE,
    const void* indices,
    const void* scores,
    const void* bboxData,
    void* keepCount,
    void* topDetections,
    const float score_shift)
{
    gtdLaunchConfig lc = gtdLaunchConfig(DT_BBOX, DT_SCORE);
    for (unsigned i = 0; i < gtdLCOptions.size(); ++i)
    {
        if (lc == gtdLCOptions[i])
        {
            DEBUG_PRINTF("gatherTopDetections kernel %d\n", i);
            return gtdLCOptions[i].function(stream,
                                          shareLocation,
                                          numImages,
                                          numPredsPerClass,
                                          numClasses,
                                          topK,
                                          keepTopK,
                                          indices,
                                          scores,
                                          bboxData,
                                          keepCount,
                                          topDetections,
                                          score_shift);
        }
    }
    return STATUS_BAD_PARAM;
}
