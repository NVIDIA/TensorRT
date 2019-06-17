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
#include <vector>
#include "plugin.h"
#include "kernel.h"

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
        T_BBOX* topDetections)
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
            // clipped bbox xmin
            topDetections[i * 7 + 3] = max(min(bboxData[bboxId], T_BBOX(1.)), T_BBOX(0.));
            // clipped bbox ymin
            topDetections[i * 7 + 4] = max(min(bboxData[bboxId + 1], T_BBOX(1.)), T_BBOX(0.));
            // clipped bbox xmax
            topDetections[i * 7 + 5] = max(min(bboxData[bboxId + 2], T_BBOX(1.)), T_BBOX(0.));
            // clipped bbox ymax
            topDetections[i * 7 + 6] = max(min(bboxData[bboxId + 3], T_BBOX(1.)), T_BBOX(0.));
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
    void* topDetections)
{
    cudaMemsetAsync(keepCount, 0, numImages * sizeof(int), stream);
    const int BS = 32;
    const int GS = 32;
    gatherTopDetections_kernel<T_BBOX, T_SCORE, BS><<<GS, BS, 0, stream>>>(shareLocation, numImages, numPredsPerClass,
                                                                           numClasses, topK, keepTopK,
                                                                           (int*) indices, (T_SCORE*) scores, (T_BBOX*) bboxData,
                                                                           (int*) keepCount, (T_BBOX*) topDetections);

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
                               void*);
struct gtdLaunchConfig
{
    DataType t_bbox;
    DataType t_score;
    gtdFunc function;

    gtdLaunchConfig(DataType t_bbox, DataType t_score)
        : t_bbox(t_bbox)
        , t_score(t_score)
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

static std::vector<gtdLaunchConfig> gtdFuncVec;

bool gtdInit()
{
    gtdFuncVec.push_back(gtdLaunchConfig(DataType::kFLOAT, DataType::kFLOAT,
                                         gatherTopDetections_gpu<float, float>));
    return true;
}

static bool initialized = gtdInit();

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
    void* topDetections)
{
    gtdLaunchConfig lc = gtdLaunchConfig(DT_BBOX, DT_SCORE);
    for (unsigned i = 0; i < gtdFuncVec.size(); ++i)
    {
        if (lc == gtdFuncVec[i])
        {
            DEBUG_PRINTF("gatherTopDetections kernel %d\n", i);
            return gtdFuncVec[i].function(stream,
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
                                          topDetections);
        }
    }
    return STATUS_BAD_PARAM;
}
