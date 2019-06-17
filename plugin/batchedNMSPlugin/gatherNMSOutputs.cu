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
#include "kernel.h"
#include "plugin.h"
#include "gatherNMSOutputs.h"
#include <vector>

template <typename T_BBOX, typename T_SCORE, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta)
    __global__ void gatherNMSOutputs_kernel(
        const bool shareLocation,
        const int numImages,
        const int numPredsPerClass,
        const int numClasses,
        const int topK,
        const int keepTopK,
        const int* indices,
        const T_SCORE* scores,
        const T_BBOX* bboxData,
        int* numDetections,
        T_BBOX* nmsedBoxes,
        T_BBOX* nmsedScores,
        T_BBOX* nmsedClasses,
        bool clipBoxes
        )
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
        if (index == -1)
        {
            nmsedClasses[i] = -1;
            nmsedScores[i] = 0;
            nmsedBoxes[i * 4] = 0;
            nmsedBoxes[i * 4 + 1] = 0;
            nmsedBoxes[i * 4 + 2] = 0;
            nmsedBoxes[i * 4 + 3] = 0;
        }
        else
        {
            const int bboxOffset = imgId * (shareLocation ? numPredsPerClass : (numClasses * numPredsPerClass));
            const int bboxId = ((shareLocation ? (index % numPredsPerClass)
                        : index % (numClasses * numPredsPerClass)) + bboxOffset) * 4;
            nmsedClasses[i] = (index % (numClasses * numPredsPerClass)) / numPredsPerClass; // label
            nmsedScores[i] = score;                                                        // confidence score
            // clipped bbox xmin
            nmsedBoxes[i * 4] = clipBoxes ? max(min(bboxData[bboxId],
                        T_BBOX(1.)), T_BBOX(0.)) : bboxData[bboxId];
            // clipped bbox ymin
            nmsedBoxes[i * 4 + 1] = clipBoxes ? max(min(bboxData[bboxId + 1],
                        T_BBOX(1.)), T_BBOX(0.)) : bboxData[bboxId + 1];
            // clipped bbox xmax
            nmsedBoxes[i * 4 + 2] = clipBoxes ? max(min(bboxData[bboxId + 2],
                        T_BBOX(1.)), T_BBOX(0.)) : bboxData[bboxId + 2];
            // clipped bbox ymax
            nmsedBoxes[i * 4 + 3] = clipBoxes ? max(min(bboxData[bboxId + 3],
                        T_BBOX(1.)), T_BBOX(0.)) : bboxData[bboxId + 3];
            atomicAdd(&numDetections[i / keepTopK], 1);
        }
    }
}

template <typename T_BBOX, typename T_SCORE>
pluginStatus_t gatherNMSOutputs_gpu(
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
    void* numDetections,
    void* nmsedBoxes,
    void* nmsedScores,
    void* nmsedClasses,
    bool clipBoxes
    )
{
    cudaMemsetAsync(numDetections, 0, numImages * sizeof(int), stream);
    const int BS = 32;
    const int GS = 32;
    gatherNMSOutputs_kernel<T_BBOX, T_SCORE, BS><<<GS, BS, 0, stream>>>(shareLocation, numImages, numPredsPerClass,
                                                                           numClasses, topK, keepTopK,
                                                                           (int*) indices, (T_SCORE*) scores, (T_BBOX*) bboxData,
                                                                           (int*) numDetections,
                                                                           (T_BBOX*) nmsedBoxes, 
                                                                           (T_BBOX*) nmsedScores, 
                                                                           (T_BBOX*) nmsedClasses,
                                                                           clipBoxes
                                                                            );

    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// gatherNMSOutputs LAUNCH CONFIG {{{
typedef pluginStatus_t (*nmsOutFunc)(cudaStream_t,
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
                               void*, 
                               void*,
                               bool);
struct nmsOutLaunchConfig
{
    DataType t_bbox;
    DataType t_score;
    nmsOutFunc function;

    nmsOutLaunchConfig(DataType t_bbox, DataType t_score)
        : t_bbox(t_bbox)
        , t_score(t_score)
    {
    }
    nmsOutLaunchConfig(DataType t_bbox, DataType t_score, nmsOutFunc function)
        : t_bbox(t_bbox)
        , t_score(t_score)
        , function(function)
    {
    }
    bool operator==(const nmsOutLaunchConfig& other)
    {
        return t_bbox == other.t_bbox && t_score == other.t_score;
    }
};

using nvinfer1::DataType;

static std::vector<nmsOutLaunchConfig> nmsOutFuncVec;

bool nmsOutputInit()
{
    nmsOutFuncVec.push_back(nmsOutLaunchConfig(DataType::kFLOAT, DataType::kFLOAT,
                                         gatherNMSOutputs_gpu<float, float>));
    return true;
}

static bool initialized = nmsOutputInit();

//}}}

pluginStatus_t gatherNMSOutputs(
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
    void* numDetections,
    void* nmsedBoxes,
    void* nmsedScores,
    void* nmsedClasses,
    bool clipBoxes
    )
{
    nmsOutLaunchConfig lc = nmsOutLaunchConfig(DT_BBOX, DT_SCORE);
    for (unsigned i = 0; i < nmsOutFuncVec.size(); ++i)
    {
        if (lc == nmsOutFuncVec[i])
        {
            DEBUG_PRINTF("gatherNMSOutputs kernel %d\n", i);
            return nmsOutFuncVec[i].function(stream,
                                          shareLocation,
                                          numImages,
                                          numPredsPerClass,
                                          numClasses,
                                          topK,
                                          keepTopK,
                                          indices,
                                          scores,
                                          bboxData,
                                          numDetections,
                                          nmsedBoxes,
                                          nmsedScores,
                                          nmsedClasses,
                                          clipBoxes
                                          );
        }
    }
    return STATUS_BAD_PARAM;
}