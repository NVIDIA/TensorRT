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
#include "cub/cub.cuh"
#include <vector>
#include "kernel.h"
#include "bboxUtils.h"
#include "cub_helper.h"


template <typename T_SCORE, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta)
    __global__ void prepareSortData(
        const int num,
        const int num_classes,
        const int num_preds_per_class,
        const int background_label_id,
        const float confidence_threshold,
        T_SCORE* conf_scores_gpu,
        T_SCORE* temp_scores,
        int* temp_idx,
        int* d_offsets)
{
    // Prepare scores data for sort
    const int cur_idx = blockIdx.x * nthds_per_cta + threadIdx.x;
    const int numPredsPerBatch = num_classes * num_preds_per_class;
    if (cur_idx < numPredsPerBatch)
    {
        const int class_idx = cur_idx / num_preds_per_class;
        for (int i = 0; i < num; i++)
        {
            const int targetIdx = i * numPredsPerBatch + cur_idx;
            const T_SCORE score = conf_scores_gpu[targetIdx];

            // "Clear" background labeled score and index
            // Because we do not care about background
            if (class_idx == background_label_id)
            {
                // Set scores to 0
                // Set label = -1
                temp_scores[targetIdx] = 0.0f;
                temp_idx[targetIdx] = -1;
                conf_scores_gpu[targetIdx] = 0.0f;
            }
            // "Clear" scores lower than threshold
            else
            {
                if (score > confidence_threshold)
                {
                    temp_scores[targetIdx] = score;
                    temp_idx[targetIdx] = cur_idx + i * numPredsPerBatch;
                }
                else
                {
                    // Set scores to 0
                    // Set label = -1
                    temp_scores[targetIdx] = 0.0f;
                    temp_idx[targetIdx] = -1;
                    conf_scores_gpu[targetIdx] = 0.0f;
                    // TODO: HERE writing memory too many times
                }
            }

            if ((cur_idx % num_preds_per_class) == 0)
            {
                const int offset_ct = i * num_classes + cur_idx / num_preds_per_class;
                d_offsets[offset_ct] = offset_ct * num_preds_per_class;
                // set the last element in d_offset
                if (blockIdx.x == 0 && threadIdx.x == 0)
                    d_offsets[num * num_classes] = num * numPredsPerBatch;
            }
        }
    }
}

template <typename T_SCORE>
pluginStatus_t sortScoresPerClass_gpu(
    cudaStream_t stream,
    const int num,
    const int num_classes,
    const int num_preds_per_class,
    const int background_label_id,
    const float confidence_threshold,
    void* conf_scores_gpu,
    void* index_array_gpu,
    void* workspace)
{
    const int num_segments = num * num_classes;
    void* temp_scores = workspace;
    const int arrayLen = num * num_classes * num_preds_per_class;
    void* temp_idx = nextWorkspacePtr((int8_t*) temp_scores, arrayLen * sizeof(T_SCORE));
    void* d_offsets = nextWorkspacePtr((int8_t*) temp_idx, arrayLen * sizeof(int));
    size_t cubOffsetSize = (num_segments + 1) * sizeof(int);
    void* cubWorkspace = nextWorkspacePtr((int8_t*) d_offsets, cubOffsetSize);

    const int BS = 512;
    const int GS = (num_classes * num_preds_per_class + BS - 1) / BS;
    prepareSortData<T_SCORE, BS><<<GS, BS, 0, stream>>>(num, num_classes, num_preds_per_class,
                                                        background_label_id, confidence_threshold,
                                                        (T_SCORE*) conf_scores_gpu,
                                                        (T_SCORE*) temp_scores,
                                                        (int*) temp_idx,
                                                        (int*) d_offsets);

    size_t temp_storage_bytes = cubSortPairsWorkspaceSize<T_SCORE, int>(arrayLen, num_segments);
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        cubWorkspace, temp_storage_bytes,
        (const T_SCORE*) (temp_scores), (T_SCORE*) (conf_scores_gpu),
        (const int*) (temp_idx), (int*) (index_array_gpu),
        arrayLen, num_segments,
        (const int*) d_offsets, (const int*) d_offsets + 1,
        0, sizeof(T_SCORE) * 8,
        stream);
    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// sortScoresPerClass LAUNCH CONFIG 
typedef pluginStatus_t (*sspcFunc)(cudaStream_t,
                                const int,
                                const int,
                                const int,
                                const int,
                                const float,
                                void*,
                                void*,
                                void*);
struct sspcLaunchConfig
{
    DataType t_score;
    sspcFunc function;

    sspcLaunchConfig(DataType t_score)
        : t_score(t_score)
    {
    }
    sspcLaunchConfig(DataType t_score, sspcFunc function)
        : t_score(t_score)
        , function(function)
    {
    }
    bool operator==(const sspcLaunchConfig& other)
    {
        return t_score == other.t_score;
    }
};

static std::vector<sspcLaunchConfig> sspcFuncVec;
bool sspcInit()
{
    sspcFuncVec.push_back(sspcLaunchConfig(DataType::kFLOAT,
                                           sortScoresPerClass_gpu<float>));
    return true;
}

static bool initialized = sspcInit();

pluginStatus_t sortScoresPerClass(
    cudaStream_t stream,
    const int num,
    const int num_classes,
    const int num_preds_per_class,
    const int background_label_id,
    const float confidence_threshold,
    const DataType DT_SCORE,
    void* conf_scores_gpu,
    void* index_array_gpu,
    void* workspace)
{
    sspcLaunchConfig lc = sspcLaunchConfig(DT_SCORE);
    for (unsigned i = 0; i < sspcFuncVec.size(); ++i)
    {
        if (lc == sspcFuncVec[i])
        {
            DEBUG_PRINTF("sortScoresPerClass kernel %d\n", i);
            return sspcFuncVec[i].function(stream,
                                           num,
                                           num_classes,
                                           num_preds_per_class,
                                           background_label_id,
                                           confidence_threshold,
                                           conf_scores_gpu,
                                           index_array_gpu,
                                           workspace);
        }
    }
    return STATUS_BAD_PARAM;
}

size_t sortScoresPerClassWorkspaceSize(
    const int num,
    const int num_classes,
    const int num_preds_per_class,
    const DataType DT_CONF)
{
    size_t wss[4];
    const int arrayLen = num * num_classes * num_preds_per_class;
    wss[0] = arrayLen * dataTypeSize(DT_CONF);      // temp scores
    wss[1] = arrayLen * sizeof(int);                // temp indices
    wss[2] = (num * num_classes + 1) * sizeof(int); // offsets
    if (DT_CONF == DataType::kFLOAT)
    {
        wss[3] = cubSortPairsWorkspaceSize<float, int>(arrayLen, num * num_classes); // cub workspace
    }
    else
    {
        printf("SCORE type not supported\n");
        return (size_t) -1;
    }

    return calculateTotalWorkspaceSize(wss, 4);
}
