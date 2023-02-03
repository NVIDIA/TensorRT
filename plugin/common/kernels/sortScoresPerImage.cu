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
#include "common/bboxUtils.h"
#include "common/cub_helper.h"
#include "common/kernel.h"
#include "cub/cub.cuh"
#include <array>

using namespace nvinfer1;
template <typename T_SCORE>
pluginStatus_t sortScoresPerImage_gpu(cudaStream_t stream, const int num_images, const int num_items_per_image,
    void* unsorted_scores, void* unsorted_bbox_indices, void* sorted_scores, void* sorted_bbox_indices, void* workspace,
    int score_bits)
{
    void* d_offsets = workspace;
    void* cubWorkspace = nextWorkspacePtr((int8_t*) d_offsets, (num_images + 1) * sizeof(int));

    setUniformOffsets(stream, num_images, num_items_per_image, (int*) d_offsets);

    const int arrayLen = num_images * num_items_per_image;
    size_t temp_storage_bytes = cubSortPairsWorkspaceSize<T_SCORE, int>(arrayLen, num_images);
    size_t begin_bit = 0;
    size_t end_bit = sizeof(T_SCORE) * 8;
    if (sizeof(T_SCORE) == 2 && score_bits > 0 && score_bits <= 10)
    {
        end_bit = 10;
        begin_bit = end_bit - score_bits;
    }
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        cubWorkspace, temp_storage_bytes,
        (const T_SCORE*) (unsorted_scores), (T_SCORE*) (sorted_scores),
        (const int*) (unsorted_bbox_indices), (int*) (sorted_bbox_indices),
        arrayLen, num_images,
        (const int*) d_offsets, (const int*) d_offsets + 1,
        begin_bit, end_bit,
        stream);
    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// sortScoresPerImage LAUNCH CONFIG
typedef pluginStatus_t (*sspiFunc)(cudaStream_t,
                                const int,
                                const int,
                                void*,
                                void*,
                                void*,
                                void*,
                                void*,
                                int);
struct sspiLaunchConfig
{
    DataType t_score;
    sspiFunc function;

    sspiLaunchConfig(DataType t_score)
        : t_score(t_score)
    {
    }
    sspiLaunchConfig(DataType t_score, sspiFunc function)
        : t_score(t_score)
        , function(function)
    {
    }
    bool operator==(const sspiLaunchConfig& other)
    {
        return t_score == other.t_score;
    }
};

static std::array<sspiLaunchConfig, 2> sspiLCOptions = {
    sspiLaunchConfig(DataType::kFLOAT, sortScoresPerImage_gpu<float>),
    sspiLaunchConfig(DataType::kHALF, sortScoresPerImage_gpu<__half>),
};

pluginStatus_t sortScoresPerImage(
    cudaStream_t stream,
    const int num_images,
    const int num_items_per_image,
    const DataType DT_SCORE,
    void* unsorted_scores,
    void* unsorted_bbox_indices,
    void* sorted_scores,
    void* sorted_bbox_indices,
    void* workspace,
    int score_bits
)
{
    sspiLaunchConfig lc = sspiLaunchConfig(DT_SCORE);
    for (unsigned i = 0; i < sspiLCOptions.size(); ++i)
    {
        if (lc == sspiLCOptions[i])
        {
            DEBUG_PRINTF("sortScoresPerImage kernel %d\n", i);
            return sspiLCOptions[i].function(stream,
                                           num_images,
                                           num_items_per_image,
                                           unsorted_scores,
                                           unsorted_bbox_indices,
                                           sorted_scores,
                                           sorted_bbox_indices,
                                           workspace,
                                           score_bits);
        }
    }
    return STATUS_BAD_PARAM;
}

size_t sortScoresPerImageWorkspaceSize(
    const int num_images,
    const int num_items_per_image,
    const DataType DT_SCORE)
{
    const int arrayLen = num_images * num_items_per_image;
    size_t wss[2];
    wss[0] = (num_images + 1) * sizeof(int); // offsets
    if (DT_SCORE == DataType::kFLOAT)
    {
        wss[1] = cubSortPairsWorkspaceSize<float, int>(arrayLen, num_images); // cub workspace
    }
    else if (DT_SCORE == DataType::kHALF)
    {
        wss[1] = cubSortPairsWorkspaceSize<__half, int>(arrayLen, num_images); // cub workspace
    }
    else
    {
        printf("SCORE type not supported.\n");
        return (size_t) -1;
    }

    return calculateTotalWorkspaceSize(wss, 2);
}
