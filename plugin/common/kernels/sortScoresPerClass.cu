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

inline __device__ __half add_fb(const __half& a, const __half& b)
{
#if __CUDA_ARCH__ >= 530
    return a + b;
#else
    return __float2half(__half2float(a) + __half2float(b));
#endif
}

// overload for float
inline __device__ float add_fb(const float & a, const float & b) {
    return a + b;
}

inline __device__ bool ge_fb(const __half & a, const __half & b) {
#if __CUDA_ARCH__ >= 530
    return a >= b;
#else
    return __half2float(a) >= __half2float(b);
#endif
}

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
        T_SCORE score_shift,
        int* temp_idx,
        int* d_offsets)
{
    // Prepare scores data for sort
    const int cur_idx = blockIdx.x * nthds_per_cta + threadIdx.x;
    const int numPredsPerBatch = num_classes * num_preds_per_class;
    T_SCORE clip_val = T_SCORE(float(score_shift) + 1.f - 1.f / 1024.f);
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
                // add shift of 1.0 to normalize the score values
                // to the range [1, 2). 
                // add a constant shift to scores will not change the sort
                // result, but will help reduce the computation because
                // we only need to sort the mantissa part of the floating-point
                // numbers
                temp_scores[targetIdx] = score_shift;
                temp_idx[targetIdx] = -1;
                conf_scores_gpu[targetIdx] = score_shift;
            }
            // "Clear" scores lower than threshold
            else
            {
                if (float(score) > confidence_threshold)
                {
                    // add shift of 1.0 to normalize the score values
                    // to the range [1, 2). 
                    // add a constant shift to scores will not change the sort
                    // result, but will help reduce the computation because
                    // we only need to sort the mantissa part of the floating-point
                    // numbers
                    temp_scores[targetIdx] = add_fb(score, score_shift);
                    if (float(score_shift) > 0.f && (ge_fb(temp_scores[targetIdx], clip_val)))
                        temp_scores[targetIdx] = clip_val;
                    temp_idx[targetIdx] = cur_idx + i * numPredsPerBatch;
                }
                else
                {
                    // Set scores to 0
                    // Set label = -1
                    // add shift of 1.0 to normalize the score values
                    // to the range [1, 2). 
                    // add a constant shift to scores will not change the sort
                    // result, but will help reduce the computation because
                    // we only need to sort the mantissa part of the floating-point
                    // numbers
                    temp_scores[targetIdx] = score_shift;
                    temp_idx[targetIdx] = -1;
                    conf_scores_gpu[targetIdx] = score_shift;
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
    void* workspace,
    const int score_bits,
    const float score_shift
)
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
    // prepare the score, index, and offsets for CUB radix sort
    // also normalize the scores to the range [1, 2)
    // so we only need to sort the mantissa of floating-point numbers
    // since their sign bit and exponential bits are identical
    // we will subtract the 1.0 shift in gatherTopDetections()
    prepareSortData<T_SCORE, BS><<<GS, BS, 0, stream>>>(num, num_classes, num_preds_per_class,
                                                        background_label_id, confidence_threshold,
                                                        (T_SCORE*) conf_scores_gpu,
                                                        (T_SCORE*) temp_scores,
                                                        T_SCORE(score_shift),
                                                        (int*) temp_idx,
                                                        (int*) d_offsets);

    size_t temp_storage_bytes = cubSortPairsWorkspaceSize<T_SCORE, int>(arrayLen, num_segments);
    size_t begin_bit = 0;
    size_t end_bit = sizeof(T_SCORE) * 8;
    if (sizeof(T_SCORE) == 2 && score_bits > 0 && score_bits <= 10)
    {
        // only sort score_bits in 10 mantissa bits.
        end_bit = 10;
        begin_bit = end_bit - score_bits;
    }
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        cubWorkspace, temp_storage_bytes,
        (const T_SCORE*) (temp_scores), (T_SCORE*) (conf_scores_gpu),
        (const int*) (temp_idx), (int*) (index_array_gpu),
        arrayLen, num_segments,
        (const int*) d_offsets, (const int*) d_offsets + 1,
        begin_bit, end_bit,
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
                                void*,
                                const int,
                                const float);
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

static std::array<sspcLaunchConfig, 2> sspcLCOptions = {
    sspcLaunchConfig(DataType::kFLOAT, sortScoresPerClass_gpu<float>),
    sspcLaunchConfig(DataType::kHALF, sortScoresPerClass_gpu<__half>)
};

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
    void* workspace,
    const int score_bits,
    const float score_shift
)
{
    sspcLaunchConfig lc = sspcLaunchConfig(DT_SCORE);
    for (unsigned i = 0; i < sspcLCOptions.size(); ++i)
    {
        if (lc == sspcLCOptions[i])
        {
            DEBUG_PRINTF("sortScoresPerClass kernel %d\n", i);
            return sspcLCOptions[i].function(stream,
                                           num,
                                           num_classes,
                                           num_preds_per_class,
                                           background_label_id,
                                           confidence_threshold,
                                           conf_scores_gpu,
                                           index_array_gpu,
                                           workspace,
                                           score_bits,
                                           score_shift);
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
    else if (DT_CONF == DataType::kHALF)
    {
        wss[3] = cubSortPairsWorkspaceSize<__half, int>(arrayLen, num * num_classes); // cub workspace
    }
    else
    {
        printf("SCORE type not supported\n");
        return (size_t) -1;
    }

    return calculateTotalWorkspaceSize(wss, 4);
}
