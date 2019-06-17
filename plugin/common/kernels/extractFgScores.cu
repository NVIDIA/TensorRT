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

template <typename T>
pluginStatus_t extractFgScores_gpu(cudaStream_t stream,
                                  int N,
                                  int A,
                                  int H,
                                  int W,
                                  const void* scores,
                                  void* fgScores)
{
    // Copy all the objectness scores for one batch
    size_t size = A * H * W * sizeof(T);
    for (int n = 0; n < N; n++)
    {
        // Find out the starting pointer of the objectness scores in the input
        size_t offset_ld = (n * 2 + 1) * A * H * W;
        // Find out the starting pointer of the objectness scores in the output
        size_t offset_st = n * A * H * W;
        CSC(cudaMemcpyAsync(((T*) fgScores) + offset_st, ((T*) scores) + offset_ld, size, cudaMemcpyDeviceToDevice, stream), STATUS_FAILURE);
    }

    return STATUS_SUCCESS;
}

template <typename T>
pluginStatus_t extractFgScores_cpu(int N,
                                  int A,
                                  int H,
                                  int W,
                                  const void* scores,
                                  void* fgScores)
{
    size_t size = A * H * W * sizeof(T);
    for (int n = 0; n < N; n++)
    {
        size_t offset_ld = (n * 2 + 1) * A * H * W;
        size_t offset_st = n * A * H * W;
        memcpy(((T*) fgScores) + offset_st, ((T*) scores) + offset_ld, size);
    }
    return STATUS_SUCCESS;
}

pluginStatus_t extractFgScores(cudaStream_t stream,
                              const int N,
                              const int A,
                              const int H,
                              const int W,
                              const DataType t_scores,
                              const DLayout_t l_scores,
                              const void* scores,
                              const DataType t_fgScores,
                              const DLayout_t l_fgScores,
                              void* fgScores)
{
    if (l_fgScores != NCHW || l_scores != NCHW)
        return STATUS_BAD_PARAM;

    if (t_fgScores != DataType::kFLOAT)
        return STATUS_BAD_PARAM;

    if (t_scores != DataType::kFLOAT)
        return STATUS_BAD_PARAM;

    return extractFgScores_gpu<float>(stream, N, A, H, W, scores, fgScores);
}
