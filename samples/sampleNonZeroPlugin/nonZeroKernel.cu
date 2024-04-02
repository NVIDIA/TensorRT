/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nonZeroKernel.h"

__global__ void findNonZeroIndicesKernel(
    float const* X, int32_t* indices, int32_t* count, int32_t const* K, int32_t R, int32_t C, bool rowMajor)
{
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the column index is within bounds
    if (col < C)
    {
        for (int32_t row = 0; row < R; ++row)
        {
            if (X[row + R * col] != 0.F)
            {
                int32_t index = atomicAdd(count, 1); // Increment count atomically and get the previous value
                if (indices)
                {
                    if(!rowMajor)
                    {
                        indices[index] = row;
                        indices[index + *K] = col;
                    }
                    else
                    {
                        indices[2 * index] = row;
                        indices[2 * index + 1] = col;
                    }
                }
            }
        }
    }
}

void nonZeroIndicesImpl(
    float const* X, int32_t* indices, int32_t* count, int32_t const* K, int32_t R, int32_t C, bool rowMajor, cudaStream_t stream)
{
    constexpr int32_t kBLOCK_SIZE = 256;
    int32_t const blocksPerGrid = (R + kBLOCK_SIZE - 1) / kBLOCK_SIZE;
        
    findNonZeroIndicesKernel<<<blocksPerGrid, kBLOCK_SIZE, 0, stream>>>(X, indices, count, K, R, C, rowMajor);
}
