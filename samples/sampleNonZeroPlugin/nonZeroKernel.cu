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

#include "nonZeroKernel.h"

inline __device__ int32_t isZero(float const& a)
{
    return a == 0.F;
}

inline __device__ int32_t isZero(half const& a)
{
#if __CUDA_ARCH__ >= 530
    return a == __float2half(0.F);
#else
    return __half2float(a) == 0.F;
#endif
}

template <typename T>
__global__ void findNonZeroIndicesKernel(
    T const* X, int32_t* indices, unsigned long long* count, unsigned long long const* K, int32_t R, int32_t C, int32_t rowOrder)
{
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the column index is within bounds
    if (col < C)
    {
        for (int32_t row = 0; row < R; ++row)
        {
            if (!isZero(X[row * C + col]))
            {
                unsigned long long index = atomicAdd(count, 1ULL); // Increment count atomically and get the previous value
                if (indices)
                {
                    if(rowOrder == 0)
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

template <typename T>
void nonZeroIndicesImpl(T const* X, int32_t* indices, int64_t* count, int64_t const* K, int32_t R, int32_t C,
    bool rowOrder, cudaStream_t stream)
{
    constexpr int32_t kBLOCK_SIZE = 256;
    int32_t const blocksPerGrid = (C + kBLOCK_SIZE - 1) / kBLOCK_SIZE;

    static_assert(sizeof(unsigned long long) == 8U, "unsigned long long must be 8 bytes in NVCC");
    findNonZeroIndicesKernel<<<blocksPerGrid, kBLOCK_SIZE, 0, stream>>>(
        X, indices, reinterpret_cast<unsigned long long*>(count), reinterpret_cast<unsigned long long const*>(K), R, C, static_cast<int32_t>(rowOrder));
}

#define NONZERO_SPECIALIZED_IMPL(T)                                                                                    \
    template void nonZeroIndicesImpl<T>(T const* X, int32_t* indices, int64_t* count, int64_t const* K, int32_t R,     \
        int32_t C, bool rowOrder, cudaStream_t stream);

NONZERO_SPECIALIZED_IMPL(float)
NONZERO_SPECIALIZED_IMPL(half)
