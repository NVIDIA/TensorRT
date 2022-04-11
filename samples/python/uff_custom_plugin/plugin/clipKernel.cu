/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <clipKernel.h>

template <typename T>
__device__ __forceinline__ const T& min(const T& a, const T& b)
{
    return (a > b) ? b : a;
}

template <typename T>
__device__ __forceinline__ const T& max(const T& a, const T& b)
{
    return (a > b) ? a : b;
}

template <typename T, unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
    __global__ void clipKernel(
        int n,
        const T clipMin,
        const T clipMax,
        const T* input,
        T* output)
{
    for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x; i < n; i += gridDim.x * nthdsPerCTA)
    {
        output[i] = min<T>(max<T>(input[i], clipMin), clipMax);
    }
}

int clipInference(
    cudaStream_t stream,
    int n,
    float clipMin,
    float clipMax,
    const void* input,
    void* output)
{
    const int blockSize = 512;
    const int gridSize = (n + blockSize - 1) / blockSize;
    clipKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(n, clipMin, clipMax,
                                                 static_cast<const float*>(input),
                                                 static_cast<float*>(output));
    return 0;
}
