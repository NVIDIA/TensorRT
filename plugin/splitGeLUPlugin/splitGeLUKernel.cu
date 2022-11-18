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

#include "splitGeLUKernel.h"

template <typename T, int32_t HHS, int32_t TPB>
__global__ void splitGeLUKernel(T const* input, T* output, float const fDiv, float const fAdd, float const fMul)
{
    int32_t indexInput = blockIdx.x * HHS * 2 + threadIdx.x;
    int32_t indexOutput = blockIdx.x * HHS + threadIdx.x;

#pragma unroll
    for (int32_t i = 0; i < HHS / TPB; ++i)
    {
        auto valueL = static_cast<float>(input[indexInput]);
        auto valueR = static_cast<float>(input[indexInput + HHS]);
        float tmp = valueR;
        tmp /= fDiv;
        tmp = erff(tmp);
        tmp += fAdd;
        tmp *= valueR;
        tmp *= fMul;
        tmp *= valueL;
        output[indexOutput] = static_cast<T>(tmp);
        indexInput += TPB;
        indexOutput += TPB;
    }
    return;
}

template <typename T>
int32_t launchSplitGeLUKernel(cudaStream_t stream, int32_t gridSize, int32_t nHalfHiddenSize, T const* input, T* output,
    float const fDiv, float const fAdd, float const fMul)
{
    constexpr int32_t TPB = 256; // thread per block
    switch (nHalfHiddenSize)
    {
    case 1280: (splitGeLUKernel<T, 1280, TPB>) <<<gridSize, TPB, 0, stream>>>(input, output, fDiv, fAdd, fMul); break;
    case 2560: (splitGeLUKernel<T, 2560, TPB>) <<<gridSize, TPB, 0, stream>>>(input, output, fDiv, fAdd, fMul); break;
    case 5120: (splitGeLUKernel<T, 5120, TPB>) <<<gridSize, TPB, 0, stream>>>(input, output, fDiv, fAdd, fMul); break;
    }

    PLUGIN_CHECK_CUDA(cudaPeekAtLastError());
    return 0;
}

template __global__ void splitGeLUKernel<float, 1280, 256>(float const*, float*, float const, float const, float const);
template __global__ void splitGeLUKernel<float, 2560, 256>(float const*, float*, float const, float const, float const);
template __global__ void splitGeLUKernel<float, 5120, 256>(float const*, float*, float const, float const, float const);
template __global__ void splitGeLUKernel<half, 1280, 256>(half const*, half*, float const, float const, float const);
template __global__ void splitGeLUKernel<half, 2560, 256>(half const*, half*, float const, float const, float const);
template __global__ void splitGeLUKernel<half, 5120, 256>(half const*, half*, float const, float const, float const);

template int32_t launchSplitGeLUKernel<float>(cudaStream_t stream, int32_t gridSize, int32_t nHalfHiddenSize,
    float const* input, float* output, float const fDiv, float const fAdd, float const fMul);

template int32_t launchSplitGeLUKernel<half>(cudaStream_t stream, int32_t gridSize, int32_t nHalfHiddenSize,
    half const* input, half* output, float const fDiv, float const fAdd, float const fMul);
