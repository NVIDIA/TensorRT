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

#include "groupNormalizationPlugin.h"

namespace nvinfer1
{
namespace plugin
{

template <typename T, unsigned TPB>
__global__ void scaleShiftChannelsInplaceKernel(T* inOut, const int ld, const float* beta, const float* gamma)
{
    // grid is blocks x C x B
    // ld should be H*W
    // blockIdx.z = batch
    // blockIdx.y = channel
    // blockIdx.x = block per col
    const T b = beta[blockIdx.y];
    const T g = gamma[blockIdx.y];

    const int offset = (blockIdx.z * gridDim.y + blockIdx.y) * ld;

    const int tx = blockIdx.x * TPB + threadIdx.x;

    if (tx < ld)
    {
        inOut[offset + tx] = g * inOut[offset + tx] + b;
    }
}

template <typename T>
cudaError_t scaleShiftChannelsInplace(T* inOut, const int B, const int C, const int channelVolume, const float* beta,
    const float* gamma, cudaStream_t stream)
{

    constexpr int TPB = 256;
    const int colBlocks = (channelVolume + TPB - 1) / TPB;
    const dim3 grid(colBlocks, C, B);

    scaleShiftChannelsInplaceKernel<T, TPB><<<grid, TPB, 0, stream>>>(inOut, channelVolume, beta, gamma);

    return cudaPeekAtLastError();
}

template cudaError_t scaleShiftChannelsInplace<float>(float* inOut, const int B, const int C, const int channelVolume, const float* beta,
    const float* gamma, cudaStream_t stream);
} /* plugin */
} /* nvinfer1 */
