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

#include "common/kernel.h"

template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA) __global__
    void pReLUKernel(const int n, const float negativeSlope, const float* input, float* output)
{
    for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x; i < n; i += gridDim.x * nthdsPerCTA)
    {
        output[i] = input[i] > 0 ? input[i] : input[i] * negativeSlope;
    }
}

pluginStatus_t lReLUGPU(cudaStream_t stream, const int n, const float negativeSlope, const void* input, void* output)
{
    const int BS = 512;
    const int GS = (n + BS - 1) / BS;
    pReLUKernel<BS><<<GS, BS, 0, stream>>>(n, negativeSlope,
                                           (const float*) input,
                                           (float*) output);
    return STATUS_SUCCESS;
}

pluginStatus_t lReLUInference(
    cudaStream_t stream, const int n, const float negativeSlope, const void* input, void* output)
{
    return lReLUGPU(stream, n, negativeSlope, (const float*) input, (float*) output);
}
