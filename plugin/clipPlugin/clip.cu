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
#include "clipPlugin/clip.h"
#include "common/checkMacrosPlugin.h"
#include <cuda_fp16.h>

// Integer division rounding up
inline __host__ __device__ constexpr int divUp(int x, int n)
{
    return (x + n - 1) / n;
}

template <typename T1, typename T2, unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
    __global__ void clipKernel(
        int n,
        const T1 clipMin,
        const T1 clipMax,
        const T2* input,
        T2* output)
{
    // each global thread handles one element
    int i = blockIdx.x * nthdsPerCTA + threadIdx.x;
    if (i < n)
    {
        T1 inputElement = static_cast<T1>(input[i]);
        T1 tmp = inputElement > clipMin ? inputElement : clipMin;
        output[i] = static_cast<T2>(tmp < clipMax ? tmp : clipMax);
    }
}

int clipInference(
    cudaStream_t stream,
    int n,
    float clipMin,
    float clipMax,
    const void* input,
    void* output,
    nvinfer1::DataType type)
{
    const int BS = 512;
    const int GS = divUp(n, BS);

    switch (type)
    {
    case nvinfer1::DataType::kFLOAT:
    {
        clipKernel<float, float, BS><<<GS, BS, 0, stream>>>(n, clipMin, clipMax,
                                                            static_cast<const float*>(input),
                                                            static_cast<float*>(output));
        break;
    }
    case nvinfer1::DataType::kHALF:
    {
        /* Implementing kHALF operation using float operands. function __float2half
         * is not supported for CUDA versions <= 9.1 causing compilation failures
         * Moreover operand > for __half operand is only supported if __CUDA_ARCH__
         * >= 530
         */
        clipKernel<float, half, BS><<<GS, BS, 0, stream>>>(n,
                                                           clipMin, clipMax,
                                                           static_cast<const half*>(input),
                                                           static_cast<half*>(output));
        break;
    }
    case nvinfer1::DataType::kINT32:
    {
        clipKernel<int32_t, int32_t, BS><<<GS, BS, 0, stream>>>(n,
                                                                static_cast<int32_t>(clipMin), static_cast<int32_t>(clipMax),
                                                                static_cast<const int32_t*>(input),
                                                                static_cast<int32_t*>(output));
        break;
    }
    case nvinfer1::DataType::kINT8:
    {
        clipKernel<int8_t, int8_t, BS><<<GS, BS, 0, stream>>>(n,
                                                              static_cast<int8_t>(clipMin), static_cast<int8_t>(clipMax),
                                                              static_cast<const int8_t*>(input),
                                                              static_cast<int8_t*>(output));
        break;
    }
    case nvinfer1::DataType::kUINT8:
    {
        PLUGIN_FAIL("unsupported datatype");
        break;
    }
    case nvinfer1::DataType::kBOOL:
    {
        PLUGIN_FAIL("unsupported datatype");
        break;
    }
    }

    return 0;
}
