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

#include <cuda.h>
#if CUDA_VERSION >= 10010

#include <cstring>
#include <vector>

#include "NvInfer.h"
#include "common/bertCommon.h"
#include "common/common.cuh"
#include "common/serialize.hpp"
#include "geluPlugin.h"

using namespace nvinfer1;

namespace nvinfer1
{
namespace plugin
{
namespace bert
{

// constants for approximating the normal cdf
constexpr float A = 0.5f;
constexpr float B = 0.7978845608028654f;   // sqrt(2.0/M_PI)
constexpr float C = 0.035677408136300125f; // 0.044715 * sqrt(2.0/M_PI)

template <typename T, unsigned TPB>
__global__ void geluKernel(const T a, const T b, const T c, int n, const T* input, T* output)
{
    const int idx = blockIdx.x * TPB + threadIdx.x;

    if (idx < n)
    {
        const T in = input[idx];
        const T cdf = a + a * tanh(in * (c * in * in + b));
        output[idx] = in * cdf;
    }
}

int computeGelu(cudaStream_t stream, int n, const float* input, float* output)
{
    constexpr int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    geluKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(A, B, C, n, input, output);

    PLUGIN_CHECK(cudaPeekAtLastError());
    return 0;
}

int computeGelu(cudaStream_t stream, int n, const half* input, half* output)
{
    constexpr int blockSize = 256;

    if (0 == (n & 1))
    {
        const int n2 = n / 2;

        const int gridSize = (n2 + blockSize - 1) / blockSize;
        const half2 A2 = __floats2half2_rn(A, A);
        const half2 B2 = __floats2half2_rn(B, B);
        const half2 C2 = __floats2half2_rn(C, C);
        const half2* input2 = reinterpret_cast<const half2*>(input);
        half2* output2 = reinterpret_cast<half2*>(output);
        geluKernel<half2, blockSize><<<gridSize, blockSize, 0, stream>>>(A2, B2, C2, n2, input2, output2);
    }
    else
    {
        const int gridSize = (n + blockSize - 1) / blockSize;
        geluKernel<half, blockSize><<<gridSize, blockSize, 0, stream>>>(A, B, C, n, input, output);
    }

    PLUGIN_CHECK(cudaPeekAtLastError());
    return 0;
}

template <typename T, int TPB>
__global__ void geluBiasKernel(const T a, const T b, const T c, T* output, const T* input, const T* bias, const int ld)
{

    const int offset = blockIdx.x * ld;

    for (int it = threadIdx.x; it < ld; it += TPB)
    {
        const int idx = it + offset;
        const T in = input[idx] + bias[it];
        const T cdf = a + a * tanh(in * (c * in * in + b));
        output[idx] = in * cdf;
    }
}

int computeGeluBias(
    float* output, const float* input, const float* bias, const int ld, const int cols, cudaStream_t stream)
{
    geluBiasKernel<float, 256><<<cols, 256, 0, stream>>>(A, B, C, output, input, bias, ld);
    return cudaPeekAtLastError();
}

int computeGeluBias(
    half* output, const half* input, const half* bias, const int ld, const int cols, cudaStream_t stream)
{
    if (ld & 1)
    {
        geluBiasKernel<half, 256><<<cols, 256, 0, stream>>>(A, B, C, output, input, bias, ld);
    }
    else
    {

        const half2 A2 = __floats2half2_rn(A, A);
        const half2 B2 = __floats2half2_rn(B, B);
        const half2 C2 = __floats2half2_rn(C, C);
        const int ld2 = ld / 2;
        const half2* input2 = reinterpret_cast<const half2*>(input);
        const half2* bias2 = reinterpret_cast<const half2*>(bias);
        half2* output2 = reinterpret_cast<half2*>(output);
        geluBiasKernel<half2, 256><<<cols, 256, 0, stream>>>(A2, B2, C2, output2, input2, bias2, ld2);
    }

    return cudaPeekAtLastError();
}

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // CUDA_VERSION >= 10010
