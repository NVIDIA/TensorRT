/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "maxPoolKernel.h"

#include <cstdint>

template <typename T>
__device__ __forceinline__ const T& max(const T& a, const T& b);

template <>
__device__ __forceinline__ const half& max(const half& a, const half& b)
{
#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
    return __hgt(a, b) ? a : b;
#else
    return (static_cast<float>(a) > static_cast<float>(b)) ? a : b;
#endif
}

template <>
__device__ __forceinline__ const float& max(const float& a, const float& b)
{
    return (a > b) ? a : b;
}

template <>
__device__ __forceinline__ int8_t const& max(int8_t const& a, int8_t const& b)
{
    return (a > b) ? a : b;
}

// Cuda kernel to find maximum in the kernelsize matrix
template <typename T>
    __global__ void maxKernel(
            int32_t B, int32_t C, int32_t H, int32_t W,
            const T* input,
            T* output, int32_t kernsize, int32_t stride, int32_t pad)
{
    // Total input volume
    int32_t const N = B * C * H * W;
    int32_t out_id;
    int32_t b, c, h, w;

    int32_t const H_out = (H + 2 * pad - kernsize) / stride + 1;
    int32_t const W_out = (W + 2 * pad - kernsize) / stride + 1;

    // Index in the output tensor
    out_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_id > B * C * H_out * W_out - 1)
    {
        return;
    }
    T maxim = static_cast<T>(0);

    // Output index of batch
    b = out_id / (C * H_out * W_out);
    int32_t const temp = out_id % (C * H_out * W_out);

    // Output index of channels
    c = temp / (H_out * W_out);
    int32_t const x = temp % (H_out * W_out);

    // Output index of height
    h = x / W_out; // row major format

    // Output index of width
    w = x % W_out;

    // Index in input tensor considering stride
    int32_t k = (b * C * H * W) + (c * (H * W)) + (h * stride * W) + (w * stride);

    maxim = input[k];
    // Find maximum value in the kernelsize matrix
    for (int32_t i = k; i < k + kernsize; i++)
    {
        for (int32_t j = 0; j < kernsize; j++)
        {
            if ((i + (j * W)) < N)
                maxim = max(maxim, input[i + (j * W)]);
        }
    }

    output[out_id] = maxim;
}

int32_t maxPoolFloat(cudaStream_t stream, int32_t batch_size, int32_t C, int32_t H, int32_t W, const void* input, void* output,
    int32_t kernsize, int32_t stride, int32_t pad)
{
    int32_t const blocksize = 512;
    // Compute number of entries in output
    int32_t const K = batch_size * C * ((H - kernsize + 2 * pad) / stride + 1) * ((W - kernsize + 2 * pad) / stride + 1);
    int32_t const g = ((K + blocksize - 1) / blocksize);

    maxKernel<float><<<g, blocksize, 0, stream>>>(
        batch_size, C, H, W, static_cast<const float*>(input), static_cast<float*>(output), kernsize, stride, pad);
    auto retVal = cudaStreamSynchronize(stream);
    if (retVal != cudaSuccess)
    {
        return 1;
    }
    return 0;
}

int32_t maxPoolHalf(cudaStream_t stream, int32_t batch_size, int32_t C, int32_t H, int32_t W, const void* input, void* output,
    int32_t kernsize, int32_t stride, int32_t pad)
{
    int32_t const blocksize = 512;
    // Compute number of entries in output
    int32_t const K = batch_size * C * ((H - kernsize + 2 * pad) / stride + 1) * ((W - kernsize + 2 * pad) / stride + 1);
    int32_t const g = ((K + blocksize - 1) / blocksize);

    maxKernel<half><<<g, blocksize, 0, stream>>>(
        batch_size, C, H, W, static_cast<const half*>(input), static_cast<half*>(output), kernsize, stride, pad);
    auto retVal = cudaStreamSynchronize(stream);
    if (retVal != cudaSuccess)
    {
        return 1;
    }
    return 0;
}

int32_t maxPoolInt8(cudaStream_t stream, int32_t batch_size, int32_t C, int32_t H, int32_t W, const void* input, void* output,
    int32_t kernsize, int32_t stride, int32_t pad)
{
    int32_t const blocksize = 512;
    // Compute number of entries in output
    int32_t const K = batch_size * C * ((H - kernsize + 2 * pad) / stride + 1) * ((W - kernsize + 2 * pad) / stride + 1);
    int32_t const g = ((K + blocksize - 1) / blocksize);

    maxKernel<int8_t><<<g, blocksize, 0, stream>>>(
        batch_size, C, H, W, static_cast<int8_t const*>(input), static_cast<int8_t*>(output), kernsize, stride, pad);
    auto retVal = cudaStreamSynchronize(stream);
    if (retVal != cudaSuccess)
    {
        return 1;
    }
    return 0;
}
