/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "NvInfer.h"
#include <cassert>
#include <cub/cub.cuh>
#include <plugin_util.hpp>

template <typename T, unsigned TPB>
__global__ void scaled_softmax_kernel_small(int ld, const float rsqrt_head_size, const T* input, T* output)
{

    typedef cub::BlockReduce<float, TPB> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    __shared__ float rZ;

    int offset = blockIdx.x * ld;

    float w(rsqrt_head_size);
    cub::Sum sum;
    float thread_data(0);

    int idx = offset + threadIdx.x;
    if (threadIdx.x < ld)
    {
        float val = input[idx];
        thread_data = myExp(val * w);
    }

    auto Z = BlockReduce(temp_storage).Reduce(thread_data, sum);

    if (threadIdx.x == 0)
    {
        rZ = (1.f) / Z;
    }
    __syncthreads();

    if (threadIdx.x < ld)
    {
        output[idx] = T(thread_data * rZ);
    }
}

template <typename T, unsigned TPB>
__global__ void scaled_softmax_kernel(int ld, const float rsqrt_head_size, const T* input, T* output)
{

    typedef cub::BlockReduce<float, TPB> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    __shared__ float rZ;

    int offset = blockIdx.x * ld;

    float w(rsqrt_head_size);
    cub::Sum sum;
    float thread_data(0);

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        int idx = offset + i;
        float val = input[idx];
        thread_data += myExp(val * w);
    }

    auto Z = BlockReduce(temp_storage).Reduce(thread_data, sum);

    if (threadIdx.x == 0)
    {
        rZ = 1.f / Z;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        int idx = offset + i;
        float val = myExp(float(input[idx]) * w) * rZ;

        output[idx] = T(val);
    }
}

template <typename T>
int compute_scaled_softmax(cudaStream_t stream, int ld, int n, const float rsqrt_head_size, const T* input, T* output)
{

    assert(n % ld == 0);
    const int gridSize = n / ld;
    if (ld <= 32)
    { 
        const int blockSize = 32;
        scaled_softmax_kernel_small<T, blockSize>
            <<<gridSize, blockSize, 0, stream>>>(ld, rsqrt_head_size, input, output);
    }
    else if (ld <= 128)
    { 
        const int blockSize = 128;
        scaled_softmax_kernel_small<T, blockSize>
            <<<gridSize, blockSize, 0, stream>>>(ld, rsqrt_head_size, input, output);
    }
    else if (ld == 384)
    { 
        const int blockSize = 384;
        scaled_softmax_kernel_small<T, blockSize>
            <<<gridSize, blockSize, 0, stream>>>(ld, rsqrt_head_size, input, output);
    }
    else
    {

        const int blockSize = 256;

        scaled_softmax_kernel<T, blockSize><<<gridSize, blockSize, 0, stream>>>(ld, rsqrt_head_size, input, output);
    }

    CHECK(cudaPeekAtLastError());
    return 0;
}

template <typename T, unsigned TPB>
__global__ void masked_scaled_softmax_kernel_small(
    const int ld, const float rsqrt_head_size, const int* mask_idx, const T* input, T* output)
{

    typedef cub::BlockReduce<float, TPB> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    __shared__ float rZ;
    __shared__ int last_valid;

    if (threadIdx.x == 0)
    {
        last_valid = min(ld, mask_idx[blockIdx.y]);
    }
    __syncthreads();

    int offset = (blockIdx.y * gridDim.x + blockIdx.x) * ld;

    float w(rsqrt_head_size);
    cub::Sum sum;
    float thread_data(0);

    int idx = offset + threadIdx.x;
    if (threadIdx.x < last_valid)
    {
        float val = input[idx];
        thread_data = myExp(val * w);
    }

    auto Z = BlockReduce(temp_storage).Reduce(thread_data, sum);

    if (threadIdx.x == 0)
    {
        rZ = (1.f) / Z;
    }
    __syncthreads();

    if (threadIdx.x < ld)
    {
        // this will be 0 for threadIdx.x >= last_valid
        output[idx] = T(thread_data * rZ);
    }
}

template <typename T, unsigned TPB>
__global__ void masked_scaled_softmax_kernel(
    int ld, const float rsqrt_head_size, const int* mask_idx, const T* input, T* output)
{

    typedef cub::BlockReduce<float, TPB> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    __shared__ float rZ;
    __shared__ int last_valid;

    if (threadIdx.x == 0)
    {
        last_valid = min(ld, mask_idx[blockIdx.y]);
    }
    __syncthreads();

    int offset = (blockIdx.y * gridDim.x + blockIdx.x) * ld;

    float w(rsqrt_head_size);
    cub::Sum sum;
    float thread_data(0);

    for (int i = threadIdx.x; i < last_valid; i += TPB)
    {
        int idx = offset + i;
        float val = input[idx];
        thread_data += myExp(val * w);
    }

    auto Z = BlockReduce(temp_storage).Reduce(thread_data, sum);

    if (threadIdx.x == 0)
    {
        rZ = 1.f / Z;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        int idx = offset + i;
        float val = (i < last_valid) ? myExp(float(input[idx]) * w) * rZ : 0.f;
        output[idx] = T(val);
    }
}

template <typename T>
int compute_masked_scaled_softmax(cudaStream_t stream, const int ld, const int B, const int N,
    const float rsqrt_head_size, const int* mask_idx, const T* input, T* output)
{
    // Mask idx is of length B and assumes the valid region is contiguous starting
    // from the beginning of the sequence

    dim3 grid(ld * N, B, 1);

    if (ld <= 32)
    {
        const int blockSize = 32;
        masked_scaled_softmax_kernel_small<T, blockSize>
            <<<grid, blockSize, 0, stream>>>(ld, rsqrt_head_size, mask_idx, input, output);
    }
    else if (ld <= 128)
    {
        const int blockSize = 128;
        masked_scaled_softmax_kernel_small<T, blockSize>
            <<<grid, blockSize, 0, stream>>>(ld, rsqrt_head_size, mask_idx, input, output);
    }
    else if (ld == 384)
    {
        const int blockSize = 384;
        masked_scaled_softmax_kernel_small<T, blockSize>
            <<<grid, blockSize, 0, stream>>>(ld, rsqrt_head_size, mask_idx, input, output);
    }
    else
    {

        const int blockSize = 256;

        // this must be true because n is the total size of the tensor
        masked_scaled_softmax_kernel<T, blockSize>
            <<<grid, blockSize, 0, stream>>>(ld, rsqrt_head_size, mask_idx, input, output);
    }

    CHECK(cudaPeekAtLastError());
    return 0;
}

template <unsigned TPB>
__global__ void mask_idx_kernel_small(int ld, const int* mask, int* mask_idx)
{

    typedef cub::BlockReduce<int, TPB> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // ld is S
    // blockIdx.x is b

    int offset = blockIdx.x * ld; // batch strides of S

    cub::Min min;
    int thread_data(ld); // if the mask admits all values

    int idx = offset + threadIdx.x;
    if (threadIdx.x < ld)
    {
        int val = mask[idx];
        if (val == 0) // masked position: report thread idx
        {
            thread_data = threadIdx.x;
        }
    }

    auto minIdx = BlockReduce(temp_storage).Reduce(thread_data, min);

    if (threadIdx.x == 0)
    {
        mask_idx[blockIdx.x] = minIdx;
    }
}

template <unsigned TPB>
__global__ void mask_idx_kernel(int ld, const int* mask, int* mask_idx)
{

    typedef cub::BlockReduce<int, TPB> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // ld is S
    // blockIdx.x is b

    int offset = blockIdx.x * ld; // batch strides of S

    cub::Min min;
    int thread_data(ld); // if the mask admits all values

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        int idx = offset + i;
        int val = mask[idx];
        if (val == 0) // masked position: report thread idx
        {
            thread_data = min(thread_data, i);
        }
    }

    auto minIdx = BlockReduce(temp_storage).Reduce(thread_data, min);

    if (threadIdx.x == 0)
    {
        mask_idx[blockIdx.x] = minIdx;
    }
}

inline int compute_mask_idx(cudaStream_t stream, const int S, const int B, const int* mask, int* mask_idx)
{
    // Mask idx is of length B and assumes the valid region is contiguous starting
    // from the beginning of the sequence

    // Assume n = BxS
    if (S <= 32)
    {
        mask_idx_kernel_small<32><<<B, 32, 0, stream>>>(S, mask, mask_idx);
    }
    else if (S <= 128)
    {
        mask_idx_kernel_small<128><<<B, 128, 0, stream>>>(S, mask, mask_idx);
    }
    else if (S == 384)
    {
        mask_idx_kernel_small<384><<<B, 384, 0, stream>>>(S, mask, mask_idx);
    }
    else
    {
        mask_idx_kernel<256><<<B, 256, 0, stream>>>(S, mask, mask_idx);
    }

    CHECK(cudaPeekAtLastError());
    return 0;
}
