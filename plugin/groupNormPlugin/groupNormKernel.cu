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

#include "common/common.cuh"
#include "groupNormKernel.h"

static inline __device__ __host__ float sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}

struct Group_sums
{
    // Is it the 1st element of the group?
    int32_t flag;
    // The sum.
    float sum;
    // The sum of squares.
    float sum_sq;
};

struct Group_sums_op
{
    inline __device__ Group_sums operator()(Group_sums const& a, Group_sums const& b)
    {
        Group_sums dst;
        dst.sum = b.flag ? b.sum : (a.sum + b.sum);
        dst.sum_sq = b.flag ? b.sum_sq : (a.sum_sq + b.sum_sq);
        dst.flag = a.flag + b.flag;
        return dst;
    }
};

template <int32_t THREADS_PER_BLOCK>
__global__ void group_norm_nhwc_sum_kernel(Group_norm_nhwc_params params)
{
    // The object in charge of doing the sums for the different blocks.
    typedef cub::BlockScan<Group_sums, THREADS_PER_BLOCK> Block_scan;

    // Allocate shared memory for Block_scan.
    __shared__ typename Block_scan::TempStorage temp_storage;
    // Allocate shared memory for the groups. We could reduce the amount of shared
    // memory reserved.
    __shared__ float2 smem[THREADS_PER_BLOCK];

    // The instance in the batch.
    int32_t ni = blockIdx.z;
    // The channel loaded by that thread (2 channels per thread for F16x2).
    int32_t ci = blockIdx.x * params.c_per_block + threadIdx.x * 2;

    // The first activation loaded by that block.
    int32_t hw_begin = blockIdx.y * params.hw_per_block;
    // The last activation loaded by that block.
    int32_t hw_end = min(hw_begin + params.hw_per_block, params.hw);

    // The sums.
    float sum = 0.f, sum_sq = 0.f;

    // Iterate over the activations to compute the sums.
    for (int32_t hwi = hw_begin; hwi < hw_end; ++hwi)
    {
        // The offset.
        int64_t offset = (int64_t) ni * params.hwc + hwi * params.c + ci;

        // Fetch two channels per thread.
        __half2 h2(0, 0);
        if (ci < params.c)
        {
            h2 = *reinterpret_cast<__half2 const*>(&params.src[offset]);
        }

        // Extract the two half values.
        float2 f2 = __half22float2(h2);

        // Update the sum.
        sum += f2.x + f2.y;
        // Update the sum of squares.
        sum_sq += f2.x * f2.x + f2.y * f2.y;
    }

    // The group that thread works on and the channel in the group (modulus).
    int32_t gi = threadIdx.x * 2 / params.c_per_group;
    int32_t cj = threadIdx.x * 2 - params.c_per_group * gi;

    // The data for the summations.
    Group_sums inp{cj == 0 ? 1 : 0, sum, sum_sq};

    // Do the segmented scan.
    Group_sums out;
    Block_scan(temp_storage).InclusiveScan(inp, out, Group_sums_op());

    // Store the results for the groups in shared memory (to produce coalesced
    // stores later).
    if (cj == params.c_per_group - 2 /* 2 channels per thread */)
    {
        smem[gi] = make_float2(out.sum, out.sum_sq);
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The global group index.
    int32_t gj = blockIdx.x * params.groups_per_block + threadIdx.x;

    // Threads that have nothing left to do, exit.
    if (threadIdx.x >= params.groups_per_block || gj >= params.groups)
    {
        return;
    }

    // The first threads (those storing to global memory, load the values).
    float2 sums = smem[threadIdx.x];

    // Store to global memory.
    atomicAdd(&params.red_buffer[(2 * ni + 0) * params.groups + gj], sums.x);
    atomicAdd(&params.red_buffer[(2 * ni + 1) * params.groups + gj], sums.y);
}

void group_norm_nhwc_sum(Group_norm_nhwc_params const& params, cudaStream_t stream)
{
    // Make sure the values are as we expect.
    assert(params.c % params.c_per_block == 0 && params.hw % params.hw_per_block == 0);
    // Make sure a group does not span multiple blocks.
    assert(params.c_per_block % params.c_per_group == 0);

    dim3 grid;

    // The number of blocks to compute all the channels.
    grid.x = params.c / params.c_per_block;
    // The number of blocks to compute all the activations in a given instance.
    grid.y = divUp(params.hw, params.hw_per_block);
    // The number of instances.
    grid.z = params.n;

    switch (params.c_per_block)
    {
    case 320: group_norm_nhwc_sum_kernel<160><<<grid, 160, 0, stream>>>(params); break;
    case 480: group_norm_nhwc_sum_kernel<256><<<grid, 256, 0, stream>>>(params); break;
    case 256: group_norm_nhwc_sum_kernel<128><<<grid, 128, 0, stream>>>(params); break;
    case 128: group_norm_nhwc_sum_kernel<64><<<grid, 64, 0, stream>>>(params); break;
    default: assert(false); // Not implemented!
    }

    PLUGIN_CUASSERT(cudaGetLastError());
}

template <int32_t THREADS_PER_BLOCK>
__global__ void group_norm_nhwc_scale_kernel(Group_norm_nhwc_params params)
{
    // The instance in the batch.
    int32_t ni = blockIdx.z;
    // The channel loaded by that thread (2 channels per thread for F16x2).
    int32_t ci = blockIdx.x * params.c_per_block + threadIdx.x * 2;
    // The group that thread works on and the channel in the group (modulus).
    int32_t gi = ci / params.c_per_group;

    // Load the sum and sum of squares for the group.
    float sum = 0.f, sum_sq = 0.f;
    if (gi < params.groups)
    {
        sum = params.red_buffer[(2 * ni + 0) * params.groups + gi];
        sum_sq = params.red_buffer[(2 * ni + 1) * params.groups + gi];
    }

    // Load gamma/beta.
    float2 gamma_f2, beta_f2;
    if (ci < params.c)
    {
        gamma_f2 = *reinterpret_cast<float2 const*>(&params.gamma[ci]);
        beta_f2 = *reinterpret_cast<float2 const*>(&params.beta[ci]);
    }

    // Compute the mean.
    float mean = sum * params.inv_hwc;
    // Compute the variance.
    float var = sum_sq * params.inv_hwc - (mean * mean);
    // Compute the inverse of the stddev.
    float inv_stddev = var <= 0.f ? 1.f : rsqrtf(var);

    // The first activation loaded by that block.
    int32_t hw_begin = blockIdx.y * params.hw_per_block;
    // The last activation loaded by that block.
    int32_t hw_end = min(hw_begin + params.hw_per_block, params.hw);

    // Iterate over the activations to compute the sums.
    for (int32_t hwi = hw_begin; hwi < hw_end; ++hwi)
    {
        // The src/dst offset.
        int64_t offset = (int64_t) ni * params.hwc + hwi * params.c + ci;

        // Fetch two channels per thread.
        __half2 h2(0, 0);
        if (ci < params.c)
        {
            h2 = *reinterpret_cast<__half2 const*>(&params.src[offset]);
        }

        // Extract the two half values.
        float2 f2 = __half22float2(h2);

        // Normalize the channels.
        f2.x = (f2.x - mean) * inv_stddev;
        f2.y = (f2.y - mean) * inv_stddev;

        // Scale by gamma and add beta.
        f2.x = gamma_f2.x * f2.x + beta_f2.x;
        f2.y = gamma_f2.y * f2.y + beta_f2.y;

        // Apply Swish if needed.
        if (params.with_swish)
        {
            f2.x = f2.x * sigmoid(f2.x);
            f2.y = f2.y * sigmoid(f2.y);
        }

        // Store the scaled values.
        if (ci < params.c)
        {
            *reinterpret_cast<__half2*>(&params.dst[offset]) = __float22half2_rn(f2);
        }
    }
}

void group_norm_nhwc_scale(Group_norm_nhwc_params const& params, cudaStream_t stream)
{
    // Make sure the dimensions are aligned with what we expect.
    assert(params.c % params.c_per_block == 0);
    // Make sure a group does not span multiple blocks.
    assert(params.c_per_block % params.c_per_group == 0);

    dim3 grid;

    // The number of blocks to compute all the channels.
    grid.x = params.c / params.c_per_block;
    // The number of blocks to compute all the activations in a given instance.
    grid.y = divUp(params.hw, params.hw_per_block);
    // The number of instances.
    grid.z = params.n;

    switch (params.c_per_block)
    {
    case 320: group_norm_nhwc_scale_kernel<160><<<grid, 160, 0, stream>>>(params); break;
    case 480: group_norm_nhwc_scale_kernel<256><<<grid, 256, 0, stream>>>(params); break;
    case 256: group_norm_nhwc_scale_kernel<128><<<grid, 128, 0, stream>>>(params); break;
    case 128: group_norm_nhwc_scale_kernel<64><<<grid, 64, 0, stream>>>(params); break;
    default: assert(false); // Not implemented!
    }

    PLUGIN_CUASSERT(cudaGetLastError());
}
