
/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#include <cuda.h>
#include "NvInfer.h"
#include "bertCommon.h"
#include <type_traits>
#include "common.cuh"
#include <cassert>
#include <cstring>
#include <vector>


using namespace nvinfer1;

namespace bert
{


inline __device__ char quantize(const float x, const float qScale)
{
    int tmpq = __float2int_rn(qScale * x);  // scale and round
    char tmpq8 = min(127, max(-127, tmpq)); // clip and cast
    return tmpq8;
}

inline __device__ void ldg(const int8_t* input, uint4& data)
{
    data = *reinterpret_cast<const uint4*>(input);
}

inline __device__ void stg(int8_t* output, uint4& data)
{
    *reinterpret_cast<uint4*>(output) = data;
}

inline __device__ void res_add(
    float (&hdata)[4], const uint32_t idata, const uint32_t ires, const float dqData, const float dqRes)
{
    char4 ires4 = reinterpret_cast<const char4&>(ires);
    char4 idata4 = reinterpret_cast<const char4&>(idata);
    hdata[0] = float(idata4.x) * dqData + float(ires4.x) * dqRes;
    hdata[1] = float(idata4.y) * dqData + float(ires4.y) * dqRes;
    hdata[2] = float(idata4.z) * dqData + float(ires4.z) * dqRes;
    hdata[3] = float(idata4.w) * dqData + float(ires4.w) * dqRes;
}

inline __device__ uint32_t pack4(const float (&hdata)[4], const float qScale)
{
    char4 ret;
    ret.x = quantize(hdata[0], qScale);
    ret.y = quantize(hdata[1], qScale);
    ret.z = quantize(hdata[2], qScale);
    ret.w = quantize(hdata[3], qScale);
    return reinterpret_cast<const uint32_t&>(ret);
}

template <int WARPS, int HEADS, int THREADS_PER_ROW>
__global__ void skipln_vec32(const int8_t* input, const int8_t* skip, int8_t* output, const half* beta,
    const half* gamma, const float dqScaleIn, const float dqScaleSkip, const float qScale, const int total)
{

    // clang-format off
    enum { HEAD_SIZE = 64 };
    enum { BYTES_PER_LDG = 16 };
    enum { THREADS_PER_CTA = WARPS * 32 };
    enum { ROWS_PER_LDG = THREADS_PER_CTA / THREADS_PER_ROW };
    enum { VECS_PER_CTA = THREADS_PER_ROW / 2 };
    enum { PARAM_BYTES = HEADS * HEAD_SIZE * 2 };
    enum { PARAM_LDGS = PARAM_BYTES / (THREADS_PER_CTA * BYTES_PER_LDG) };
    enum { LDGS = HEADS * 2 / ROWS_PER_LDG };
    // clang-format on
    static_assert(VECS_PER_CTA == 4, "");
    static_assert(PARAM_LDGS == 1, "");
    static_assert(ROWS_PER_LDG == HEADS , "");
    static_assert(LDGS == 2, "");
    static_assert(LDGS * ROWS_PER_LDG == HEADS * 2, "");
    static_assert(THREADS_PER_CTA * BYTES_PER_LDG == PARAM_BYTES, "");
    static_assert(PARAM_LDGS == 1, "");

    extern __shared__ char smem_[];

    // space for CTA-wide reduction
    __shared__ half2 smem_red[VECS_PER_CTA][WARPS];

    constexpr float rld = 1.f / (float(HEADS) * float(HEAD_SIZE));
    const int bidx = blockIdx.x;
    const int tidx = threadIdx.x;
    const int row = tidx / THREADS_PER_ROW;
    const int col = tidx % THREADS_PER_ROW;
    const int lane = tidx % 32;
    const int warp = tidx / 32;

    const bool is_warp_lead = (lane < THREADS_PER_ROW) && ((lane & 1) == 0);
    const bool is_cta_lead = (tidx < THREADS_PER_ROW) && ((tidx & 1) == 0);

    // token position: every two threads load together the 32B at one token
    // position
    const int pos = col / 2;

    const int pos_offset = bidx * VECS_PER_CTA + pos; // for token positions per block, disabling 2 threads per pos
    const bool my_pred = pos_offset < total;
    const int row_stride_bytes = total * 32;

    uint4 in_data[LDGS];
    uint4 in_skip[LDGS];
    float hdata[LDGS * 4][4];
    const int gmem_offset = row * row_stride_bytes + (bidx * THREADS_PER_ROW + col) * BYTES_PER_LDG;
#pragma unroll
    for (int ii = 0; ii < LDGS; ii++)
    {
        in_data[ii] = {0, 0, 0, 0};
        in_skip[ii] = {0, 0, 0, 0};
        if (my_pred)
        {
            ldg(input + gmem_offset + ii * ROWS_PER_LDG * row_stride_bytes, in_data[ii]);
            ldg(skip + gmem_offset + ii * ROWS_PER_LDG * row_stride_bytes, in_skip[ii]);
        }
    }

    uint4* smem_b = reinterpret_cast<uint4*>(&smem_[0]) + tidx;
    uint4* smem_g = reinterpret_cast<uint4*>(&smem_[PARAM_BYTES]) + tidx;

    const int8_t* beta_ptr = reinterpret_cast<const int8_t*>(beta) + tidx * BYTES_PER_LDG;
    const int8_t* gamma_ptr = reinterpret_cast<const int8_t*>(gamma) + tidx * BYTES_PER_LDG;
    ldg(beta_ptr, *smem_b);
    ldg(gamma_ptr, *smem_g);

    half* b = reinterpret_cast<half*>(&smem_[0]);
    half* g = reinterpret_cast<half*>(&smem_[PARAM_BYTES]);
#pragma unroll
    for (int ii = 0; ii < LDGS; ii++)
    {
        res_add(hdata[ii * 4 + 0], in_data[ii].x, in_skip[ii].x, dqScaleIn, dqScaleSkip);
        res_add(hdata[ii * 4 + 1], in_data[ii].y, in_skip[ii].y, dqScaleIn, dqScaleSkip);
        res_add(hdata[ii * 4 + 2], in_data[ii].z, in_skip[ii].z, dqScaleIn, dqScaleSkip);
        res_add(hdata[ii * 4 + 3], in_data[ii].w, in_skip[ii].w, dqScaleIn, dqScaleSkip);
    }

    half2 stats_local = {0, 0};

#pragma unroll
    for (int ii = 0; ii < LDGS * 4; ii++)
    {
#pragma unroll
        for (int jj = 0; jj < 4; jj++)
        {
            const float tmp = hdata[ii][jj] * (rld);
            stats_local = stats_local + __floats2half2_rn(tmp, tmp * hdata[ii][jj]);
        }
    }
    stats_local = stats_local + __shfl_xor_sync(uint32_t(-1), stats_local, 1); __syncwarp();

    if (VECS_PER_CTA == 1)
    { 
        stats_local = stats_local + __shfl_xor_sync(uint32_t(-1), stats_local, 2); __syncwarp();
        stats_local = stats_local + __shfl_xor_sync(uint32_t(-1), stats_local, 4); __syncwarp();
    }
    else if (VECS_PER_CTA == 2)
    {
        stats_local = stats_local + __shfl_xor_sync(uint32_t(-1), stats_local, 4); __syncwarp();
    }
    
    stats_local = stats_local + __shfl_xor_sync(uint32_t(-1), stats_local, 8); __syncwarp();
    stats_local = stats_local + __shfl_xor_sync(uint32_t(-1), stats_local, 16); __syncwarp();

    if (is_warp_lead)
    {
        smem_red[pos][warp] = stats_local;
    }

    __syncthreads();

    if (is_cta_lead)
    {
        for (int ii = 1; ii < WARPS; ii++)
        {
            stats_local = stats_local + smem_red[pos][ii];
        }

        float mu = __low2float(stats_local);
        float sos = __high2float(stats_local);
        float rsigma = rsqrtf(sos - mu * mu);

        smem_red[pos][0] = __floats2half2_rn(mu, rsigma);
    }
    __syncthreads();
    // load params into smem:  2x Headsx32x2x2B
    const float2 statsf = __half22float2(smem_red[pos][0]);

#pragma unroll
    for (int ii = 0; ii < LDGS; ii++)
    {
#pragma unroll
        for (int jj = 0; jj < 4; jj++)
        {
#pragma unroll
            for (int kk = 0; kk < 4; kk++)
            {
                const int param_idx = (ii * ROWS_PER_LDG + row) * 32 + (jj * 4 + kk) + (tidx & 1) * 16;
                const float bb = b[param_idx];
                const float gg = g[param_idx];
                hdata[ii * 4 + jj][kk] = gg * statsf.y * (hdata[ii * 4 + jj][kk] - statsf.x) + bb;
            }
        }
    }


#pragma unroll
    for (int ii = 0; ii < LDGS; ii++)
    {
        in_data[ii].x = pack4(hdata[ii * 4 + 0], qScale);
        in_data[ii].y = pack4(hdata[ii * 4 + 1], qScale);
        in_data[ii].z = pack4(hdata[ii * 4 + 2], qScale);
        in_data[ii].w = pack4(hdata[ii * 4 + 3], qScale);
    }

#pragma unroll
    for (int ii = 0; ii < LDGS; ii++)
    {
        if (my_pred)
        {
            stg(output + gmem_offset + ii * ROWS_PER_LDG * row_stride_bytes, in_data[ii]);
        }
    }
    // store
}

void launch_large(cudaStream_t stream, const int ld, const int total, const int8_t* input, const int8_t* skip,
    const half* beta, const half* gamma, int8_t* output, const float dqScaleIn, const float dqScaleSkip,
    const float qScale)
{
    if (ld == 1024)
    {
        constexpr int WARPS = 4;
        constexpr int THREADS_PER_ROW = 8;
        constexpr int HEADS = 16;
        constexpr int PARAM_BYTES = HEADS * 64 * 2 * sizeof(half);
        constexpr int VECS_PER_CTA = THREADS_PER_ROW / 2;
        const int blocks = (total + VECS_PER_CTA - 1) / VECS_PER_CTA;

        skipln_vec32<WARPS, HEADS, THREADS_PER_ROW><<<blocks, WARPS * 32, PARAM_BYTES, stream>>>(
            input, skip, output, beta, gamma, dqScaleIn, dqScaleSkip, qScale, total);
    }
    else if (ld == 768)
    {
        constexpr int WARPS = 3;
        constexpr int THREADS_PER_ROW = 8;
        constexpr int HEADS = 12;
        constexpr int PARAM_BYTES = HEADS * 64 * 2 * sizeof(half);
        constexpr int VECS_PER_CTA = THREADS_PER_ROW / 2;
        const int blocks = (total + VECS_PER_CTA - 1) / VECS_PER_CTA;

        skipln_vec32<WARPS, HEADS, THREADS_PER_ROW><<<blocks, WARPS * 32, PARAM_BYTES, stream>>>(
            input, skip, output, beta, gamma, dqScaleIn, dqScaleSkip, qScale, total);
    }
    else
    {
        ASSERT(false);
    }
}

// naive kernel that only changes the addressing seems to be faster for small problem sizes
template <int TPB, int VPT>
__global__ void skiplnDQQ_vec(const int ld, const int8_t* input, const int8_t* skip, int8_t* output, const half* beta,
    const half* gamma, const float dqScaleIn, const float dqScaleSkip, const float qScale, const int total)
{
    const int hinner = threadIdx.x % 4;
    const int houter = threadIdx.x / 4;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int idx = houter * total * 32 + bidx * 32 + hinner * VPT;
    // 4 * 1024 * 4 * 2 Bytes = 16KB per block
    int8_t in_local[VPT];
    int8_t skip_local[VPT];

    half in_local_dq[VPT]; // dequantized input + skip 
    half beta_local[VPT];  
    half gamma_local[VPT];

    // load input tensors
    copy<sizeof(int8_t) * VPT>(&input[idx], in_local);
    copy<sizeof(int8_t) * VPT>(&skip[idx], skip_local);

    // load parameters
    copy<sizeof(half) * VPT>(&beta[tidx * VPT], beta_local);
    copy<sizeof(half) * VPT>(&gamma[tidx * VPT], gamma_local);

    half2 stats_local = __floats2half2_rn(0.f, 0.f); // accumulator

    const half rld = half(1.f) / half(ld);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        // DQ input and skip
        const float tmp_in = in_local[it];
        const float tmp_skip = skip_local[it];
        in_local_dq[it] = dqScaleIn * tmp_in + dqScaleSkip * tmp_skip;

        const half tmp = rld * in_local_dq[it];
        const half2 tmp2 = __halves2half2(tmp, tmp * in_local_dq[it]);
        stats_local = stats_local + tmp2;
    }

    using BlockReduce = cub::BlockReduce<half2, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ half mu;     // mean
    __shared__ half rsigma; // 1 / std.dev.

    const half2 sum2 = BlockReduce(temp_storage).Reduce(stats_local, cub::Sum());

    if (tidx == 0)
    {
        mu = __low2half(sum2);
        rsigma = rsqrtf(__high2half(sum2) - mu * mu);
    }

    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        const float tmp = gamma_local[it] * (in_local_dq[it] - mu) * rsigma + beta_local[it];
        in_local[it] = quantize(tmp, qScale);
    }

    copy<sizeof(int8_t) * VPT>(in_local, &output[idx]);
}

void launch_small(cudaStream_t stream, const int ld, const int total, const int8_t* input, const int8_t* skip,
    const half* beta, const half* gamma, int8_t* output, const float dqScaleIn, const float dqScaleSkip,
    const float qScale)
{
    const int gridSize = total;
    // we align reads with the number of parameters, i.e. 8-wide instead of 16
    constexpr int VPT = 16 / sizeof(half); // 8
    if (ld == 768)
    {
        constexpr int TPB = 768 / VPT;
        skiplnDQQ_vec<TPB, VPT>
            <<<gridSize, TPB, 0, stream>>>(ld, input, skip, output, beta, gamma, dqScaleIn, dqScaleSkip, qScale, total);
    }
    else if (ld == 1024)
    {
        constexpr int TPB = 1024 / VPT; // 128
        skiplnDQQ_vec<TPB, VPT>
            <<<gridSize, TPB, 0, stream>>>(ld, input, skip, output, beta, gamma, dqScaleIn, dqScaleSkip, qScale, total);
    }
    else
    {
        std::cout << "SkipLayerNormDQQ - FATAL: unsupported hidden layer size: " << ld << std::endl;
        exit(0);
    }
    CHECK(cudaPeekAtLastError());
}

} // namespace bert

