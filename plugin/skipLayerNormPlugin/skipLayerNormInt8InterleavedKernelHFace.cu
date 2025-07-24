/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include "NvInfer.h"
#include "common/bertCommon.h"
#include "common/common.cuh"
#include "common/cubCcclCompat.h"
#include <cassert>
#include <cstring>
#include <cuda.h>
#include <type_traits>
#include <vector>

using namespace nvinfer1;

namespace nvinfer1
{
namespace plugin
{
namespace bert
{

inline __device__ void resAdd(
    float (&hdata)[4], const uint32_t idata, const uint32_t ires, float const dqData, float const dqRes)
{
    char4 ires4 = reinterpret_cast<char4 const&>(ires);
    char4 idata4 = reinterpret_cast<char4 const&>(idata);
    hdata[0] = float(idata4.x) * dqData + float(ires4.x) * dqRes;
    hdata[1] = float(idata4.y) * dqData + float(ires4.y) * dqRes;
    hdata[2] = float(idata4.z) * dqData + float(ires4.z) * dqRes;
    hdata[3] = float(idata4.w) * dqData + float(ires4.w) * dqRes;
}

template <int32_t tWARPS, int32_t tHEADS, int32_t tTHREADS_PER_ROW>
__global__ void skipln_vec32_hface(int8_t const* input, int8_t const* skip, int8_t* output, half const* beta,
    half const* gamma, float const dqScaleIn, float const dqScaleSkip, float const qScale, int32_t const total)
{

    // clang-format off
    enum { kHEAD_SIZE = 64 };
    enum { kBYTES_PER_LDG = 16 };
    enum { kTHREADS_PER_CTA = tWARPS * 32 };
    enum { kROWS_PER_LDG = kTHREADS_PER_CTA / tTHREADS_PER_ROW };
    enum { kVECS_PER_CTA = tTHREADS_PER_ROW / 2 };
    enum { kPARAM_BYTES = tHEADS * kHEAD_SIZE * 2 };
    enum { kPARAM_LDGS = kPARAM_BYTES / (kTHREADS_PER_CTA * kBYTES_PER_LDG) };
    enum { kLDGS = tHEADS * 2 / kROWS_PER_LDG };
    // clang-format on
    static_assert(kVECS_PER_CTA == 4, "");
    static_assert(kPARAM_LDGS == 1, "");
    static_assert(kROWS_PER_LDG == tHEADS, "");
    static_assert(kLDGS == 2, "");
    static_assert(kLDGS * kROWS_PER_LDG == tHEADS * 2, "");
    static_assert(kTHREADS_PER_CTA * kBYTES_PER_LDG == kPARAM_BYTES, "");
    static_assert(kPARAM_LDGS == 1, "");

    extern __shared__ char smem_[];

    // space for CTA-wide reduction
    __shared__ half2 smemRed[kVECS_PER_CTA][tWARPS];

    constexpr float rld = 1.F / (float(tHEADS) * float(kHEAD_SIZE));
    int32_t const bidx = blockIdx.x;
    int32_t const tidx = threadIdx.x;
    int32_t const row = tidx / tTHREADS_PER_ROW;
    int32_t const col = tidx % tTHREADS_PER_ROW;
    int32_t const lane = tidx % 32;
    int32_t const warp = tidx / 32;

    bool const isWarpLead = (lane < tTHREADS_PER_ROW) && ((lane & 1) == 0);
    bool const isCtaLead = (tidx < tTHREADS_PER_ROW) && ((tidx & 1) == 0);

    // token position: every two threads load together the 32B at one token
    // position
    int32_t const pos = col / 2;

    int32_t const posOffset = bidx * kVECS_PER_CTA + pos; // for token positions per block, disabling 2 threads per pos
    bool const myPred = posOffset < total;
    int32_t const rowStrideBytes = total * 32;

    uint4 inData[kLDGS];
    uint4 inSkip[kLDGS];
    float hdata[kLDGS * 4][4];
    int32_t const gmemOffset = row * rowStrideBytes + (bidx * tTHREADS_PER_ROW + col) * kBYTES_PER_LDG;
#pragma unroll
    for (int32_t ii = 0; ii < kLDGS; ii++)
    {
        inData[ii] = {0, 0, 0, 0};
        inSkip[ii] = {0, 0, 0, 0};
        if (myPred)
        {
            ldg(input + gmemOffset + ii * kROWS_PER_LDG * rowStrideBytes, inData[ii]);
            ldg(skip + gmemOffset + ii * kROWS_PER_LDG * rowStrideBytes, inSkip[ii]);
        }
    }

    uint4* smemB = reinterpret_cast<uint4*>(&smem_[0]) + tidx;
    uint4* smemG = reinterpret_cast<uint4*>(&smem_[kPARAM_BYTES]) + tidx;

    int8_t const* betaPtr = reinterpret_cast<int8_t const*>(beta) + tidx * kBYTES_PER_LDG;
    int8_t const* gammaPtr = reinterpret_cast<int8_t const*>(gamma) + tidx * kBYTES_PER_LDG;
    ldg(betaPtr, *smemB);
    ldg(gammaPtr, *smemG);

    half* b = reinterpret_cast<half*>(&smem_[0]);
    half* g = reinterpret_cast<half*>(&smem_[kPARAM_BYTES]);
#pragma unroll
    for (int32_t ii = 0; ii < kLDGS; ii++)
    {
        resAdd(hdata[ii * 4 + 0], inData[ii].x, inSkip[ii].x, dqScaleIn, dqScaleSkip);
        resAdd(hdata[ii * 4 + 1], inData[ii].y, inSkip[ii].y, dqScaleIn, dqScaleSkip);
        resAdd(hdata[ii * 4 + 2], inData[ii].z, inSkip[ii].z, dqScaleIn, dqScaleSkip);
        resAdd(hdata[ii * 4 + 3], inData[ii].w, inSkip[ii].w, dqScaleIn, dqScaleSkip);
    }

    half2 statsLocal = {0, 0};

#pragma unroll
    for (int32_t ii = 0; ii < kLDGS * 4; ii++)
    {
#pragma unroll
        for (int32_t jj = 0; jj < 4; jj++)
        {
            float const tmp = hdata[ii][jj] * (rld);
            statsLocal = statsLocal + __floats2half2_rn(tmp, tmp * hdata[ii][jj]);
        }
    }
    statsLocal = statsLocal + __shfl_xor_sync(uint32_t(-1), statsLocal, 1);
    __syncwarp();

    if (kVECS_PER_CTA == 1)
    {
        statsLocal = statsLocal + __shfl_xor_sync(uint32_t(-1), statsLocal, 2);
        __syncwarp();
        statsLocal = statsLocal + __shfl_xor_sync(uint32_t(-1), statsLocal, 4);
        __syncwarp();
    }
    else if (kVECS_PER_CTA == 2)
    {
        statsLocal = statsLocal + __shfl_xor_sync(uint32_t(-1), statsLocal, 4);
        __syncwarp();
    }

    statsLocal = statsLocal + __shfl_xor_sync(uint32_t(-1), statsLocal, 8);
    __syncwarp();
    statsLocal = statsLocal + __shfl_xor_sync(uint32_t(-1), statsLocal, 16);
    __syncwarp();

    if (isWarpLead)
    {
        smemRed[pos][warp] = statsLocal;
    }

    __syncthreads();

    if (isCtaLead)
    {
        for (int32_t ii = 1; ii < tWARPS; ii++)
        {
            statsLocal = statsLocal + smemRed[pos][ii];
        }

        float mu = __low2float(statsLocal);
        float sos = __high2float(statsLocal);
        float rsigma = rsqrtf(sos - mu * mu + std::numeric_limits<float>::epsilon());

        smemRed[pos][0] = __floats2half2_rn(mu, rsigma);
    }
    __syncthreads();
    // load params into smem:  2x Headsx32x2x2B
    const float2 statsf = __half22float2(smemRed[pos][0]);

#pragma unroll
    for (int32_t ii = 0; ii < kLDGS; ii++)
    {
#pragma unroll
        for (int32_t jj = 0; jj < 4; jj++)
        {
#pragma unroll
            for (int32_t kk = 0; kk < 4; kk++)
            {
                int32_t const paramIdx = (ii * kROWS_PER_LDG + row) * 32 + (jj * 4 + kk) + (tidx & 1) * 16;
                float const bb = b[paramIdx];
                float const gg = g[paramIdx];
                hdata[ii * 4 + jj][kk] = gg * statsf.y * (hdata[ii * 4 + jj][kk] - statsf.x) + bb;
            }
        }
    }

#pragma unroll
    for (int32_t ii = 0; ii < kLDGS; ii++)
    {
        inData[ii].x = pack4(hdata[ii * 4 + 0], qScale);
        inData[ii].y = pack4(hdata[ii * 4 + 1], qScale);
        inData[ii].z = pack4(hdata[ii * 4 + 2], qScale);
        inData[ii].w = pack4(hdata[ii * 4 + 3], qScale);
    }

#pragma unroll
    for (int32_t ii = 0; ii < kLDGS; ii++)
    {
        if (myPred)
        {
            stg(output + gmemOffset + ii * kROWS_PER_LDG * rowStrideBytes, inData[ii]);
        }
    }
    // store
}

int32_t launch_large_hface(cudaStream_t stream, int32_t const ld, int32_t const total, int8_t const* input,
    int8_t const* skip, half const* beta, half const* gamma, int8_t* output, float const dqScaleIn,
    float const dqScaleSkip, float const qScale)
{
    if (ld == 1024)
    {
        constexpr int32_t tWARPS = 4;
        constexpr int32_t tTHREADS_PER_ROW = 8;
        constexpr int32_t tHEADS = 16;
        constexpr int32_t kPARAM_BYTES = tHEADS * 64 * 2 * sizeof(half);
        constexpr int32_t kVECS_PER_CTA = tTHREADS_PER_ROW / 2;
        int32_t const blocks = (total + kVECS_PER_CTA - 1) / kVECS_PER_CTA;

        skipln_vec32_hface<tWARPS, tHEADS, tTHREADS_PER_ROW><<<blocks, tWARPS * 32, kPARAM_BYTES, stream>>>(
            input, skip, output, beta, gamma, dqScaleIn, dqScaleSkip, qScale, total);
    }
    else if (ld == 768)
    {
        constexpr int32_t tWARPS = 3;
        constexpr int32_t tTHREADS_PER_ROW = 8;
        constexpr int32_t tHEADS = 12;
        constexpr int32_t kPARAM_BYTES = tHEADS * 64 * 2 * sizeof(half);
        constexpr int32_t kVECS_PER_CTA = tTHREADS_PER_ROW / 2;
        int32_t const blocks = (total + kVECS_PER_CTA - 1) / kVECS_PER_CTA;

        skipln_vec32_hface<tWARPS, tHEADS, tTHREADS_PER_ROW><<<blocks, tWARPS * 32, kPARAM_BYTES, stream>>>(
            input, skip, output, beta, gamma, dqScaleIn, dqScaleSkip, qScale, total);
    }
    else
    {
        return STATUS_FAILURE;
    }

    return cudaPeekAtLastError();
}

// naive kernel that only changes the addressing seems to be faster for small
// problem sizes
template <int32_t TPB, int32_t VPT>
__global__ void skiplnDQQ_vec3(int32_t const ld, int8_t const* input, int8_t const* skip, int8_t* output,
    half const* beta, half const* gamma, float const dqScaleIn, float const dqScaleSkip, float const qScale,
    int32_t const total)
{
    int32_t const hinner = threadIdx.x % 4;
    int32_t const houter = threadIdx.x / 4;

    int32_t const tidx = threadIdx.x;
    int32_t const bidx = blockIdx.x;
    int32_t const idx = houter * total * 32 + bidx * 32 + hinner * VPT;
    // 4 * 1024 * 4 * 2 Bytes = 16KB per block
    int8_t inLocal[VPT];
    int8_t skipLocal[VPT];

    half inLocalDQ[VPT]; // dequantized input + skip
    half betaLocal[VPT];
    half gammaLocal[VPT];

    // load input tensors
    copy<sizeof(int8_t) * VPT>(&input[idx], inLocal);
    copy<sizeof(int8_t) * VPT>(&skip[idx], skipLocal);

    // load parameters
    copy<sizeof(half) * VPT>(&beta[tidx * VPT], betaLocal);
    copy<sizeof(half) * VPT>(&gamma[tidx * VPT], gammaLocal);

    half2 statsLocal = __floats2half2_rn(0.F, 0.F); // accumulator

    half const rld = half(1.F) / half(ld);
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        // DQ input and skip
        float const tmpIn = inLocal[it];
        float const tmpSkip = skipLocal[it];
        inLocalDQ[it] = dqScaleIn * tmpIn + dqScaleSkip * tmpSkip;

        half const tmp = rld * inLocalDQ[it];
        half2 const tmp2 = __halves2half2(tmp, tmp * inLocalDQ[it]);
        statsLocal = statsLocal + tmp2;
    }

    using BlockReduce = cub::BlockReduce<half2, TPB>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ half mu;     // mean
    __shared__ half rsigma; // 1 / std.dev.

    half2 const sum2 = BlockReduce(tempStorage).Reduce(statsLocal, compat::getCudaSumOp());

    if (tidx == 0)
    {
        mu = __low2half(sum2);
        rsigma = rsqrtf(__high2half(sum2) - mu * mu + std::numeric_limits<half>::epsilon());
    }

    __syncthreads();

    static_assert(VPT % 4 == 0, "");
    uint32_t outLocal[VPT / 4];
#pragma unroll
    for (int32_t it = 0; it < VPT / 4; it++)
    {
        float const tmp0 = gammaLocal[it * 4 + 0] * (inLocalDQ[it * 4 + 0] - mu) * rsigma + betaLocal[it * 4 + 0];
        float const tmp1 = gammaLocal[it * 4 + 1] * (inLocalDQ[it * 4 + 1] - mu) * rsigma + betaLocal[it * 4 + 1];
        float const tmp2 = gammaLocal[it * 4 + 2] * (inLocalDQ[it * 4 + 2] - mu) * rsigma + betaLocal[it * 4 + 2];
        float const tmp3 = gammaLocal[it * 4 + 3] * (inLocalDQ[it * 4 + 3] - mu) * rsigma + betaLocal[it * 4 + 3];
        outLocal[it] = float4_to_char4(tmp0 * qScale, tmp1 * qScale, tmp2 * qScale, tmp3 * qScale);
    }

    copy<sizeof(int8_t) * VPT>(outLocal, &output[idx]);
}

int32_t launch_small_hface(cudaStream_t stream, int32_t const ld, int32_t const total, int8_t const* input,
    int8_t const* skip, half const* beta, half const* gamma, int8_t* output, float const dqScaleIn,
    float const dqScaleSkip, float const qScale)
{
    int32_t const gridSize = total;
    // we align reads with the number of parameters, i.e. 8-wide instead of 16
    constexpr int32_t VPT = 16 / sizeof(half); // 8
    if (ld == 768)
    {
        constexpr int32_t TPB = 768 / VPT;
        skiplnDQQ_vec3<TPB, VPT>
            <<<gridSize, TPB, 0, stream>>>(ld, input, skip, output, beta, gamma, dqScaleIn, dqScaleSkip, qScale, total);
    }
    else if (ld == 1024)
    {
        constexpr int32_t TPB = 1024 / VPT; // 128
        skiplnDQQ_vec3<TPB, VPT>
            <<<gridSize, TPB, 0, stream>>>(ld, input, skip, output, beta, gamma, dqScaleIn, dqScaleSkip, qScale, total);
    }
    else
    {
        std::cout << "SkipLayerNormDQQ - FATAL: unsupported hidden layer size: " << ld << std::endl;
        return STATUS_FAILURE;
    }
    return cudaPeekAtLastError();
}

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
