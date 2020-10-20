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
#include <cassert>
#include <cstring>
#include <vector>

#include "NvInfer.h"
#include "bertCommon.h"
#include "common.cuh"
#include "embLayerNormPlugin.h"
#include "serialize.hpp"

using namespace nvinfer1;

namespace bert
{

__global__ void cuSeqlensToPackedMaskKernel(
    const uint32_t warps_m, const uint32_t warps_n, const uint32_t S, const int* cuSeqlens, uint32_t* inputMaskX)
{

    extern __shared__ int shm_mask[]; // S mask elements of this batch

    const size_t xmmas_n = (S + 16 * warps_n - 1) / (16 * warps_n);
    const uint32_t threads_per_cta = blockDim.x;
    const uint32_t xmmas_m = gridDim.x;

    const uint32_t mi = blockIdx.x;
    const uint32_t bi = blockIdx.y;
    const uint32_t tidx = threadIdx.x;

    const uint32_t sum_s = cuSeqlens[bi];
    const uint32_t s_b = cuSeqlens[bi + 1] - sum_s;

    const size_t warp = tidx / 32;
    const size_t warp_n = warp / warps_m;
    const size_t lane = tidx % 32;
    const size_t col = warp_n * 16 + lane % 4 * 2;

    // TODO get rid of shared mem roundtrip
    // load the mask corresponding to one batch
    for (uint32_t si = tidx; si < S; si += threads_per_cta)
    {
        shm_mask[si] = si < s_b;
    }
    __syncthreads();

    uint32_t mask = 0u;

    for (size_t ni = 0; ni < xmmas_n; ++ni)
    {
        const int offset = ni * 16 * warps_n + col;
        mask |= (shm_mask[offset + 0] == 1 ? 1u : 0u) << (8 * ni + 0);
        mask |= (shm_mask[offset + 1] == 1 ? 1u : 0u) << (8 * ni + 1);
        mask |= (shm_mask[offset + 0] == 1 ? 1u : 0u) << (8 * ni + 2);
        mask |= (shm_mask[offset + 1] == 1 ? 1u : 0u) << (8 * ni + 3);
        mask |= (shm_mask[offset + 8] == 1 ? 1u : 0u) << (8 * ni + 4);
        mask |= (shm_mask[offset + 9] == 1 ? 1u : 0u) << (8 * ni + 5);
        mask |= (shm_mask[offset + 8] == 1 ? 1u : 0u) << (8 * ni + 6);
        mask |= (shm_mask[offset + 9] == 1 ? 1u : 0u) << (8 * ni + 7);
    }

    inputMaskX[(bi * xmmas_m + mi) * threads_per_cta + tidx] = mask;
}

void cuSeqlensToPackedMask(const uint32_t S, const uint32_t B, const uint32_t warps_m, const uint32_t warps_n,
    const uint32_t warps_k, const int* cuSeqlens, uint32_t* inputMaskX, cudaStream_t stream)
{
    const size_t xmmas_m = (S + 16 * warps_m - 1) / (16 * warps_m);

    const size_t threads_per_cta = warps_m * warps_n * warps_k * 32;
    dim3 grid(xmmas_m, B);
    cuSeqlensToPackedMaskKernel<<<grid, threads_per_cta, S * sizeof(int), stream>>>(
        warps_m, warps_n, S, cuSeqlens, inputMaskX);
    CHECK(cudaPeekAtLastError());
}

template <typename T, unsigned TPB, unsigned VPT>
__global__ void embLayerNormKernelVarSeqlen(int ld, const uint32_t* cuSeqlens, const int* inputIds,
    const int* segmentIds, const T* beta, const T* gamma, const T* tokEmb, const T* posEmb, const T* segEmb, T* output)
{

    using BlockReduce = cub::BlockReduce<kvp<T>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int b = blockIdx.x;
    const int s = blockIdx.y;

    const int sum_s = cuSeqlens[b];
    const int s_b = cuSeqlens[b + 1] - sum_s;

    // either the whole CTA has work or not
    if (s >= s_b)
        return;

    const int inOffset = (sum_s + s);
    const int outOffset = (sum_s + s) * ld;

    // 1. lookup word and token of the block
    // blockIdx.x = position in the sequence
    // blockIdx.y = batch
    // gridDim.x = S
    // gridDim.y = B
    __shared__ int inputId;
    __shared__ int segmentId;

    if (threadIdx.x == 0)
    {
        inputId = inputIds[inOffset];
        segmentId = segmentIds[inOffset];
    }
    __syncthreads();

    // 2. load pos/tok/word embeddings and add them toghether
    // offset into embeddings is given by wordId * hidden_size
    const int poffset = s * ld;
    const int ioffset = inputId * ld;
    const int soffset = segmentId * ld;

    // 16B per thread: 8 elements. there should be ld / VPT threads per CTA
    // 1024: 128 threads
    // 768: 96 threads
    const int toffset = threadIdx.x * VPT;
    // 4 * 1024 * 4 * 2 Bytes = 16KB per block
    T i_local[VPT];
    T s_local[VPT];
    T p_local[VPT];

    // read embeddings
    copy<sizeof(T) * VPT>(&tokEmb[ioffset + toffset], i_local);
    copy<sizeof(T) * VPT>(&segEmb[soffset + toffset], s_local);
    copy<sizeof(T) * VPT>(&posEmb[poffset + toffset], p_local);
    T local = 0.f;
    T local2 = 0.f;

    const T rld = T(1) / T(ld);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        i_local[it] += s_local[it] + p_local[it];
        const T tmp = rld * i_local[it];
        local += tmp;
        local2 += tmp * i_local[it];
    }

    // load params
    copy<sizeof(T) * VPT>(&beta[toffset], p_local);
    copy<sizeof(T) * VPT>(&gamma[toffset], s_local);

    __shared__ T mu;     // mean
    __shared__ T rsigma; // 1 / std.dev.

    const auto sumKV = BlockReduce(temp_storage).Reduce(kvp<T>(local, local2), cub::Sum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu);
    }
    __syncthreads();
    ///*
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        i_local[it] = s_local[it] * (i_local[it] - mu) * rsigma + p_local[it];
    }
    /* */

    copy<sizeof(T) * VPT>(i_local, &output[outOffset + toffset]);
}

template <typename T>
int embSkipLayerNormVarSeqlen(cudaStream_t stream, int ld, int B, int S, const uint32_t* cuSeqlens, const int* inputIds,
    const int* token_ids, const T* beta, const T* gamma, const T* wordEmb, const T* posEmb, const T* tokEmb, T* output)
{

    const dim3 grid(B, S, 1);

    if (ld == 1024)
    {
        constexpr int VPT = 16 / sizeof(T);
        constexpr int TPB = 1024 / VPT;
        const dim3 block(TPB, 1, 1);
        embLayerNormKernelVarSeqlen<T, TPB, VPT><<<grid, block, 0, stream>>>(
            ld, cuSeqlens, inputIds, token_ids, beta, gamma, wordEmb, posEmb, tokEmb, output);
    }
    else if (ld == 768)
    {
        constexpr int VPT = 16 / sizeof(T);
        constexpr int TPB = 768 / VPT;
        const dim3 block(TPB, 1, 1);
        embLayerNormKernelVarSeqlen<T, TPB, VPT><<<grid, block, 0, stream>>>(
            ld, cuSeqlens, inputIds, token_ids, beta, gamma, wordEmb, posEmb, tokEmb, output);
    }
    else
    {
        assert(false && "Unsupported hidden dimension");
    }

    CHECK(cudaPeekAtLastError());

    return 0;
}

template int embSkipLayerNormVarSeqlen<float>(cudaStream_t, int, int, int, const uint32_t*, const int*, const int*,
    const float*, const float*, const float*, const float*, const float*, float*);

template int embSkipLayerNormVarSeqlen<half>(cudaStream_t, int, int, int, const uint32_t*, const int*, const int*,
    const half*, const half*, const half*, const half*, const half*, half*);

/// REDO BASED ON OLD KERNEL TO REPRODUCE EXACT RESULTS

template <typename T, unsigned TPB>
__global__ void embLayerNormKernel2(int ld, const int* inputIds, const int* tokenIds, const int* cuSeqlens,
    const float* beta, const float* gamma, const T* wordEmb, const T* posEmb, const T* tokEmb, T* output)
{
    // this code currently assumes the input shape is SxB, row-major => seqPos = s * B + b
    // instead we want BxS, row-major => seqPos = b * S + s

    cub::Sum pairSum;
    // 1. lookup word and token of the block
    // blockIdx.x = position in the sequence
    // blockIdx.y = batch
    // gridDim.x = S
    // gridDim.y = B
    const int s = blockIdx.x;
    const int b = blockIdx.y;

    const int sumS = cuSeqlens[b];
    const int s_b = cuSeqlens[b + 1] - sumS;
    if (s >= s_b)
        return; // This CTA has nothing to do
    __shared__ int wordId;
    __shared__ int tokenId;

    const T rld = T(1.f) / T(ld);
    // seqPos = b + s * B
    // const int seqPos = blockIdx.y + blockIdx.x * gridDim.y;

    // const int seqPos = s * B + s;
    const int seqPos = sumS + s;
    if (threadIdx.x == 0)
    {
        wordId = inputIds[seqPos];
        tokenId = tokenIds[seqPos];
    }
    __syncthreads();

    // 2. load pos/tok/word embeddings and add them toghether
    // offset into embeddings is given by wordId * hidden_size
    const int poffset = blockIdx.x * ld;
    const int woffset = wordId * ld;
    const int toffset = tokenId * ld;
    // the output offset is given by b * (S*hidden_size) + s * hidden_size
    const int outOffset = seqPos * ld;

    kvp<T> threadData(0, 0);

    for (int it = threadIdx.x; it < ld; it += TPB)
    {
        const T w(wordEmb[woffset + it]);
        const T t(tokEmb[toffset + it]);
        const T p(posEmb[poffset + it]);
        const T val = w + t + p;

        output[outOffset + it] = val;
        const T rldval = rld * val;
        threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
    }

    // 3. layer norm on the sum
    layerNorm<T, T, float, TPB>(threadData, ld, outOffset, beta, gamma, output);
}

template <typename T>
int embSkipLayerNorm2(cudaStream_t stream, int ld, int B, int S, const int* inputIds, const int* tokenIds,
    const int* cuSeqlens, const float* beta, const float* gamma, const T* wordEmb, const T* posEmb, const T* tokEmb,
    T* output)
{

    constexpr int tpb = 256;
    const dim3 grid(S, B, 1);
    const dim3 block(tpb, 1, 1);

    embLayerNormKernel2<T, tpb>
        <<<grid, block, 0, stream>>>(ld, inputIds, tokenIds, cuSeqlens, beta, gamma, wordEmb, posEmb, tokEmb, output);
    CHECK(cudaPeekAtLastError());

    return 0;
}

template int embSkipLayerNorm2<float>(cudaStream_t, int, int, int, const int*, const int*, const int*, const float*,
    const float*, const float*, const float*, const float*, float*);

template int embSkipLayerNorm2<half>(cudaStream_t, int, int, int, const int*, const int*, const int*, const float*,
    const float*, const half*, const half*, const half*, half*);

} // namespace bert
