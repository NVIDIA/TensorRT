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

#include <cassert>
#include <cstring>
#include <vector>

#include "NvInfer.h"
#include "common/bertCommon.h"
#include "common/common.cuh"
#include "common/serialize.hpp"
#include "embLayerNormPlugin.h"

using namespace nvinfer1;

namespace nvinfer1
{
namespace plugin
{
namespace bert
{

__global__ void fillSBSMaskKernel(
    uint32_t const warps_m, uint32_t const warps_n, uint32_t const S, int32_t const* inputMaskSB, uint32_t* inputMaskX)
{

    extern __shared__ int shm_mask[]; // S mask elements of this batch

    size_t const xmmas_n = (S + 16 * warps_n - 1) / (16 * warps_n);
    uint32_t const threads_per_cta = blockDim.x;
    uint32_t const xmmas_m = gridDim.x;
    uint32_t const B = gridDim.y;

    uint32_t const mi = blockIdx.x;
    uint32_t const bi = blockIdx.y;
    uint32_t const tidx = threadIdx.x;

    size_t const warp = tidx / 32;
    size_t const warp_n = warp / warps_m;
    size_t const lane = tidx % 32;
    size_t const col = warp_n * 16 + lane % 4 * 2;

    // load the mask corresponding to one batch
    for (uint32_t si = tidx; si < S; si += threads_per_cta)
    {
        // not coalesced to conform to current input format: SxB
        shm_mask[si] = inputMaskSB[si * B + bi];
    }
    __syncthreads();

    uint32_t mask = 0u;

    for (size_t ni = 0; ni < xmmas_n; ++ni)
    {
        int32_t const offset = ni * 16 * warps_n + col;
        mask |= (shm_mask[offset + 0] == 1.f ? 1u : 0u) << (8 * ni + 0);
        mask |= (shm_mask[offset + 1] == 1.f ? 1u : 0u) << (8 * ni + 1);
        mask |= (shm_mask[offset + 0] == 1.f ? 1u : 0u) << (8 * ni + 2);
        mask |= (shm_mask[offset + 1] == 1.f ? 1u : 0u) << (8 * ni + 3);
        mask |= (shm_mask[offset + 8] == 1.f ? 1u : 0u) << (8 * ni + 4);
        mask |= (shm_mask[offset + 9] == 1.f ? 1u : 0u) << (8 * ni + 5);
        mask |= (shm_mask[offset + 8] == 1.f ? 1u : 0u) << (8 * ni + 6);
        mask |= (shm_mask[offset + 9] == 1.f ? 1u : 0u) << (8 * ni + 7);
    }

    inputMaskX[(bi * xmmas_m + mi) * threads_per_cta + tidx] = mask;
}

cudaError_t convertMask(uint32_t const S, uint32_t const B, uint32_t const warps_m, uint32_t const warps_n,
    uint32_t const warps_k, int32_t const* inputMaskSB, uint32_t* inputMaskX, cudaStream_t stream)
{
    size_t const xmmas_m = (S + 16 * warps_m - 1) / (16 * warps_m);

    size_t const threads_per_cta = warps_m * warps_n * warps_k * 32;
    dim3 grid(xmmas_m, B);
    fillSBSMaskKernel<<<grid, threads_per_cta, S * sizeof(int), stream>>>(warps_m, warps_n, S, inputMaskSB, inputMaskX);
    return cudaPeekAtLastError();
}

template <unsigned TPB>
__global__ void maskIdxKernelSmall(int ld, int32_t const* mask, int* maskIdx)
{

    using BlockReduce = cub::BlockReduce<int32_t, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub::Min min;
    int threadData(ld); // if the mask admits all values

    if (threadIdx.x < ld)
    {
        // mask has input dims {S, B} and gridDims.x is B
        int32_t const idx = threadIdx.x * gridDim.x + blockIdx.x;

        int32_t const val = mask[idx];
        if (val == 0) // masked position: report thread idx
        {
            threadData = threadIdx.x;
        }
    }

    const auto minIdx = BlockReduce(tmpStorage).Reduce(threadData, min);

    if (threadIdx.x == 0)
    {
        maskIdx[blockIdx.x] = minIdx;
    }
}

template <unsigned TPB>
__global__ void maskIdxKernel(int ld, int32_t const* mask, int* maskIdx)
{

    using BlockReduce = cub::BlockReduce<int32_t, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub::Min min;
    int threadData(ld); // if the mask admits all values

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        // mask has input dims {S, B} and gridDims.x is B
        int32_t const idx = i * gridDim.x + blockIdx.x;

        int32_t const val = mask[idx];
        if (val == 0) // masked position: report thread idx
        {
            threadData = min(threadData, i);
        }
    }

    const auto minIdx = BlockReduce(tmpStorage).Reduce(threadData, min);

    if (threadIdx.x == 0)
    {
        maskIdx[blockIdx.x] = minIdx;
    }
}

int computeMaskIdx(cudaStream_t stream, int32_t const S, int32_t const B, int32_t const* mask, int* maskIdx)
{
    // Mask idx is of length B and assumes the valid region is contiguous starting
    // from the beginning of the sequence

    // Assume n = BxS
    if (S <= 32)
    {
        maskIdxKernelSmall<32><<<B, 32, 0, stream>>>(S, mask, maskIdx);
    }
    else if (S <= 128)
    {
        maskIdxKernelSmall<128><<<B, 128, 0, stream>>>(S, mask, maskIdx);
    }
    else if (S == 384)
    {
        maskIdxKernelSmall<384><<<B, 384, 0, stream>>>(S, mask, maskIdx);
    }
    else
    {
        maskIdxKernel<256><<<B, 256, 0, stream>>>(S, mask, maskIdx);
    }

    return cudaPeekAtLastError();
}

template <typename T, unsigned TPB>
__global__ void embLayerNormKernel(int ld, int32_t const* inputIds, int32_t const* tokenIds, float const* beta,
    float const* gamma, T const* wordEmb, T const* posEmb, T const* tokEmb, int32_t const wordSize,
    int32_t const tokSize, T* output)
{

    cub::Sum pairSum;
    // 1. lookup word and token of the block
    // blockIdx.x = position in the sequence
    // blockIdx.y = batch
    // gridDim.x = S
    // gridDim.y = B
    __shared__ int wordId;
    __shared__ int tokenId;

    T const rld = T(1.f) / T(ld);
    int32_t const seqPos = blockIdx.y + blockIdx.x * gridDim.y;
    if (threadIdx.x == 0)
    {
        wordId = inputIds[seqPos];
        tokenId = tokenIds[seqPos];
    }
    __syncthreads();

    // 2. load pos/tok/word embeddings and add them toghether
    // offset into embeddings is given by wordId * hidden_size
    int32_t const poffset = blockIdx.x * ld;
    int32_t const woffset = wordId * ld;
    int32_t const toffset = tokenId * ld;
    // the output offset is given by b * (S*hidden_size) + s * hidden_size
    int32_t const outOffset = seqPos * ld;

    kvp<T> threadData(0, 0);

    if (wordId >= 0 && wordId < wordSize && tokenId >= 0 && tokenId < tokSize)
    {
        for (int it = threadIdx.x; it < ld; it += TPB)
        {
            T const w(wordEmb[woffset + it]);
            T const t(tokEmb[toffset + it]);
            T const p(posEmb[poffset + it]);
            T const val = w + t + p;

            output[outOffset + it] = val;
            T const rldval = rld * val;
            threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
        }
    }

    // 3. layer norm on the sum
    layerNorm<T, T, float, TPB>(threadData, ld, outOffset, beta, gamma, output);
}

template <typename T>
int embSkipLayerNorm(cudaStream_t stream, int ld, int B, int S, int32_t const* inputIds, int32_t const* tokenIds,
    float const* beta, float const* gamma, T const* wordEmb, T const* posEmb, T const* tokEmb, int32_t const wordSize,
    int32_t const tokSize, T* output)
{

    constexpr int tpb = 256;
    dim3 const grid(S, B, 1);
    dim3 const block(tpb, 1, 1);

    embLayerNormKernel<T, tpb><<<grid, block, 0, stream>>>(
        ld, inputIds, tokenIds, beta, gamma, wordEmb, posEmb, tokEmb, wordSize, tokSize, output);
    PLUGIN_CHECK(cudaPeekAtLastError());

    return 0;
}

template int embSkipLayerNorm<float>(cudaStream_t, int32_t, int32_t, int32_t, int32_t const*, int32_t const*,
    float const*, float const*, float const*, float const*, float const*, int32_t const, int32_t const, float*);

template int embSkipLayerNorm<half>(cudaStream_t, int32_t, int32_t, int32_t, int32_t const*, int32_t const*,
    float const*, float const*, half const*, half const*, half const*, int32_t const, int32_t const, half*);

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // CUDA_VERSION >= 10010
