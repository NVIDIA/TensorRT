/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#if CUDA_VERSION >= 10010

#include <cassert>
#include <cstring>
#include <vector>

#include "NvInfer.h"
#include "embLayerNormPlugin.h"
#include "bertCommon.h"
#include "common.cuh"
#include "serialize.hpp"

using namespace nvinfer1;

namespace bert
{

__global__ void fillSBSMaskKernel(
    const uint32_t warps_m, const uint32_t warps_n, const uint32_t S, const int* inputMaskSB, uint32_t* inputMaskX)
{

    extern __shared__ int shm_mask[]; // S mask elements of this batch

    const size_t xmmas_n = (S + 16 * warps_n - 1) / (16 * warps_n);
    const uint32_t threads_per_cta = blockDim.x;
    const uint32_t xmmas_m = gridDim.x;
    const uint32_t B = gridDim.y;

    const uint32_t mi = blockIdx.x;
    const uint32_t bi = blockIdx.y;
    const uint32_t tidx = threadIdx.x;

    const size_t warp = tidx / 32;
    const size_t warp_n = warp / warps_m;
    const size_t lane = tidx % 32;
    const size_t col = warp_n * 16 + lane % 4 * 2;

    //load the mask corresponding to one batch
    for (uint32_t si = tidx; si < S; si += threads_per_cta)
    {
        // not coalesced to conform to current input format: SxB
        shm_mask[si] = inputMaskSB[si * B + bi];
    }
    __syncthreads();

    uint32_t mask = 0u;

    for (size_t ni = 0; ni < xmmas_n; ++ni)
    {
        const int offset = ni * 16 * warps_n + col;
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

cudaError_t convertMask(const uint32_t S, const uint32_t B, const uint32_t warps_m, const uint32_t warps_n,
    const uint32_t warps_k, const int* inputMaskSB, uint32_t* inputMaskX, cudaStream_t stream)
{
    const size_t xmmas_m = (S + 16 * warps_m - 1) / (16 * warps_m);

    const size_t threads_per_cta = warps_m * warps_n * warps_k * 32;
    dim3 grid(xmmas_m, B);
    fillSBSMaskKernel<<<grid, threads_per_cta, S * sizeof(int), stream>>>(warps_m, warps_n, S, inputMaskSB, inputMaskX);
    return cudaPeekAtLastError();
}

template <unsigned TPB>
__global__ void maskIdxKernelSmall(int ld, const int* mask, int* maskIdx)
{

    using BlockReduce = cub::BlockReduce<int, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub::Min min;
    int threadData(ld); // if the mask admits all values

    if (threadIdx.x < ld)
    {
        // mask has input dims {S, B} and gridDims.x is B
        const int idx = threadIdx.x * gridDim.x + blockIdx.x;

        const int val = mask[idx];
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
__global__ void maskIdxKernel(int ld, const int* mask, int* maskIdx)
{

    using BlockReduce = cub::BlockReduce<int, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub::Min min;
    int threadData(ld); // if the mask admits all values

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        // mask has input dims {S, B} and gridDims.x is B
        const int idx = i * gridDim.x + blockIdx.x;

        const int val = mask[idx];
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

int computeMaskIdx(cudaStream_t stream, const int S, const int B, const int* mask, int* maskIdx)
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
__global__ void embLayerNormKernel(int ld, const int* inputIds, const int* tokenIds, const float* beta,
    const float* gamma, const T* wordEmb, const T* posEmb, const T* tokEmb, T* output)
{

    cub::Sum pairSum;
    // 1. lookup word and token of the block
    // blockIdx.x = position in the sequence
    // blockIdx.y = batch
    // gridDim.x = S
    // gridDim.y = B
    __shared__ int wordId;
    __shared__ int tokenId;

    const T rld = T(1.f) / T(ld);
    const int seqPos = blockIdx.y + blockIdx.x * gridDim.y;
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
int embSkipLayerNorm(cudaStream_t stream, int ld, int B, int S, const int* inputIds, const int* token_ids,
    const float* beta, const float* gamma, const T* wordEmb, const T* posEmb, const T* tokEmb, T* output)
{

    constexpr int tpb = 256;
    const dim3 grid(S, B, 1);
    const dim3 block(tpb, 1, 1);

    embLayerNormKernel<T, tpb>
        <<<grid, block, 0, stream>>>(ld, inputIds, token_ids, beta, gamma, wordEmb, posEmb, tokEmb, output);
    CHECK(cudaPeekAtLastError());

    return 0;
}

template int embSkipLayerNorm<float>(cudaStream_t, int, int, int, const int*, const int*, const float*, const float*,
    const float*, const float*, const float*, float*);

template int embSkipLayerNorm<half>(cudaStream_t, int, int, int, const int*, const int*, const float*, const float*,
    const half*, const half*, const half*, half*);

} // namespace bert

#endif // CUDA_VERSION >= 10010
