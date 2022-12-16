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

#include "NvInfer.h"
#include "common/bertCommon.h"
#include "common/common.cuh"
#include "common/plugin.h"
#include "common/serialize.hpp"

#include <cassert>
#include <cstring>
#include <cuda.h>
#include <vector>

using namespace nvinfer1;

namespace nvinfer1
{
namespace plugin
{
namespace bert
{

template <typename T, unsigned TPB>
__global__ void embLayerNormKernelHFace(int32_t ld, int32_t const* inputIds, int32_t const* tokenIds,
    int32_t const* cuSeqlens, float const* beta, float const* gamma, T const* wordEmb, T const* posEmb, T const* tokEmb,
    int32_t const wordSize, int32_t const tokSize, T* output)
{
    // this code currently assumes the input shape is SxB, row-major => seqPos = s * B + b
    // instead we want BxS, row-major => seqPos = b * S + s

    cub::Sum pairSum;
    // 1. lookup word and token of the block
    // blockIdx.x = position in the sequence
    // blockIdx.y = batch
    // gridDim.x = S
    // gridDim.y = B
    int32_t const s = blockIdx.x;
    int32_t const b = blockIdx.y;

    int32_t const sumS = cuSeqlens[b];
    int32_t const s_b = cuSeqlens[b + 1] - sumS;
    if (s >= s_b)
    {
        return; // This CTA has nothing to do
    }
    __shared__ int32_t wordId;
    __shared__ int32_t tokenId;

    T const rld = T(1.f) / T(ld);
    // seqPos = b + s * B
    // int32_t const seqPos = blockIdx.y + blockIdx.x * gridDim.y;

    // int32_t const seqPos = s * B + s;
    int32_t const seqPos = sumS + s;
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
        for (int32_t it = threadIdx.x; it < ld; it += TPB)
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
int32_t embSkipLayerNormHFace(cudaStream_t stream, int32_t ld, int32_t B, int32_t S, int32_t const* inputIds,
    int32_t const* tokenIds, int32_t const* cuSeqlens, float const* beta, float const* gamma, T const* wordEmb,
    T const* posEmb, T const* tokEmb, int32_t const wordSize, int32_t const tokSize, T* output)
{

    constexpr int32_t tpb = 256;
    dim3 const grid(S, B, 1);
    dim3 const block(tpb, 1, 1);

    embLayerNormKernelHFace<T, tpb><<<grid, block, 0, stream>>>(
        ld, inputIds, tokenIds, cuSeqlens, beta, gamma, wordEmb, posEmb, tokEmb, wordSize, tokSize, output);
    return cudaPeekAtLastError();
}

template int32_t embSkipLayerNormHFace<float>(cudaStream_t, int32_t, int32_t, int32_t, int32_t const*, int32_t const*,
    int32_t const*, float const*, float const*, float const*, float const*, float const*, int32_t const, int32_t const,
    float*);

template int32_t embSkipLayerNormHFace<half>(cudaStream_t, int32_t, int32_t, int32_t, int32_t const*, int32_t const*,
    int32_t const*, float const*, float const*, half const*, half const*, half const*, int32_t const, int32_t const,
    half*);

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
