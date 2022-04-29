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


#include <stdio.h>
#include <cuda_fp16.h>
#include "disentangledAttentionPlugin.h"

#define IND(i,j,k,dim) ((i)*dim.y*dim.z + (j)*dim.z + (k)) // caveat: must use brackets around var name! otherwise IND(i,j+3,k,dim) = (i*dim.y*dim.z + j+3*dim.z + k)...

#define TILE_DIM (32)
#define BLOCK_ROWS (8)

namespace nvinfer1
{
namespace plugin
{

using namespace nvinfer1;

/**
 * Fused kernel for Disentangled Attention design (first proposed in Microsoft DeBERTa), Version 1.
 *
 * @tparam TDataType type of the input data
 * @param data1 content-to-position ("c2p") attention QcKr^T
 * @param index1 c2p gather index
 * @param data2 position-to-content ("p2c") attention KcQr^T
 * @param index2 p2c gather index
 * @param result attention result
 * @param dimData1, dimIndex1, dimData2, dimIndex2, dimResult dimension of the tensors
 */
template <typename TDataType = __half>
__global__ void GatherAddGatherTranspose_fused(TDataType const* data1, int32_t const* index1, TDataType const* data2,
    int32_t const* index2, TDataType* result, dim3 dimData1, dim3 dimIndex1, dim3 dimData2, dim3 dimIndex2,
    dim3 dimResult)
{
    // map block to the output (result)
    int32_t i;
    int32_t j;
    int32_t k;
    int32_t c;
    int32_t ty;
    TDataType res1;
    TDataType res2;

    __shared__ TDataType T[TILE_DIM][TILE_DIM + 1]; // avoid bank conflict

    // (i,j,k) location of data2 (transposed)
    i = blockIdx.z;
    j = blockIdx.x * TILE_DIM + threadIdx.y;
    k = blockIdx.y * TILE_DIM + threadIdx.x;

// gather data2
#pragma unroll
    for (c = 0, ty = 0; c < TILE_DIM / BLOCK_ROWS; c++, ty += BLOCK_ROWS)
    {

        if (j + ty - k <= -256)
        {
            res2 = data2[IND(i, j + ty, 511, dimData2)];
        }
        else if (j + ty - k >= 256)
        {
            res2 = data2[IND(i, j + ty, 0, dimData2)];
        }
        else
        {
            res2 = data2[IND(i, j + ty, index2[IND(i, j + ty, k, dimIndex2)], dimData2)];
        }
        T[ty + threadIdx.y][threadIdx.x] = res2;
    }

    __syncthreads();

    // (i,j,k) location of data1 (non-transposed) and output. i unchanged
    j = blockIdx.y * TILE_DIM + threadIdx.y;
    k = blockIdx.x * TILE_DIM + threadIdx.x;

// gather data1 + add + write
#pragma unroll
    for (c = 0, ty = 0; c < TILE_DIM / BLOCK_ROWS; c++, ty += BLOCK_ROWS)
    {

        if (j + ty - k <= -256)
        {
            res1 = data1[IND(i, j + ty, 0, dimData1)];
        }
        else if (j + ty - k >= 256)
        {
            res1 = data1[IND(i, j + ty, 511, dimData1)];
        }
        else
        {
            res1 = data1[IND(i, j + ty, index1[IND(i, j + ty, k, dimIndex1)], dimData1)];
        }
        result[IND(i, j + ty, k, dimResult)]
            = __hadd(T[threadIdx.x][ty + threadIdx.y], res1); // fused add (for non-transposed matrix 1, just fetch
                                                              // element at the transposed location & add to the result)
    }
}

/**
 * Fused kernel for Disentangled Attention design (first proposed in Microsoft DeBERTa), Version 2.
 *
 * @tparam TDataType type of the input data
 * @tparam tTileSize dimension of the shared memory tile (square) and also the BlockDimX
 * @tparam tBlockDimY 2D thread block is (tTileSize, tBlockDimY)
 * @param data0 content-to-content ("c2c") attention QcKc^T
 * @param data1 content-to-position ("c2p") attention QcKr^T
 * @param data2 position-to-content ("p2c") attention KcQr^T
 * @param result attention result
 * @param dimData0, dimData1, dimData2, dimResult dimension of the tensors
 * @param factor scaling factor applied on attention for stabilizing model training, 1/sqrt(3d), d is hidden size per
 * head = H/N. H is hidden size, N is number of heads
 * @param span relative distance hyper-parameter, k, in Disentangled attention
 * @note C++ 17 and above due to constexpr if
 */
template <typename TDataType = __half, int32_t tTileSize = 32, int32_t tBlockDimY = 8>
__global__ void GatherAddGatherTransposeAddMul_fused(TDataType const* data0, TDataType const* data1,
    TDataType const* data2, TDataType* result, dim3 dimData0, dim3 dimData1, dim3 dimData2, dim3 dimResult,
    TDataType factor, int32_t span)
{
    // TILE_DIM should be a multiple of BLOCK_ROWS
    assert(tBlockDimY * (tTileSize / tBlockDimY) == tTileSize);

    // map block to the output (result)
    int32_t i;
    int32_t j;
    int32_t k;
    int32_t c;
    int32_t ty;
    TDataType res0;
    TDataType res1;
    TDataType res2;
    TDataType res;

    __shared__ TDataType T[tTileSize][tTileSize + 1]; // +1 to avoid bank conflict

    // (i,j,k) location of data2 (transposed)
    i = blockIdx.z;
    j = blockIdx.x * tTileSize + threadIdx.y;
    k = blockIdx.y * tTileSize + threadIdx.x;

// gather data2
#pragma unroll
    for (c = 0, ty = 0; c < tTileSize / tBlockDimY; c++, ty += tBlockDimY)
    {

        if (k - (j + ty) >= span)
        {
            res2 = data2[IND(i, j + ty, 2 * span - 1, dimData2)];
        }
        else if (k - (j + ty) <= -span)
        {
            res2 = data2[IND(i, j + ty, 0, dimData2)];
        }
        else
        {
            res2 = data2[IND(i, j + ty, k - (j + ty) + span, dimData2)]; // compute index on the fly
        }
        T[ty + threadIdx.y][threadIdx.x] = res2;
    }

    __syncthreads();

    // (i,j,k) location of data1 (non-transposed) and output. i unchanged
    j = blockIdx.y * tTileSize + threadIdx.y;
    k = blockIdx.x * tTileSize + threadIdx.x;

// read data0 + gather data1 + add all + write
#pragma unroll
    for (c = 0, ty = 0; c < tTileSize / tBlockDimY; c++, ty += tBlockDimY)
    {

        // for non-transposed matrix 1, just fetch element at the transposed location & add to the result)
        if (j + ty - k <= -span)
        {
            res1 = data1[IND(i, j + ty, 0, dimData1)];
        }
        else if (j + ty - k >= span)
        {
            res1 = data1[IND(i, j + ty, 2 * span - 1, dimData1)];
        }
        else
        {
            res1 = data1[IND(i, j + ty, j + ty - k + span, dimData1)]; // compute index on the fly
        }
        // for non-tranposed matrix 0, same as matrix 1
        res0 = data0[IND(i, j + ty, k, dimData0)];

        // (res0 + res1 + res2) / sqrt(3d), d is the hidden states size per head
        if constexpr (std::is_same<TDataType, double>::value || std::is_same<TDataType, float>::value)
        {
            // double, float32
            res = (res0 + res1 + T[threadIdx.x][ty + threadIdx.y]) * factor;
        }
        else if constexpr (std::is_same<TDataType, __half>::value || std::is_same<TDataType, half>::value)
        {
            // fp16
            res = __hmul(__hadd(res0, __hadd(res1, T[threadIdx.x][ty + threadIdx.y])),
                factor); // note: __hmul only supported >= sm_53
        }
        else if constexpr (std::is_same<TDataType, int8_t>::value || std::is_same<TDataType, uint8_t>::value)
        {
            // int8_t
            res = (res0 + res1 + T[threadIdx.x][ty + threadIdx.y]) * factor;
        }
        // write
        result[IND(i, j + ty, k, dimResult)] = res;
    }
}

template <typename TDataType>
void disentangled_kernel_wrapper_v1(TDataType const* data1, int32_t const* index1, TDataType const* data2,
    int32_t const* index2, TDataType* result, dim3 dimData1, dim3 dimIndex1, dim3 dimData2, dim3 dimIndex2,
    dim3 dimResult, dim3 block, dim3 grid, cudaStream_t stream)
{
    GatherAddGatherTranspose_fused<__half><<<grid, block, 0, stream>>>(
        data1, index1, data2, index2, result, dimData1, dimIndex1, dimData2, dimIndex2, dimResult);
}

template <typename TDataType, int32_t tTileSize, int32_t tBlockDimY>
void disentangled_kernel_wrapper_v2(TDataType const* data0, TDataType const* data1, TDataType const* data2,
    TDataType* result, dim3 dimData0, dim3 dimData1, dim3 dimData2, dim3 dimResult, TDataType factor, int32_t span,
    dim3 block, dim3 grid, cudaStream_t stream)
{
    GatherAddGatherTransposeAddMul_fused<TDataType, tTileSize, tBlockDimY><<<grid, block, 0, stream>>>(
        data0, data1, data2, result, dimData0, dimData1, dimData2, dimResult, factor, span);
}

template void disentangled_kernel_wrapper_v1<__half>(__half const*, int32_t const*, __half const*, int32_t const*,
    __half*, dim3, dim3, dim3, dim3, dim3, dim3, dim3, cudaStream_t);

template void disentangled_kernel_wrapper_v2<float, 32, 8>(
    float const*, float const*, float const*, float*, dim3, dim3, dim3, dim3, float, int32_t, dim3, dim3, cudaStream_t);

template void disentangled_kernel_wrapper_v2<__half, 32, 8>(__half const*, __half const*, __half const*, __half*, dim3,
    dim3, dim3, dim3, __half, int32_t, dim3, dim3, cudaStream_t);

template void disentangled_kernel_wrapper_v2<int8_t, 32, 8>(int8_t const*, int8_t const*, int8_t const*, int8_t*, dim3,
    dim3, dim3, dim3, int8_t, int32_t, dim3, dim3, cudaStream_t);

#undef TILE_DIM
#undef BLOCK_ROWS
#undef IND


} /* plugin */
} /* nvinfer1 */