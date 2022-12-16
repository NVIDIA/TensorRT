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

#include "disentangledAttentionPlugin.h"
#include <cuda_fp16.h>
#include <stdio.h>

#define IND(i, j, k, dim)                                                                                              \
    ((i) *dim.y * dim.z + (j) *dim.z + (k)) // caveat: must use brackets around var name! otherwise IND(i,j+3,k,dim) =
                                            // (i*dim.y*dim.z + j+3*dim.z + k)...

namespace nvinfer1
{
namespace plugin
{

using namespace nvinfer1;

// template specialization for double/float
template <typename TDataType,
    typename std::enable_if<std::is_same<std::decay_t<TDataType>, double>::value
            || std::is_same<std::decay_t<TDataType>, float>::value,
        TDataType>::type* dummy
    = nullptr>
__forceinline__ __device__ void compute_attention(
    TDataType& res, const TDataType& res0, const TDataType& res1, const TDataType& res2, const TDataType& factor)
{
    res = (res0 + res1 + res2) * factor;
}

// template specialization for half
template <typename TDataType,
    typename std::enable_if<std::is_same<std::decay_t<TDataType>, __half>::value
            || std::is_same<std::decay_t<TDataType>, half>::value,
        TDataType>::type* dummy
    = nullptr>
__forceinline__ __device__ void compute_attention(
    TDataType& res, const TDataType& res0, const TDataType& res1, const TDataType& res2, const TDataType& factor)
{
#if __CUDA_ARCH__ >= 530
    // __hmul only supported >= sm_53
    res = __hmul(__hadd(res0, __hadd(res1, res2)), factor);
#else
    // for < sm_53, workaround/fallback is convert to float and downconvert
    res = __float2half((__half2float(res0) + __half2float(res1) + __half2float(res2)) * __half2float(factor));
#endif
}

// template specialization for int8
template <typename TDataType,
    typename std::enable_if<std::is_same<std::decay_t<TDataType>, int8_t>::value
            || std::is_same<std::decay_t<TDataType>, uint8_t>::value,
        TDataType>::type* dummy
    = nullptr>
__forceinline__ __device__ void compute_attention(
    TDataType& res, const TDataType& res0, const TDataType& res1, const TDataType& res2, const TDataType& factor)
{
    res = (res0 + res1 + res2) * factor;
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
    // Tile size should be a multiple of number of block rows
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

#if kDISENTANGLED_VERSION == 2
    int32_t bucket;
    int32_t mid = span / 2;
    int32_t index;

    // tmp values are precomputed for re-use; must be at least float to ensure accuracy
    float tmp1 = logf(mid);

    // Multiply by (1 - epsilon) to ensure that taking the ceil of approximately an integer
    // results in that integer when computing the bucket later on.
    // This corrects for the mathematical imprecision from using float.
    constexpr float kEPSILON = 1e-7;
    float tmp = (mid - 1) / (logf(dimResult.y - 1) - tmp1) * (1 - kEPSILON);
#endif

    __shared__ TDataType T[tTileSize][tTileSize + 1]; // +1 to avoid bank conflict

    // (i,j,k) location of data2 (transposed)
    i = blockIdx.z;
    j = blockIdx.x * tTileSize + threadIdx.y;
    k = blockIdx.y * tTileSize + threadIdx.x;

// gather data2
#pragma unroll
    for (c = 0, ty = 0; c < tTileSize / tBlockDimY; c++, ty += tBlockDimY)
    {
#if kDISENTANGLED_VERSION == 1
        // relative position -- version 1
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
#elif kDISENTANGLED_VERSION == 2
        // relative position w/ log bucket -- version 2
        if (k - (j + ty) >= -mid && k - (j + ty) <= mid)
        {
            // preserved region, (i - j) + span
            bucket = k - (j + ty);
        }
        else
        {
            // log bucket region, bucket(i,j) + span
            bucket = ceilf((logf(fabsf(k - (j + ty))) - tmp1) * tmp) + mid;
            bucket = k - (j + ty) < 0 ? -bucket : bucket;
        }
        // clamp [0,2k]. Although this is guaranteed by equation, but numerically the floating precision can still break
        // boundary
        index = bucket + span;
        index = min(max(0, index), 2 * span - 1);
        res2 = data2[IND(i, j + ty, index, dimData2)];
        T[ty + threadIdx.y][threadIdx.x] = res2;
#endif
    }

    __syncthreads();

    // (i,j,k) location of data1 (non-transposed) and output. i unchanged
    j = blockIdx.y * tTileSize + threadIdx.y;
    k = blockIdx.x * tTileSize + threadIdx.x;

// read data0 + gather data1 + add all + write
#pragma unroll
    for (c = 0, ty = 0; c < tTileSize / tBlockDimY; c++, ty += tBlockDimY)
    {
#if kDISENTANGLED_VERSION == 1
        // relative position -- version 1
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
#elif kDISENTANGLED_VERSION == 2
        // relative position w/ log bucket -- version 2
        if (j + ty - k >= -mid && j + ty - k <= mid)
        {
            // preserved region, (i - j) + span
            bucket = j + ty - k;
        }
        else
        {
            // log bucket region, bucket(i,j) + span
            bucket = ceilf((logf(fabsf((j + ty) - k)) - tmp1) * tmp) + mid;
            bucket = (j + ty) - k < 0 ? -bucket : bucket;
        }
        // clamp [0,2k]. Although this is guaranteed by equation, but numerically the floating precision can still break
        // boundary
        index = bucket + span;
        index = min(max(0, index), 2 * span - 1);
        res1 = data1[IND(i, j + ty, index, dimData1)];
#endif

        // for non-tranposed matrix 0, same as matrix 1
        res0 = data0[IND(i, j + ty, k, dimData0)];

        // (res0 + res1 + res2) / sqrt(3d), d is the hidden states size per head
#if __cplusplus >= 201703L
        // C++ 17 has more convenient `if constexpr` for conditional implementation at compile time; before C++ 17,
        // switch to template specialization
        if constexpr (std::is_same<TDataType, double>::value || std::is_same<TDataType, float>::value)
        {
            // double, float32
            res = (res0 + res1 + T[threadIdx.x][ty + threadIdx.y]) * factor;
        }
        else if constexpr (std::is_same<TDataType, __half>::value || std::is_same<TDataType, half>::value)
        {
            // fp16
#if __CUDA_ARCH__ >= 530
            // __hmul only supported >= sm_53
            res = __hmul(__hadd(res0, __hadd(res1, T[threadIdx.x][ty + threadIdx.y])), factor);
#else
            // for < sm_53, workaround/fallback is convert to float and downconvert
            res = __float2half(
                (__half2float(res0) + __half2float(res1) + __half2float(T[threadIdx.x][ty + threadIdx.y]))
                * __half2float(factor));
#endif
        }
        else if constexpr (std::is_same<TDataType, int8_t>::value || std::is_same<TDataType, uint8_t>::value)
        {
            // int8_t
            res = (res0 + res1 + T[threadIdx.x][ty + threadIdx.y]) * factor;
        }
#else
        // before C++ 17, use template specialization
        compute_attention<TDataType>(res, res0, res1, T[threadIdx.x][ty + threadIdx.y], factor);
#endif
        // write
        result[IND(i, j + ty, k, dimResult)] = res;
    }
}

template <typename TDataType, int32_t tTileSize, int32_t tBlockDimY>
void disentangled_kernel_wrapper(TDataType const* data0, TDataType const* data1, TDataType const* data2,
    TDataType* result, dim3 dimData0, dim3 dimData1, dim3 dimData2, dim3 dimResult, TDataType factor, int32_t span,
    dim3 block, dim3 grid, cudaStream_t stream)
{
    GatherAddGatherTransposeAddMul_fused<TDataType, tTileSize, tBlockDimY><<<grid, block, 0, stream>>>(
        data0, data1, data2, result, dimData0, dimData1, dimData2, dimResult, factor, span);
}

template void disentangled_kernel_wrapper<float, kDISENTANGLED_TILESIZE_V1, kDISENTANGLED_BLOCKDIMY_V1>(
    float const*, float const*, float const*, float*, dim3, dim3, dim3, dim3, float, int32_t, dim3, dim3, cudaStream_t);

template void disentangled_kernel_wrapper<__half, kDISENTANGLED_TILESIZE_V1, kDISENTANGLED_BLOCKDIMY_V1>(__half const*,
    __half const*, __half const*, __half*, dim3, dim3, dim3, dim3, __half, int32_t, dim3, dim3, cudaStream_t);

template void disentangled_kernel_wrapper<int8_t, kDISENTANGLED_TILESIZE_V1, kDISENTANGLED_BLOCKDIMY_V1>(int8_t const*,
    int8_t const*, int8_t const*, int8_t*, dim3, dim3, dim3, dim3, int8_t, int32_t, dim3, dim3, cudaStream_t);

template void disentangled_kernel_wrapper<float, kDISENTANGLED_TILESIZE_V2, kDISENTANGLED_BLOCKDIMY_V2>(
    float const*, float const*, float const*, float*, dim3, dim3, dim3, dim3, float, int32_t, dim3, dim3, cudaStream_t);

template void disentangled_kernel_wrapper<__half, kDISENTANGLED_TILESIZE_V2, kDISENTANGLED_BLOCKDIMY_V2>(__half const*,
    __half const*, __half const*, __half*, dim3, dim3, dim3, dim3, __half, int32_t, dim3, dim3, cudaStream_t);

template void disentangled_kernel_wrapper<int8_t, kDISENTANGLED_TILESIZE_V2, kDISENTANGLED_BLOCKDIMY_V2>(int8_t const*,
    int8_t const*, int8_t const*, int8_t*, dim3, dim3, dim3, dim3, int8_t, int32_t, dim3, dim3, cudaStream_t);

#undef IND

} /* plugin */
} // namespace nvinfer1
