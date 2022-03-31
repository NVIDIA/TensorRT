/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************
 * Modified from Deformable DETR
 * Copyright (c) 2020 SenseTime. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 [see LICENSE for details]
 * https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE
 **************************************************************************
 * Modified from DCN (https://github.com/msracver/Deformable-ConvNets)
 * Copyright (c) 2018 Microsoft
 **************************************************************************
*/

#ifndef TRT_MULTISCALE_DEFORMABLE_IM2COL_CUDA_H
#define TRT_MULTISCALE_DEFORMABLE_IM2COL_CUDA_H

#include <algorithm>
#include <cstdio>
#include <cstring>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 768;
inline int GET_BLOCKS(const int N, const int numThreads)
{
    return (N + numThreads - 1) / numThreads;
}

template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(const scalar_t*& bottomData, const int& height, const int& width,
    const int& nHeads, const int& channels, const scalar_t& h, const scalar_t& w, const int& m, const int& c)
{
    const int hLow = floor(h);
    const int wLow = floor(w);
    const int hHigh = hLow + 1;
    const int wHigh = wLow + 1;

    const scalar_t lh = h - hLow;
    const scalar_t lw = w - wLow;
    const scalar_t hh = 1 - lh, hw = 1 - lw;

    const int wStride = nHeads * channels;
    const int hStride = width * wStride;
    const int hLowPtrOffset = hLow * hStride;
    const int hHighPtrOffset = hLowPtrOffset + hStride;
    const int wLowPtrOffset = wLow * wStride;
    const int wHighPtrOffset = wLowPtrOffset + wStride;
    const int basePtr = m * channels + c;

    scalar_t v1 = 0;
    if (hLow >= 0 && wLow >= 0)
    {
        const int ptr1 = hLowPtrOffset + wLowPtrOffset + basePtr;
        v1 = bottomData[ptr1];
    }
    scalar_t v2 = 0;
    if (hLow >= 0 && wHigh <= width - 1)
    {
        const int ptr2 = hLowPtrOffset + wHighPtrOffset + basePtr;
        v2 = bottomData[ptr2];
    }
    scalar_t v3 = 0;
    if (hHigh <= height - 1 && wLow >= 0)
    {
        const int ptr3 = hHighPtrOffset + wLowPtrOffset + basePtr;
        v3 = bottomData[ptr3];
    }
    scalar_t v4 = 0;
    if (hHigh <= height - 1 && wHigh <= width - 1)
    {
        const int ptr4 = hHighPtrOffset + wHighPtrOffset + basePtr;
        v4 = bottomData[ptr4];
    }

    const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

#if __CUDA_ARCH__ >= 530
template <>
__device__ __half ms_deform_attn_im2col_bilinear<__half>(const __half*& bottomData, const int& height, const int& width,
    const int& nHeads, const int& channels, const __half& h, const __half& w, const int& m, const int& c)
{
    const int hLow = __half2int_rd(h);
    const int wLow = __half2int_rd(w);
    const int hHigh = hLow + 1;
    const int wHigh = wLow + 1;

    const __half zero = __int2half_rz(0);
    const __half one = __int2half_rz(1);
    const __half lh = __hsub(h, __int2half_rd(hLow));
    const __half lw = __hsub(w, __int2half_rd(wLow));
    const __half hh = __hsub(one, lh), hw = __hsub(one, lw);

    const int wStride = nHeads * channels;
    const int hStride = width * wStride;
    const int hLowPtrOffset = hLow * hStride;
    const int hHighPtrOffset = hLowPtrOffset + hStride;
    const int wLowPtrOffset = wLow * wStride;
    const int wHighPtrOffset = wLowPtrOffset + wStride;
    const int basePtr = m * channels + c;

    __half v1 = zero;
    if (hLow >= 0 && wLow >= 0)
    {
        const int ptr1 = hLowPtrOffset + wLowPtrOffset + basePtr;
        v1 = bottomData[ptr1];
    }
    __half v2 = zero;
    if (hLow >= 0 && wHigh <= width - 1)
    {
        const int ptr2 = hLowPtrOffset + wHighPtrOffset + basePtr;
        v2 = bottomData[ptr2];
    }
    __half v3 = zero;
    if (hHigh <= height - 1 && wLow >= 0)
    {
        const int ptr3 = hHighPtrOffset + wLowPtrOffset + basePtr;
        v3 = bottomData[ptr3];
    }
    __half v4 = zero;
    if (hHigh <= height - 1 && wHigh <= width - 1)
    {
        const int ptr4 = hHighPtrOffset + wHighPtrOffset + basePtr;
        v4 = bottomData[ptr4];
    }

    __half w1 = __hmul(__hmul(hh, hw), v1);
    __half w2 = __hmul(__hmul(hh, lw), v2);
    __half w3 = __hmul(__hmul(lh, hw), v3);
    __half w4 = __hmul(__hmul(lh, lw), v4);

    w1 = __hadd(w1, w2);
    w3 = __hadd(w3, w4);

    const __half val = __hadd(w1, w3);
    return val;
}
#endif

template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear(const scalar_t*& bottomData, const int& height, const int& width,
    const int& nHeads, const int& channels, const scalar_t& h, const scalar_t& w, const int& m, const int& c,
    const scalar_t& topGrad, const scalar_t& attnWeight, scalar_t*& gradValue, scalar_t* gradSamplingLoc,
    scalar_t* gradAttnWeight)
{
    const int hLow = floor(h);
    const int wLow = floor(w);
    const int hHigh = hLow + 1;
    const int wHigh = wLow + 1;

    const scalar_t lh = h - hLow;
    const scalar_t lw = w - wLow;
    const scalar_t hh = 1 - lh, hw = 1 - lw;

    const int wStride = nHeads * channels;
    const int hStride = width * wStride;
    const int hLowPtrOffset = hLow * hStride;
    const int hHighPtrOffset = hLowPtrOffset + hStride;
    const int wLowPtrOffset = wLow * wStride;
    const int wHighPtrOffset = wLowPtrOffset + wStride;
    const int basePtr = m * channels + c;

    const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    const scalar_t topGradvalue = topGrad * attnWeight;
    scalar_t gradHWeight = 0, gradWWeight = 0;

    scalar_t v1 = 0;
    if (hLow >= 0 && wLow >= 0)
    {
        const int ptr1 = hLowPtrOffset + wLowPtrOffset + basePtr;
        v1 = bottomData[ptr1];
        gradHWeight -= hw * v1;
        gradWWeight -= hh * v1;
        atomicAdd(gradValue + ptr1, w1 * topGradvalue);
    }
    scalar_t v2 = 0;
    if (hLow >= 0 && wHigh <= width - 1)
    {
        const int ptr2 = hLowPtrOffset + wHighPtrOffset + basePtr;
        v2 = bottomData[ptr2];
        gradHWeight -= lw * v2;
        gradWWeight += hh * v2;
        atomicAdd(gradValue + ptr2, w2 * topGradvalue);
    }
    scalar_t v3 = 0;
    if (hHigh <= height - 1 && wLow >= 0)
    {
        const int ptr3 = hHighPtrOffset + wLowPtrOffset + basePtr;
        v3 = bottomData[ptr3];
        gradHWeight += hw * v3;
        gradWWeight -= lh * v3;
        atomicAdd(gradValue + ptr3, w3 * topGradvalue);
    }
    scalar_t v4 = 0;
    if (hHigh <= height - 1 && wHigh <= width - 1)
    {
        const int ptr4 = hHighPtrOffset + wHighPtrOffset + basePtr;
        v4 = bottomData[ptr4];
        gradHWeight += lw * v4;
        gradWWeight += lh * v4;
        atomicAdd(gradValue + ptr4, w4 * topGradvalue);
    }

    const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    *gradAttnWeight = topGrad * val;
    *gradSamplingLoc = width * gradWWeight * topGradvalue;
    *(gradSamplingLoc + 1) = height * gradHWeight * topGradvalue;
}

template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear_gm(const scalar_t*& bottomData, const int& height, const int& width,
    const int& nHeads, const int& channels, const scalar_t& h, const scalar_t& w, const int& m, const int& c,
    const scalar_t& topGrad, const scalar_t& attnWeight, scalar_t*& gradValue, scalar_t* gradSamplingLoc,
    scalar_t* gradAttnWeight)
{
    const int hLow = floor(h);
    const int wLow = floor(w);
    const int hHigh = hLow + 1;
    const int wHigh = wLow + 1;

    const scalar_t lh = h - hLow;
    const scalar_t lw = w - wLow;
    const scalar_t hh = 1 - lh, hw = 1 - lw;

    const int wStride = nHeads * channels;
    const int hStride = width * wStride;
    const int hLowPtrOffset = hLow * hStride;
    const int hHighPtrOffset = hLowPtrOffset + hStride;
    const int wLowPtrOffset = wLow * wStride;
    const int wHighPtrOffset = wLowPtrOffset + wStride;
    const int basePtr = m * channels + c;

    const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    const scalar_t topGradvalue = topGrad * attnWeight;
    scalar_t gradHWeight = 0, gradWWeight = 0;

    scalar_t v1 = 0;
    if (hLow >= 0 && wLow >= 0)
    {
        const int ptr1 = hLowPtrOffset + wLowPtrOffset + basePtr;
        v1 = bottomData[ptr1];
        gradHWeight -= hw * v1;
        gradWWeight -= hh * v1;
        atomicAdd(gradValue + ptr1, w1 * topGradvalue);
    }
    scalar_t v2 = 0;
    if (hLow >= 0 && wHigh <= width - 1)
    {
        const int ptr2 = hLowPtrOffset + wHighPtrOffset + basePtr;
        v2 = bottomData[ptr2];
        gradHWeight -= lw * v2;
        gradWWeight += hh * v2;
        atomicAdd(gradValue + ptr2, w2 * topGradvalue);
    }
    scalar_t v3 = 0;
    if (hHigh <= height - 1 && wLow >= 0)
    {
        const int ptr3 = hHighPtrOffset + wLowPtrOffset + basePtr;
        v3 = bottomData[ptr3];
        gradHWeight += hw * v3;
        gradWWeight -= lh * v3;
        atomicAdd(gradValue + ptr3, w3 * topGradvalue);
    }
    scalar_t v4 = 0;
    if (hHigh <= height - 1 && wHigh <= width - 1)
    {
        const int ptr4 = hHighPtrOffset + wHighPtrOffset + basePtr;
        v4 = bottomData[ptr4];
        gradHWeight += lw * v4;
        gradWWeight += lh * v4;
        atomicAdd(gradValue + ptr4, w4 * topGradvalue);
    }

    const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    atomicAdd(gradAttnWeight, topGrad * val);
    atomicAdd(gradSamplingLoc, width * gradWWeight * topGradvalue);
    atomicAdd(gradSamplingLoc + 1, height * gradHWeight * topGradvalue);
}
#if 0
template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(const int n,
                                                const scalar_t *dataValue,
                                                const int32_t *dataSpatialShapes,
                                                const int32_t *dataLevelStartIndex,
                                                const scalar_t *dataSamplingLoc,
                                                const scalar_t *dataAttnWeight,
                                                const int batchSize,
                                                const int spatialSize,
                                                const int numHeads,
                                                const int channels,
                                                const int numLevels,
                                                const int numQuery,
                                                const int numPoint,
                                                scalar_t *dataCol)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    int cCol = _temp % channels;
    _temp /= channels;
    int samplingIndex = _temp;
    int mCol = _temp % numHeads;
    _temp /= numHeads;
    _temp /= numQuery;
    int bCol = _temp;

    scalar_t *dataColPtr = dataCol + index;
    int dataWeightPtr = samplingIndex * numLevels * numPoint;
    int dataLocWPtr = dataWeightPtr << 1;
    int qidStride = numHeads * channels;
    int dataValuePtrInitOffset = bCol * spatialSize * qidStride;
    scalar_t col = 0;

    for (int lCol = 0; lCol < numLevels; ++lCol)
    {
      const int &spatialH = dataSpatialShapes[lCol << 1];
      const int &spatialW = dataSpatialShapes[lCol << 1 + 1];
      const scalar_t *dataValuePtr = dataValue + (dataValuePtrInitOffset + dataLevelStartIndex[lCol] * qidStride);
      for (int pCol = 0; pCol < numPoint; ++pCol)
      {
        scalar_t locW = dataSamplingLoc[dataLocWPtr];
        scalar_t locH = dataSamplingLoc[dataLocWPtr + 1];
        scalar_t weight = dataAttnWeight[dataWeightPtr];

        scalar_t hIm = locH * spatialH - 0.5;
        scalar_t wIm = locW * spatialW - 0.5;

        if (hIm > -1 && hIm < spatialH && wIm > -1 && wIm < spatialW)
        {
          col += ms_deform_attn_im2col_bilinear(dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol, cCol) * weight;
        }

        dataWeightPtr += 1;
        dataLocWPtr += 2;
      }
    }
    *dataColPtr = col;
  }
}

#if __CUDA_ARCH__ >= 530
template <>
__global__ void ms_deformable_im2col_gpu_kernel<__half>(const int n,
                                                const __half *dataValue,
                                                const int32_t *dataSpatialShapes,
                                                const int32_t *dataLevelStartIndex,
                                                const __half *dataSamplingLoc,
                                                const __half *dataAttnWeight,
                                                const int batchSize,
                                                const int spatialSize,
                                                const int numHeads,
                                                const int channels,
                                                const int numLevels,
                                                const int numQuery,
                                                const int numPoint,
                                                __half *dataCol)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    int cCol = _temp % channels;
    _temp /= channels;
    int samplingIndex = _temp;
    int mCol = _temp % numHeads;
    _temp /= numHeads;
    _temp /= numQuery;
    int bCol = _temp;

    __half *dataColPtr = dataCol + index;
    int dataWeightPtr = samplingIndex * numLevels * numPoint;
    int dataLocWPtr = dataWeightPtr << 1;
    int qidStride = numHeads * channels;
    int dataValuePtrInitOffset = bCol * spatialSize * qidStride;
    __half zeroPointFive = __float2half(0.5f);
    __half minusOne = __float2half(-1.0f);
    __half zero = __int2half_rz(0);
    __half tpVal = zero;
    __half col = zero;

    for (int lCol = 0; lCol < numLevels; ++lCol)
    {
      const int &spatialH = dataSpatialShapes[lCol << 1];
      const int &spatialW = dataSpatialShapes[lCol << 1 + 1];
      __half spatialHHalf = __int2half_rd(spatialH);
      __half spatialWHalf = __int2half_rd(spatialW);
      const __half *dataValuePtr = dataValue + (dataValuePtrInitOffset + dataLevelStartIndex[lCol] * qidStride);
      for (int pCol = 0; pCol < numPoint; ++pCol)
      {
        __half locW = dataSamplingLoc[dataLocWPtr];
        __half locH = dataSamplingLoc[dataLocWPtr + 1];
        __half weight = dataAttnWeight[dataWeightPtr];

        __half hIm = __hsub(__hmul(locH, spatialHHalf), zeroPointFive);
        __half wIm = __hsub(__hmul(locW, spatialWHalf), zeroPointFive);

        if (__hgt(hIm, minusOne) && __hlt(hIm, spatialHHalf) && 
            __hgt(wIm, minusOne) && __hlt(wIm, spatialWHalf)) {
          tpVal = ms_deform_attn_im2col_bilinear(dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol, cCol);
          col = __hadd(col, __hmul(tpVal, weight));
        }

        dataWeightPtr += 1;
        dataLocWPtr += 2;
      }
    }
    *dataColPtr = col;
  }
}
#endif // CUDA_ARCH>=530 check
#endif
#if 1
template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(const int n, const scalar_t* dataValue,
    const int32_t* dataSpatialShapes, const int32_t* dataLevelStartIndex, const scalar_t* dataSamplingLoc,
    const scalar_t* dataAttnWeight, const int batchSize, const int spatialSize, const int numHeads, const int channels,
    const int numLevels, const int numQuery, const int numPoint, scalar_t* dataCol)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        int _temp = index;
        const int cCol = _temp % channels;
        _temp /= channels;
        const int samplingIndex = _temp;
        const int mCol = _temp % numHeads;
        _temp /= numHeads;
        _temp /= numQuery;
        const int bCol = _temp;

        scalar_t* dataColPtr = dataCol + index;
        int dataWeightPtr = samplingIndex * numLevels * numPoint;
        int dataLocWPtr = dataWeightPtr << 1;
        const int qidStride = numHeads * channels;
        const int dataValuePtrInitOffset = bCol * spatialSize * qidStride;
        scalar_t col = 0;

        for (int lCol = 0; lCol < numLevels; ++lCol)
        {
            const int levelStartId = dataLevelStartIndex[lCol];
            const int spatialHPtr = lCol << 1;
            const int spatialH = dataSpatialShapes[spatialHPtr];
            const int spatialW = dataSpatialShapes[spatialHPtr + 1];
            const scalar_t* dataValuePtr = dataValue + (dataValuePtrInitOffset + levelStartId * qidStride);
            for (int pCol = 0; pCol < numPoint; ++pCol)
            {
                const scalar_t locW = dataSamplingLoc[dataLocWPtr];
                const scalar_t locH = dataSamplingLoc[dataLocWPtr + 1];
                const scalar_t weight = dataAttnWeight[dataWeightPtr];

                const scalar_t hIm = locH * spatialH - 0.5;
                const scalar_t wIm = locW * spatialW - 0.5;

                if (hIm > -1 && wIm > -1 && hIm < spatialH && wIm < spatialW)
                {
                    col += ms_deform_attn_im2col_bilinear(
                               dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol, cCol)
                        * weight;
                }

                dataWeightPtr += 1;
                dataLocWPtr += 2;
            }
        }
        *dataColPtr = col;
    }
}

#if __CUDA_ARCH__ >= 530
template <>
__global__ void ms_deformable_im2col_gpu_kernel<__half>(const int n, const __half* dataValue,
    const int32_t* dataSpatialShapes, const int32_t* dataLevelStartIndex, const __half* dataSamplingLoc,
    const __half* dataAttnWeight, const int batchSize, const int spatialSize, const int numHeads, const int channels,
    const int numLevels, const int numQuery, const int numPoint, __half* dataCol)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        int _temp = index;
        const int cCol = _temp % channels;
        _temp /= channels;
        const int samplingIndex = _temp;
        const int mCol = _temp % numHeads;
        _temp /= numHeads;
        _temp /= numQuery;
        const int bCol = _temp;

        __half* dataColPtr = dataCol + index;
        int dataWeightPtr = samplingIndex * numLevels * numPoint;
        int dataLocWPtr = dataWeightPtr << 1;
        const int qidStride = numHeads * channels;
        const int dataValuePtrInitOffset = bCol * spatialSize * qidStride;
        const __half zeroPointFive = __float2half(0.5f);
        const __half minusOne = __float2half(-1.0f);
        const __half zero = __int2half_rz(0);
        __half tpVal = zero;
        __half col = zero;

        for (int lCol = 0; lCol < numLevels; ++lCol)
        {
            const int levelStartId = dataLevelStartIndex[lCol];
            const int spatialHPtr = lCol << 1;
            const int spatialH = dataSpatialShapes[spatialHPtr];
            const int spatialW = dataSpatialShapes[spatialHPtr + 1];
            const __half spatialHHalf = __int2half_rd(spatialH);
            const __half spatialWHalf = __int2half_rd(spatialW);
            const __half* dataValuePtr = dataValue + (dataValuePtrInitOffset + levelStartId * qidStride);
            for (int pCol = 0; pCol < numPoint; ++pCol)
            {
                const __half locW = dataSamplingLoc[dataLocWPtr];
                const __half locH = dataSamplingLoc[dataLocWPtr + 1];
                const __half weight = dataAttnWeight[dataWeightPtr];

                const __half hIm = __hsub(__hmul(locH, spatialHHalf), zeroPointFive);
                const __half wIm = __hsub(__hmul(locW, spatialWHalf), zeroPointFive);

                if (__hgt(hIm, minusOne) && __hgt(wIm, minusOne) && __hlt(hIm, spatialHHalf)
                    && __hlt(wIm, spatialWHalf))
                {
                    tpVal = ms_deform_attn_im2col_bilinear(
                        dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol, cCol);
                    col = __hadd(col, __hmul(tpVal, weight));
                }

                dataWeightPtr += 1;
                dataLocWPtr += 2;
            }
        }
        *dataColPtr = col;
    }
}
#endif // CUDA_ARCH >=530 check
#endif

template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(const int n, const scalar_t* grad_col,
    const scalar_t* dataValue, const int32_t* dataSpatialShapes, const int32_t* dataLevelStartIndex,
    const scalar_t* dataSamplingLoc, const scalar_t* dataAttnWeight, const int batchSize, const int spatialSize,
    const int numHeads, const int channels, const int numLevels, const int numQuery, const int numPoint,
    scalar_t* gradValue, scalar_t* gradSamplingLoc, scalar_t* gradAttnWeight)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        __shared__ scalar_t cacheGradSamplingLoc[blockSize * 2];
        __shared__ scalar_t cacheGradAttnWeight[blockSize];
        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int cCol = _temp % channels;
        _temp /= channels;
        const int samplingIndex = _temp;
        const int mCol = _temp % numHeads;
        _temp /= numHeads;
        const int qCol = _temp % numQuery;
        _temp /= numQuery;
        const int bCol = _temp;

        const scalar_t topGrad = grad_col[index];

        int dataWeightPtr = samplingIndex * numLevels * numPoint;
        int dataLocWPtr = dataWeightPtr << 1;
        const int gradSamplingPtr = dataWeightPtr;
        gradSamplingLoc += gradSamplingPtr << 1;
        gradAttnWeight += gradSamplingPtr;
        const int gradWeightStride = 1;
        const int gradLocStride = 2;
        const int qidStride = numHeads * channels;
        const int dataValuePtrInitOffset = bCol * spatialSize * qidStride;

        for (int lCol = 0; lCol < numLevels; ++lCol)
        {
            const int levelStartId = dataLevelStartIndex[lCol];
            const int spatialHPtr = lCol << 1;
            const int spatialH = dataSpatialShapes[spatialHPtr];
            const int spatialW = dataSpatialShapes[spatialHPtr + 1];
            const int valuePtrOffset = dataValuePtrInitOffset + levelStartId * qidStride;
            const scalar_t* dataValuePtr = dataValue + valuePtrOffset;
            scalar_t* gradValuePtr = gradValue + valuePtrOffset;

            for (int pCol = 0; pCol < numPoint; ++pCol)
            {
                const scalar_t locW = dataSamplingLoc[dataLocWPtr];
                const scalar_t locH = dataSamplingLoc[dataLocWPtr + 1];
                const scalar_t weight = dataAttnWeight[dataWeightPtr];

                const scalar_t hIm = locH * spatialH - 0.5;
                const scalar_t wIm = locW * spatialW - 0.5;
                *(cacheGradSamplingLoc + (threadIdx.x << 1)) = 0;
                *(cacheGradSamplingLoc + ((threadIdx.x << 1) + 1)) = 0;
                *(cacheGradAttnWeight + threadIdx.x) = 0;
                if (hIm > -1 && wIm > -1 && hIm < spatialH && wIm < spatialW)
                {
                    ms_deform_attn_col2im_bilinear(dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol,
                        cCol, topGrad, weight, gradValuePtr, cacheGradSamplingLoc + (threadIdx.x << 1),
                        cacheGradAttnWeight + threadIdx.x);
                }

                __syncthreads();
                if (tid == 0)
                {
                    scalar_t _gradW = cacheGradSamplingLoc[0], _gradH = cacheGradSamplingLoc[1],
                             _gradA = cacheGradAttnWeight[0];
                    int sid = 2;
                    for (unsigned int tid = 1; tid < blockSize; ++tid)
                    {
                        _gradW += cacheGradSamplingLoc[sid];
                        _gradH += cacheGradSamplingLoc[sid + 1];
                        _gradA += cacheGradAttnWeight[tid];
                        sid += 2;
                    }

                    *gradSamplingLoc = _gradW;
                    *(gradSamplingLoc + 1) = _gradH;
                    *gradAttnWeight = _gradA;
                }
                __syncthreads();

                dataWeightPtr += 1;
                dataLocWPtr += 2;
                gradAttnWeight += gradWeightStride;
                gradSamplingLoc += gradLocStride;
            }
        }
    }
}

template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2(const int n, const scalar_t* grad_col,
    const scalar_t* dataValue, const int32_t* dataSpatialShapes, const int32_t* dataLevelStartIndex,
    const scalar_t* dataSamplingLoc, const scalar_t* dataAttnWeight, const int batchSize, const int spatialSize,
    const int numHeads, const int channels, const int numLevels, const int numQuery, const int numPoint,
    scalar_t* gradValue, scalar_t* gradSamplingLoc, scalar_t* gradAttnWeight)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        __shared__ scalar_t cacheGradSamplingLoc[blockSize * 2];
        __shared__ scalar_t cacheGradAttnWeight[blockSize];
        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int cCol = _temp % channels;
        _temp /= channels;
        const int samplingIndex = _temp;
        const int mCol = _temp % numHeads;
        _temp /= numHeads;
        const int qCol = _temp % numQuery;
        _temp /= numQuery;
        const int bCol = _temp;

        const scalar_t topGrad = grad_col[index];

        int dataWeightPtr = samplingIndex * numLevels * numPoint;
        int dataLocWPtr = dataWeightPtr << 1;
        const int gradSamplingPtr = dataWeightPtr;
        gradSamplingLoc += gradSamplingPtr << 1;
        gradAttnWeight += gradSamplingPtr;
        const int gradWeightStride = 1;
        const int gradLocStride = 2;
        const int qidStride = numHeads * channels;
        const int dataValuePtrInitOffset = bCol * spatialSize * qidStride;

        for (int lCol = 0; lCol < numLevels; ++lCol)
        {
            const int levelStartId = dataLevelStartIndex[lCol];
            const int spatialHPtr = lCol << 1;
            const int spatialH = dataSpatialShapes[spatialHPtr];
            const int spatialW = dataSpatialShapes[spatialHPtr + 1];
            const int valuePtrOffset = dataValuePtrInitOffset + levelStartId * qidStride;
            const scalar_t* dataValuePtr = dataValue + valuePtrOffset;
            scalar_t* gradValuePtr = gradValue + valuePtrOffset;

            for (int pCol = 0; pCol < numPoint; ++pCol)
            {
                const scalar_t locW = dataSamplingLoc[dataLocWPtr];
                const scalar_t locH = dataSamplingLoc[dataLocWPtr + 1];
                const scalar_t weight = dataAttnWeight[dataWeightPtr];

                const scalar_t hIm = locH * spatialH - 0.5;
                const scalar_t wIm = locW * spatialW - 0.5;
                *(cacheGradSamplingLoc + (threadIdx.x << 1)) = 0;
                *(cacheGradSamplingLoc + ((threadIdx.x << 1) + 1)) = 0;
                *(cacheGradAttnWeight + threadIdx.x) = 0;
                if (hIm > -1 && wIm > -1 && hIm < spatialH && wIm < spatialW)
                {
                    ms_deform_attn_col2im_bilinear(dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol,
                        cCol, topGrad, weight, gradValuePtr, cacheGradSamplingLoc + (threadIdx.x << 1),
                        cacheGradAttnWeight + threadIdx.x);
                }

                __syncthreads();

                for (unsigned int s = blockSize / 2; s > 0; s >>= 1)
                {
                    if (tid < s)
                    {
                        const unsigned int xid1 = tid << 1;
                        const unsigned int xid2 = (tid + s) << 1;
                        cacheGradAttnWeight[tid] += cacheGradAttnWeight[tid + s];
                        cacheGradSamplingLoc[xid1] += cacheGradSamplingLoc[xid2];
                        cacheGradSamplingLoc[xid1 + 1] += cacheGradSamplingLoc[xid2 + 1];
                    }
                    __syncthreads();
                }

                if (tid == 0)
                {
                    *gradSamplingLoc = cacheGradSamplingLoc[0];
                    *(gradSamplingLoc + 1) = cacheGradSamplingLoc[1];
                    *gradAttnWeight = cacheGradAttnWeight[0];
                }
                __syncthreads();

                dataWeightPtr += 1;
                dataLocWPtr += 2;
                gradAttnWeight += gradWeightStride;
                gradSamplingLoc += gradLocStride;
            }
        }
    }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v1(const int n, const scalar_t* grad_col,
    const scalar_t* dataValue, const int32_t* dataSpatialShapes, const int32_t* dataLevelStartIndex,
    const scalar_t* dataSamplingLoc, const scalar_t* dataAttnWeight, const int batchSize, const int spatialSize,
    const int numHeads, const int channels, const int numLevels, const int numQuery, const int numPoint,
    scalar_t* gradValue, scalar_t* gradSamplingLoc, scalar_t* gradAttnWeight)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        extern __shared__ int _s[];
        scalar_t* cacheGradSamplingLoc = (scalar_t*) _s;
        scalar_t* cacheGradAttnWeight = cacheGradSamplingLoc + 2 * blockDim.x;
        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int cCol = _temp % channels;
        _temp /= channels;
        const int samplingIndex = _temp;
        const int mCol = _temp % numHeads;
        _temp /= numHeads;
        const int qCol = _temp % numQuery;
        _temp /= numQuery;
        const int bCol = _temp;

        const scalar_t topGrad = grad_col[index];

        int dataWeightPtr = samplingIndex * numLevels * numPoint;
        int dataLocWPtr = dataWeightPtr << 1;
        const int gradSamplingPtr = dataWeightPtr;
        gradSamplingLoc += gradSamplingPtr << 1;
        gradAttnWeight += gradSamplingPtr;
        const int gradWeightStride = 1;
        const int gradLocStride = 2;
        const int qidStride = numHeads * channels;
        const int dataValuePtrInitOffset = bCol * spatialSize * qidStride;

        for (int lCol = 0; lCol < numLevels; ++lCol)
        {
            const int levelStartId = dataLevelStartIndex[lCol];
            const int spatialHPtr = lCol << 1;
            const int spatialH = dataSpatialShapes[spatialHPtr];
            const int spatialW = dataSpatialShapes[spatialHPtr + 1];
            const int valuePtrOffset = dataValuePtrInitOffset + levelStartId * qidStride;
            const scalar_t* dataValuePtr = dataValue + valuePtrOffset;
            scalar_t* gradValuePtr = gradValue + valuePtrOffset;

            for (int pCol = 0; pCol < numPoint; ++pCol)
            {
                const scalar_t locW = dataSamplingLoc[dataLocWPtr];
                const scalar_t locH = dataSamplingLoc[dataLocWPtr + 1];
                const scalar_t weight = dataAttnWeight[dataWeightPtr];

                const scalar_t hIm = locH * spatialH - 0.5;
                const scalar_t wIm = locW * spatialW - 0.5;
                *(cacheGradSamplingLoc + (threadIdx.x << 1)) = 0;
                *(cacheGradSamplingLoc + ((threadIdx.x << 1) + 1)) = 0;
                *(cacheGradAttnWeight + threadIdx.x) = 0;
                if (hIm > -1 && wIm > -1 && hIm < spatialH && wIm < spatialW)
                {
                    ms_deform_attn_col2im_bilinear(dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol,
                        cCol, topGrad, weight, gradValuePtr, cacheGradSamplingLoc + (threadIdx.x << 1),
                        cacheGradAttnWeight + threadIdx.x);
                }

                __syncthreads();
                if (tid == 0)
                {
                    scalar_t _gradW = cacheGradSamplingLoc[0], _gradH = cacheGradSamplingLoc[1],
                             _gradA = cacheGradAttnWeight[0];
                    int sid = 2;
                    for (unsigned int tid = 1; tid < blockDim.x; ++tid)
                    {
                        _gradW += cacheGradSamplingLoc[sid];
                        _gradH += cacheGradSamplingLoc[sid + 1];
                        _gradA += cacheGradAttnWeight[tid];
                        sid += 2;
                    }

                    *gradSamplingLoc = _gradW;
                    *(gradSamplingLoc + 1) = _gradH;
                    *gradAttnWeight = _gradA;
                }
                __syncthreads();

                dataWeightPtr += 1;
                dataLocWPtr += 2;
                gradAttnWeight += gradWeightStride;
                gradSamplingLoc += gradLocStride;
            }
        }
    }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2(const int n, const scalar_t* grad_col,
    const scalar_t* dataValue, const int32_t* dataSpatialShapes, const int32_t* dataLevelStartIndex,
    const scalar_t* dataSamplingLoc, const scalar_t* dataAttnWeight, const int batchSize, const int spatialSize,
    const int numHeads, const int channels, const int numLevels, const int numQuery, const int numPoint,
    scalar_t* gradValue, scalar_t* gradSamplingLoc, scalar_t* gradAttnWeight)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        extern __shared__ int _s[];
        scalar_t* cacheGradSamplingLoc = (scalar_t*) _s;
        scalar_t* cacheGradAttnWeight = cacheGradSamplingLoc + 2 * blockDim.x;
        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int cCol = _temp % channels;
        _temp /= channels;
        const int samplingIndex = _temp;
        const int mCol = _temp % numHeads;
        _temp /= numHeads;
        const int qCol = _temp % numQuery;
        _temp /= numQuery;
        const int bCol = _temp;

        const scalar_t topGrad = grad_col[index];

        int dataWeightPtr = samplingIndex * numLevels * numPoint;
        int dataLocWPtr = dataWeightPtr << 1;
        const int gradSamplingPtr = dataWeightPtr;
        gradSamplingLoc += gradSamplingPtr << 1;
        gradAttnWeight += gradSamplingPtr;
        const int gradWeightStride = 1;
        const int gradLocStride = 2;
        const int qidStride = numHeads * channels;
        const int dataValuePtrInitOffset = bCol * spatialSize * qidStride;

        for (int lCol = 0; lCol < numLevels; ++lCol)
        {
            const int levelStartId = dataLevelStartIndex[lCol];
            const int spatialHPtr = lCol << 1;
            const int spatialH = dataSpatialShapes[spatialHPtr];
            const int spatialW = dataSpatialShapes[spatialHPtr + 1];
            const int valuePtrOffset = dataValuePtrInitOffset + levelStartId * qidStride;
            const scalar_t* dataValuePtr = dataValue + valuePtrOffset;
            scalar_t* gradValuePtr = gradValue + valuePtrOffset;

            for (int pCol = 0; pCol < numPoint; ++pCol)
            {
                const scalar_t locW = dataSamplingLoc[dataLocWPtr];
                const scalar_t locH = dataSamplingLoc[dataLocWPtr + 1];
                const scalar_t weight = dataAttnWeight[dataWeightPtr];

                const scalar_t hIm = locH * spatialH - 0.5;
                const scalar_t wIm = locW * spatialW - 0.5;
                *(cacheGradSamplingLoc + (threadIdx.x << 1)) = 0;
                *(cacheGradSamplingLoc + ((threadIdx.x << 1) + 1)) = 0;
                *(cacheGradAttnWeight + threadIdx.x) = 0;
                if (hIm > -1 && wIm > -1 && hIm < spatialH && wIm < spatialW)
                {
                    ms_deform_attn_col2im_bilinear(dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol,
                        cCol, topGrad, weight, gradValuePtr, cacheGradSamplingLoc + (threadIdx.x << 1),
                        cacheGradAttnWeight + threadIdx.x);
                }

                __syncthreads();

                for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0; s >>= 1, spre >>= 1)
                {
                    if (tid < s)
                    {
                        const unsigned int xid1 = tid << 1;
                        const unsigned int xid2 = (tid + s) << 1;
                        cacheGradAttnWeight[tid] += cacheGradAttnWeight[tid + s];
                        cacheGradSamplingLoc[xid1] += cacheGradSamplingLoc[xid2];
                        cacheGradSamplingLoc[xid1 + 1] += cacheGradSamplingLoc[xid2 + 1];
                        if (tid + (s << 1) < spre)
                        {
                            cacheGradAttnWeight[tid] += cacheGradAttnWeight[tid + (s << 1)];
                            cacheGradSamplingLoc[xid1] += cacheGradSamplingLoc[xid2 + (s << 1)];
                            cacheGradSamplingLoc[xid1 + 1] += cacheGradSamplingLoc[xid2 + 1 + (s << 1)];
                        }
                    }
                    __syncthreads();
                }

                if (tid == 0)
                {
                    *gradSamplingLoc = cacheGradSamplingLoc[0];
                    *(gradSamplingLoc + 1) = cacheGradSamplingLoc[1];
                    *gradAttnWeight = cacheGradAttnWeight[0];
                }
                __syncthreads();

                dataWeightPtr += 1;
                dataLocWPtr += 2;
                gradAttnWeight += gradWeightStride;
                gradSamplingLoc += gradLocStride;
            }
        }
    }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks(const int n, const scalar_t* grad_col,
    const scalar_t* dataValue, const int32_t* dataSpatialShapes, const int32_t* dataLevelStartIndex,
    const scalar_t* dataSamplingLoc, const scalar_t* dataAttnWeight, const int batchSize, const int spatialSize,
    const int numHeads, const int channels, const int numLevels, const int numQuery, const int numPoint,
    scalar_t* gradValue, scalar_t* gradSamplingLoc, scalar_t* gradAttnWeight)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        extern __shared__ int _s[];
        scalar_t* cacheGradSamplingLoc = (scalar_t*) _s;
        scalar_t* cacheGradAttnWeight = cacheGradSamplingLoc + 2 * blockDim.x;
        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int cCol = _temp % channels;
        _temp /= channels;
        const int samplingIndex = _temp;
        const int mCol = _temp % numHeads;
        _temp /= numHeads;
        const int qCol = _temp % numQuery;
        _temp /= numQuery;
        const int bCol = _temp;

        const scalar_t topGrad = grad_col[index];

        int dataWeightPtr = samplingIndex * numLevels * numPoint;
        int dataLocWPtr = dataWeightPtr << 1;
        const int gradSamplingPtr = dataWeightPtr;
        gradSamplingLoc += gradSamplingPtr << 1;
        gradAttnWeight += gradSamplingPtr;
        const int gradWeightStride = 1;
        const int gradLocStride = 2;
        const int qidStride = numHeads * channels;
        const int dataValuePtrInitOffset = bCol * spatialSize * qidStride;

        for (int lCol = 0; lCol < numLevels; ++lCol)
        {
            const int levelStartId = dataLevelStartIndex[lCol];
            const int spatialHPtr = lCol << 1;
            const int spatialH = dataSpatialShapes[spatialHPtr];
            const int spatialW = dataSpatialShapes[spatialHPtr + 1];
            const int valuePtrOffset = dataValuePtrInitOffset + levelStartId * qidStride;
            const scalar_t* dataValuePtr = dataValue + valuePtrOffset;
            scalar_t* gradValuePtr = gradValue + valuePtrOffset;

            for (int pCol = 0; pCol < numPoint; ++pCol)
            {
                const scalar_t locW = dataSamplingLoc[dataLocWPtr];
                const scalar_t locH = dataSamplingLoc[dataLocWPtr + 1];
                const scalar_t weight = dataAttnWeight[dataWeightPtr];

                const scalar_t hIm = locH * spatialH - 0.5;
                const scalar_t wIm = locW * spatialW - 0.5;
                *(cacheGradSamplingLoc + (threadIdx.x << 1)) = 0;
                *(cacheGradSamplingLoc + ((threadIdx.x << 1) + 1)) = 0;
                *(cacheGradAttnWeight + threadIdx.x) = 0;
                if (hIm > -1 && wIm > -1 && hIm < spatialH && wIm < spatialW)
                {
                    ms_deform_attn_col2im_bilinear(dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol,
                        cCol, topGrad, weight, gradValuePtr, cacheGradSamplingLoc + (threadIdx.x << 1),
                        cacheGradAttnWeight + threadIdx.x);
                }

                __syncthreads();

                for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0; s >>= 1, spre >>= 1)
                {
                    if (tid < s)
                    {
                        const unsigned int xid1 = tid << 1;
                        const unsigned int xid2 = (tid + s) << 1;
                        cacheGradAttnWeight[tid] += cacheGradAttnWeight[tid + s];
                        cacheGradSamplingLoc[xid1] += cacheGradSamplingLoc[xid2];
                        cacheGradSamplingLoc[xid1 + 1] += cacheGradSamplingLoc[xid2 + 1];
                        if (tid + (s << 1) < spre)
                        {
                            cacheGradAttnWeight[tid] += cacheGradAttnWeight[tid + (s << 1)];
                            cacheGradSamplingLoc[xid1] += cacheGradSamplingLoc[xid2 + (s << 1)];
                            cacheGradSamplingLoc[xid1 + 1] += cacheGradSamplingLoc[xid2 + 1 + (s << 1)];
                        }
                    }
                    __syncthreads();
                }

                if (tid == 0)
                {
                    atomicAdd(gradSamplingLoc, cacheGradSamplingLoc[0]);
                    atomicAdd(gradSamplingLoc + 1, cacheGradSamplingLoc[1]);
                    atomicAdd(gradAttnWeight, cacheGradAttnWeight[0]);
                }
                __syncthreads();

                dataWeightPtr += 1;
                dataLocWPtr += 2;
                gradAttnWeight += gradWeightStride;
                gradSamplingLoc += gradLocStride;
            }
        }
    }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_gm(const int n, const scalar_t* grad_col, const scalar_t* dataValue,
    const int32_t* dataSpatialShapes, const int32_t* dataLevelStartIndex, const scalar_t* dataSamplingLoc,
    const scalar_t* dataAttnWeight, const int batchSize, const int spatialSize, const int numHeads, const int channels,
    const int numLevels, const int numQuery, const int numPoint, scalar_t* gradValue, scalar_t* gradSamplingLoc,
    scalar_t* gradAttnWeight)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        int _temp = index;
        const int cCol = _temp % channels;
        _temp /= channels;
        const int samplingIndex = _temp;
        const int mCol = _temp % numHeads;
        _temp /= numHeads;
        const int qCol = _temp % numQuery;
        _temp /= numQuery;
        const int bCol = _temp;

        const scalar_t topGrad = grad_col[index];

        int dataWeightPtr = samplingIndex * numLevels * numPoint;
        int dataLocWPtr = dataWeightPtr << 1;
        const int gradSamplingPtr = dataWeightPtr;
        gradSamplingLoc += gradSamplingPtr << 1;
        gradAttnWeight += gradSamplingPtr;
        const int gradWeightStride = 1;
        const int gradLocStride = 2;
        const int qidStride = numHeads * channels;
        const int dataValuePtrInitOffset = bCol * spatialSize * qidStride;

        for (int lCol = 0; lCol < numLevels; ++lCol)
        {
            const int levelStartId = dataLevelStartIndex[lCol];
            const int spatialHPtr = lCol << 1;
            const int spatialH = dataSpatialShapes[spatialHPtr];
            const int spatialW = dataSpatialShapes[spatialHPtr + 1];
            const int valuePtrOffset = dataValuePtrInitOffset + levelStartId * qidStride;
            const scalar_t* dataValuePtr = dataValue + valuePtrOffset;
            scalar_t* gradValuePtr = gradValue + valuePtrOffset;

            for (int pCol = 0; pCol < numPoint; ++pCol)
            {
                const scalar_t locW = dataSamplingLoc[dataLocWPtr];
                const scalar_t locH = dataSamplingLoc[dataLocWPtr + 1];
                const scalar_t weight = dataAttnWeight[dataWeightPtr];

                const scalar_t hIm = locH * spatialH - 0.5;
                const scalar_t wIm = locW * spatialW - 0.5;
                if (hIm > -1 && wIm > -1 && hIm < spatialH && wIm < spatialW)
                {
                    ms_deform_attn_col2im_bilinear_gm(dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm,
                        mCol, cCol, topGrad, weight, gradValuePtr, gradSamplingLoc, gradAttnWeight);
                }
                dataWeightPtr += 1;
                dataLocWPtr += 2;
                gradAttnWeight += gradWeightStride;
                gradSamplingLoc += gradLocStride;
            }
        }
    }
}

template <typename scalar_t>
void ms_deformable_im2col_cuda(cudaStream_t stream, const scalar_t* dataValue, const int32_t* dataSpatialShapes,
    const int32_t* dataLevelStartIndex, const scalar_t* dataSamplingLoc, const scalar_t* dataAttnWeight,
    const int batchSize, const int spatialSize, const int numHeads, const int channels, const int numLevels,
    const int numQuery, const int numPoint, scalar_t* dataCol)
{
    const int numKernels = batchSize * numQuery * numHeads * channels;
    const int numActualKernels = batchSize * numQuery * numHeads * channels;
    const int numThreads = CUDA_NUM_THREADS;
    cudaError_t err = cudaSuccess;

    ms_deformable_im2col_gpu_kernel<scalar_t><<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(
        numKernels, dataValue, dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize,
        spatialSize, numHeads, channels, numLevels, numQuery, numPoint, dataCol);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
    }
}

template <typename scalar_t>
void ms_deformable_col2im_cuda(cudaStream_t stream, const scalar_t* grad_col, const scalar_t* dataValue,
    const int32_t* dataSpatialShapes, const int32_t* dataLevelStartIndex, const scalar_t* dataSamplingLoc,
    const scalar_t* dataAttnWeight, const int batchSize, const int spatialSize, const int numHeads, const int channels,
    const int numLevels, const int numQuery, const int numPoint, scalar_t* gradValue, scalar_t* gradSamplingLoc,
    scalar_t* gradAttnWeight)
{
    const int numThreads = (channels > CUDA_NUM_THREADS) ? CUDA_NUM_THREADS : channels;
    const int numKernels = batchSize * numQuery * numHeads * channels;
    const int numActualKernels = batchSize * numQuery * numHeads * channels;
    if (channels > 1024)
    {
        if ((channels & 1023) == 0)
        {
            ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks<scalar_t>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, numThreads * 3 * sizeof(scalar_t), stream>>>(
                    numKernels, grad_col, dataValue, dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc,
                    dataAttnWeight, batchSize, spatialSize, numHeads, channels, numLevels, numQuery, numPoint,
                    gradValue, gradSamplingLoc, gradAttnWeight);
        }
        else
        {
            ms_deformable_col2im_gpu_kernel_gm<scalar_t>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
        }
    }
    else
    {
        switch (channels)
        {
        case 1:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 1>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 2:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 2>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 4:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 4>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 8:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 8>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 16:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 16>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 32:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 32>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 64:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 64>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 128:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 128>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 256:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 256>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 512:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 512>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 1024:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 1024>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        default:
            if (channels < 64)
            {
                ms_deformable_col2im_gpu_kernel_shm_reduce_v1<scalar_t><<<GET_BLOCKS(numActualKernels, numThreads),
                    numThreads, numThreads * 3 * sizeof(scalar_t), stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            }
            else
            {
                ms_deformable_col2im_gpu_kernel_shm_reduce_v2<scalar_t><<<GET_BLOCKS(numActualKernels, numThreads),
                    numThreads, numThreads * 3 * sizeof(scalar_t), stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            }
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in ms_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
    }
}

#define CUDA_KERNEL_LOOP_RANGE(tid, nDataMin, nDataMax)                                                                \
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; ((tid >= (nDataMin)) && (tid < (nDataMax)));                 \
         tid += blockDim.x * gridDim.x)

__global__ void float2half_input(const int nData1, const int nData2, const int nData3, const float* data1Float,
    const float* data2Float, const float* data3Float, __half* data1Half, __half* data2Half, __half* data3Half)
{
    CUDA_KERNEL_LOOP(index, nData1)
    {
        data1Half[index] = __float2half(data1Float[index]);
        data2Half[index] = __float2half(data2Float[index]);
        data3Half[index] = __float2half(data3Float[index]);
    }

    CUDA_KERNEL_LOOP_RANGE(index, nData1, nData2)
    {
        data2Half[index] = __float2half(data2Float[index]);
        data3Half[index] = __float2half(data3Float[index]);
    }

    CUDA_KERNEL_LOOP_RANGE(index, nData2, nData3)
    {
        data3Half[index] = __float2half(data3Float[index]);
    }
}

__global__ void half2float_output(const int n_data, const __half* data_half, float* data_float)
{
    CUDA_KERNEL_LOOP(index, n_data)
    {
        data_float[index] = __half2float(data_half[index]);
    }
}

#endif