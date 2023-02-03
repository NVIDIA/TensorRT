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
 * 
 * ************************************************************************
 * Modified from Pytorch
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * See https://github.com/pytorch/pytorch/blob/master/LICENSE for details
 * ************************************************************************
 * Modified from ONNX Runtime
 * Copyright (c) Microsoft Corporation
 * 
 * See https://github.com/microsoft/onnxruntime/blob/master/LICENSE for details
 * ************************************************************************
 */


#include <cuda.h>
#include "cuda_fp16.h"
#include "common/common.cuh"
#include "roiAlignKernel.h"

using half = __half;

__device__ half floatMax(half a, half b)
{
    #if __CUDA_ARCH__ >= 800
        return __hmax(a, b);
    #else
        return __float2half(max(__half2float(a), __half2float(b)));
    #endif
}

__device__ float floatMax(float a, float b)
{
    return max(a, b);
}

template <typename T>
__device__ T bilinearInterpolate(T const* bottomData, int32_t const height, int32_t const width, T y, T x,
    int32_t const isModeAvg, int32_t const index /* index for debug only*/)
{
    // deal with cases that inverse elements are out of feature map boundary
    if (y < static_cast<T>(-1.0) || y > static_cast<T>(height) || x < static_cast<T>(-1.0) || x > static_cast<T>(width))
    {
        // empty
        return 0;
    }

    if (y <= static_cast<T>(0))
    {
        y = 0;
    }
    if (x <= static_cast<T>(0))
    {
        x = 0;
    }

    int32_t yLow = static_cast<int32_t>(y);
    int32_t xLow = static_cast<int32_t>(x);
    int32_t yHigh;
    int32_t xHigh;

    if (yLow >= height - 1)
    {
        yHigh = yLow = height - 1;
        y = static_cast<T>(yLow);
    }
    else
    {
        yHigh = yLow + 1;
    }

    if (xLow >= width - 1)
    {
        xHigh = xLow = width - 1;
        x = static_cast<T>(xLow);
    }
    else
    {
        xHigh = xLow + 1;
    }

    T ly = y - static_cast<T>(yLow);
    T lx = x - static_cast<T>(xLow);
    T hy = static_cast<T>(1.) - ly, hx = static_cast<T>(1.) - lx;
    // do bilinear interpolation
    T v1 = bottomData[yLow * width + xLow];
    T v2 = bottomData[yLow * width + xHigh];
    T v3 = bottomData[yHigh * width + xLow];
    T v4 = bottomData[yHigh * width + xHigh];
    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    T val;
    if (isModeAvg)
    {
        val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4); // mode Avg
    }
    else
    {
        val = floatMax(floatMax(floatMax(w1 * v1, w2 * v2), w3 * v3), w4 * v4); // mode Max
    }

    return val;
}

template <typename T>
__global__ void RoIAlignForward(int32_t const nthreads, T const* bottomData, T const spatialScale, int32_t const channels,
    int32_t const height, int32_t const width, int32_t const pooledHeight, int32_t const pooledWidth, int32_t const samplingRatio,
    T const* bottomRois, T* topData, int32_t const isModeAvg, int32_t const* batchIndicesPtr,
    int32_t const aligned)
{
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x)
    {
        // (n, c, ph, pw) is an element in the pooled output
        int32_t pw = index % pooledWidth;
        int32_t ph = (index / pooledWidth) % pooledHeight;
        int32_t c = (index / pooledWidth / pooledHeight) % channels;
        int32_t n = index / pooledWidth / pooledHeight / channels;

        T const* offsetBottomRois = bottomRois + n * 4;
        auto const roiBatchInd = batchIndicesPtr[n];

        bool continuousCoordinate = aligned;
        // Do not using rounding; this implementation detail is critical
        T roiOffset = static_cast<T>(continuousCoordinate ? 0.5 : 0);
        T roiStartW = offsetBottomRois[0] * spatialScale - roiOffset;
        T roiStartH = offsetBottomRois[1] * spatialScale - roiOffset;
        T roiEndW = offsetBottomRois[2] * spatialScale - roiOffset;
        T roiEndH = offsetBottomRois[3] * spatialScale - roiOffset;

        T roiWidth = roiEndW - roiStartW;
        T roiHeight = roiEndH - roiStartH;
        if (!continuousCoordinate)
        { // backward compatiblity
            // Force malformed ROIs to be 1x1
            roiWidth = floatMax(roiWidth, static_cast<T>(1.));
            roiHeight = floatMax(roiHeight, static_cast<T>(1.));
        }
        T binSizeH = static_cast<T>(roiHeight) / static_cast<T>(pooledHeight);
        T binSizeW = static_cast<T>(roiWidth) / static_cast<T>(pooledWidth);

        T const* offsetBottomData = bottomData + static_cast<int32_t>((roiBatchInd * channels + c) * height * width);

        // We use roiBinGrid to sample the grid and mimic integral
        int32_t roiBinGridH;
        if (samplingRatio > 0)
        {
            roiBinGridH = samplingRatio;
        }
        else
        {
            roiBinGridH = ceilf(roiHeight / static_cast<T>(pooledHeight));
        }

        int32_t roiBinGridW;
        if (samplingRatio > 0)
        {
            roiBinGridW = samplingRatio;
        }
        else
        {
            roiBinGridW = ceilf(roiWidth / static_cast<T>(pooledWidth));
        }
        // We do average (integral) pooling inside a bin
        T const count = roiBinGridH * roiBinGridW; // e.g. = 4

        T const yOff = roiStartH + static_cast<T>(ph) * binSizeH;
        T const yFac = binSizeH / static_cast<T>(roiBinGridH);

        T const xOff = roiStartW + static_cast<T>(pw) * binSizeW;
        T const xFac = binSizeW / static_cast<T>(roiBinGridW);

        T outputVal = 0.;
        bool maxFlag = false;
        for (int32_t iy = 0; iy < roiBinGridH; iy++) // e.g., iy = 0, 1
        {
            T const y = yOff + static_cast<T>(iy + .5F) * yFac; // e.g., 0.5, 1.5
            for (int32_t ix = 0; ix < roiBinGridW; ix++)
            {
                T const x = xOff + static_cast<T>(ix + .5F) * xFac;

                T val = bilinearInterpolate(offsetBottomData, height, width, y, x, isModeAvg, index);

                if (isModeAvg)
                {
                    outputVal += val;
                }
                else
                {
                    if (!maxFlag)
                    {
                        outputVal = val;
                        maxFlag = true;
                    }
                    else
                    {
                        outputVal = floatMax(outputVal, val);
                    }
                }
            }
        }
        if (isModeAvg)
        {
            outputVal = outputVal / count;
        }

        topData[index] = outputVal;
    }
}

template <typename T>
cudaError_t RoiAlignImpl(cudaStream_t stream, int32_t const maxThreadsPerBlock, T const* bottomData, T const spatialScale,
    int32_t const numRois, int32_t const channels, int32_t const height, int32_t const width, int32_t const pooledHeight,
    int32_t const pooledWidth, int32_t const samplingRatio, T const* bottomRois, T* topData, int32_t const isModeAvg,
    int32_t const* batchIndicesPtr, int32_t const aligned)
{
    PLUGIN_ASSERT(bottomData != nullptr);
    PLUGIN_ASSERT(bottomRois != nullptr);
    PLUGIN_ASSERT(batchIndicesPtr != nullptr);
    PLUGIN_ASSERT(topData != nullptr);

    PLUGIN_ASSERT(numRois >= 0);
    PLUGIN_ASSERT(maxThreadsPerBlock > 0);

    PLUGIN_ASSERT(height > 0);
    PLUGIN_ASSERT(width > 0);
    PLUGIN_ASSERT(pooledHeight > 0);
    PLUGIN_ASSERT(pooledWidth > 0);
    PLUGIN_ASSERT(samplingRatio >= 0);
    PLUGIN_ASSERT(isModeAvg == 0 || isModeAvg == 1);
    PLUGIN_ASSERT(static_cast<float>(spatialScale) > 0.0F);
    PLUGIN_ASSERT(aligned == 0 || aligned == 1);

    int32_t const outputSize = numRois * channels * pooledHeight * pooledWidth;

    int32_t blocksPerGrid = static_cast<int32_t>(ceil(static_cast<float>(outputSize)
        / maxThreadsPerBlock)); 

    RoIAlignForward<T><<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(outputSize,// nthreads
        bottomData,                                                                 // bottomData
        spatialScale,                                                               // spatialScale
        channels,                                                                   // channels
        height,                                                                     // height
        width,                                                                      // width
        pooledHeight,                                                               // pooledHeight
        pooledWidth,                                                                // pooledWidth
        samplingRatio,                                                              // samplingRatio
        bottomRois,                                                                 // bottomRois
        topData,                                                                    // topData
        isModeAvg,                                                                  // isModeAvg
        batchIndicesPtr,                                                            // batchIndicesPtr
        aligned);

    return cudaGetLastError();
}

#define SPECIALIZED_IMPL(T)                                                                                            \
    template cudaError_t RoiAlignImpl<T>(cudaStream_t stream, int32_t const maxThreadsPerBlock, T const* bottomData,   \
        T const spatialScale, int32_t const numRois, int32_t const channels, int32_t const height,                     \
        int32_t const width, int32_t const pooledHeight, int32_t const pooledWidth, int32_t const samplingRatio,       \
        T const* bottomRois, T* topData, int32_t const isModeAvg, int32_t const* batchIndicesPtr,                      \
        int32_t const aligned);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(half)
