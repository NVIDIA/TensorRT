/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*
 **************************************************************************
 * Modified from mmcv (https://github.com/open-mmlab/mmcv/tree/master/mmcv)
 * Copyright (c) OpenMMLab. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 [see LICENSE for details]
 * https://github.com/open-mmlab/mmcv/blob/master/LICENSE
 **************************************************************************
 */
#include <algorithm>
#include <assert.h>
#include <cuda_fp16.h>
#include <stdexcept>

#include "common/checkMacrosPlugin.h"
#include "modulatedDeformConvPluginKernel.h"

using namespace nvinfer1::pluginInternal;

template <typename T>
__device__ __forceinline__ T dmcnIm2colBilinear(
    T const* input, int32_t const dataWidth, int32_t const height, int32_t const width, float h, float w)
{
    if (h <= -1 || height <= h || w <= -1 || width <= w)
    {
        return 0;
    }
    int32_t hLow = floorf(h);
    int32_t wLow = floorf(w);
    int32_t hHigh = hLow + 1;
    int32_t wHigh = wLow + 1;

    T lh = h - hLow;
    T lw = w - wLow;
    T hh = 1 - lh, hw = 1 - lw;

    T v1 = 0;
    if (hLow >= 0 && wLow >= 0)
    {
        v1 = input[hLow * dataWidth + wLow];
    }
    T v2 = 0;
    if (hLow >= 0 && wHigh <= width - 1)
    {
        v2 = input[hLow * dataWidth + wHigh];
    }
    T v3 = 0;
    if (hHigh <= height - 1 && wLow >= 0)
    {
        v3 = input[hHigh * dataWidth + wLow];
    }
    T v4 = 0;
    if (hHigh <= height - 1 && wHigh <= width - 1)
    {
        v4 = input[hHigh * dataWidth + wHigh];
    }

    T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;
}

template <>
__device__ __forceinline__ __half dmcnIm2colBilinear(
    __half const* input, int32_t const dataWidth, int32_t const height, int32_t const width, float h, float w)
{
    if (h <= -1 || height <= h || w <= -1 || width <= w)
    {
        return 0;
    }
    int32_t hLow = floorf(h);
    int32_t wLow = floorf(w);
    int32_t hHigh = hLow + 1;
    int32_t wHigh = wLow + 1;

    half lh = __float2half(h - hLow);
    half lw = __float2half(w - wLow);
    half hh = __float2half(1) - lh;
    half hw = __float2half(1) - lw;

    half v1 = 0;
    if (hLow >= 0 && wLow >= 0)
    {
        v1 = input[hLow * dataWidth + wLow];
    }
    half v2 = 0;
    if (hLow >= 0 && wHigh <= width - 1)
    {
        v2 = input[hLow * dataWidth + wHigh];
    }
    half v3 = 0;
    if (hHigh <= height - 1 && wLow >= 0)
    {
        v3 = input[hHigh * dataWidth + wLow];
    }
    half v4 = 0;
    if (hHigh <= height - 1 && wHigh <= width - 1)
    {
        v4 = input[hHigh * dataWidth + wHigh];
    }

    half const w1 = __hmul(hh, hw);
    half const w2 = __hmul(hh, lw);
    half const w3 = __hmul(lh, hw);
    half const w4 = __hmul(lh, lw);

    half const val = __hadd(__hadd(__hmul(w1, v1), __hmul(w2, v2)), __hadd(__hmul(w3, v3), __hmul(w4, v4)));

    return val;
}


template <typename T>
__global__ void modulatedDeformableIm2colGpuKernel(int32_t const n, T const* dataIm, T const* dataOffset,
    T const* dataMask, int32_t const height, int32_t const width, int32_t const kernelH, int32_t const kernelW,
    int32_t const padH, int32_t const padW, int32_t const strideH, int32_t const strideW, int32_t const dilationH,
    int32_t const dilationW, int32_t const channelPerDeformableGroup, int32_t const batchSize,
    int32_t const numChannels, int32_t const deformableGroup, int32_t const heightCol, int32_t const widthCol,
    T* dataCol)
{
    for (int32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        // index index of output matrix
        int32_t const wCol = index % widthCol;
        int32_t const hCol = (index / widthCol) % heightCol;
        int32_t const bCol = (index / widthCol / heightCol) % batchSize;
        int32_t const cIm = (index / widthCol / heightCol) / batchSize;
        int32_t const cCol = cIm * kernelH * kernelW;

        // compute deformable group index
        int32_t const deformableGroupIndex = cIm / channelPerDeformableGroup;

        int32_t const hIn = hCol * strideH - padH;
        int32_t const wIn = wCol * strideW - padW;

        T* dataColPtr = dataCol + ((cCol * batchSize + bCol) * heightCol + hCol) * widthCol + wCol;
        T const* dataImPtr = dataIm + (bCol * numChannels + cIm) * height * width;
        T const* dataOffsetPtr = dataOffset
            + (bCol * deformableGroup + deformableGroupIndex) * 2 * kernelH * kernelW * heightCol * widthCol;

        T const* dataMaskPtr
            = dataMask + (bCol * deformableGroup + deformableGroupIndex) * kernelH * kernelW * heightCol * widthCol;

        for (int32_t i = 0; i < kernelH; ++i)
        {
            for (int32_t j = 0; j < kernelW; ++j)
            {
                int32_t const dataOffsetHPtr = ((2 * (i * kernelW + j)) * heightCol + hCol) * widthCol + wCol;
                int32_t const dataOffsetWPtr = ((2 * (i * kernelW + j) + 1) * heightCol + hCol) * widthCol + wCol;
                int32_t const dataMaskHwPtr = ((i * kernelW + j) * heightCol + hCol) * widthCol + wCol;
                T const offsetH = dataOffsetPtr[dataOffsetHPtr];
                T const offsetW = dataOffsetPtr[dataOffsetWPtr];
                T const mask = dataMaskPtr[dataMaskHwPtr];
                T val = static_cast<T>(0);
                T const hIm = hIn + i * dilationH + (float)offsetH;
                T const wIm = wIn + j * dilationW + (float)offsetW;
                val = dmcnIm2colBilinear(dataImPtr, width, height, width, hIm, wIm);
                *dataColPtr = val * mask;
                dataColPtr += batchSize * heightCol * widthCol;
            }
        }
    }
}

template <typename T>
cudaError_t trtModulatedDeformableIm2col(T const* dataIm, T const* dataOffset, T const* dataMask,
    int32_t const batchSize, int32_t const channels, int32_t const heightIm, int32_t const widthIm,
    int32_t const heightCol, int32_t const widthCol, int32_t const kernelH, int32_t const kernelW, int32_t const padH,
    int32_t const padW, int32_t const strideH, int32_t const strideW, int32_t const dilationH, int32_t const dilationW,
    int32_t const deformableGroup, T* dataCol, cudaStream_t stream)
{
    int32_t const channelPerDeformableGroup = channels / deformableGroup;
    int32_t const numKernels = channels * batchSize * heightCol * widthCol;

    modulatedDeformableIm2colGpuKernel<T><<<get_blocks(numKernels), THREADS_PER_BLOCK, 0, stream>>>(numKernels, dataIm,
        dataOffset, dataMask, heightIm, widthIm, kernelH, kernelW, padH, padW, strideH, strideW, dilationH, dilationW,
        channelPerDeformableGroup, batchSize, channels, deformableGroup, heightCol, widthCol, dataCol);

    PLUGIN_CHECK_CUDA(cudaPeekAtLastError());
    return cudaPeekAtLastError();
}

template <typename TScalar>
__global__ void outputAddBiasKernel(
    TScalar* output, TScalar const* bias, int32_t stepBatch, int32_t stepChannel, int32_t n)
{
    for (int32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        output[index] += bias[(index % stepBatch) / stepChannel];
    }
}

template <typename TScalar>
void outputAddBias(TScalar* output, TScalar const* bias, int32_t batch, int32_t channel, int32_t height, int32_t width,
    cudaStream_t stream)
{
    int32_t stepChannel = height * width;
    int32_t stepBatch = stepChannel * channel;
    int32_t n = stepBatch * batch;
    outputAddBiasKernel<<<get_blocks(n), THREADS_PER_BLOCK, 0, stream>>>(output, bias, stepBatch, stepChannel, n);
}

template <typename TScalar>
cudaError_t ModulatedDeformConvForwardCUDAKernelLauncher(TScalar const* input, TScalar const* weight,
    TScalar const* bias, TScalar const* offset, TScalar const* mask, TScalar* output, void* workspace, int32_t batch,
    int32_t channels, int32_t height, int32_t width, int32_t channelsOut, int32_t kernelW, int32_t kernelH,
    int32_t strideW, int32_t strideH, int32_t padW, int32_t padH, int32_t dilationW, int32_t dilationH, int32_t group,
    int32_t deformableGroup, int32_t im2colStep, cublasHandle_t cublasHandle, cudaStream_t stream)
{
    bool withBias = (bias != nullptr);

    im2colStep = std::min(int(batch), im2colStep);
    assert(batch % im2colStep == 0);

    int32_t const heightOut = (height + 2 * padH - (dilationH * (kernelH - 1) + 1)) / strideH + 1;
    int32_t const widthOut = (width + 2 * padW - (dilationW * (kernelW - 1) + 1)) / strideW + 1;

    TScalar* columns = (TScalar*) workspace;

    int32_t const inputStep = channels * height * width;
    int32_t const offsetStep = deformableGroup * kernelH * kernelW * 2 * heightOut * widthOut;
    int32_t const maskStep = deformableGroup * kernelH * kernelW * heightOut * widthOut;
    int32_t const outStep = channelsOut * heightOut * widthOut;
    int32_t const outGroupStep = outStep / group;
    int32_t const colGStep = channels * kernelW * kernelH / group * heightOut * widthOut;
    int32_t const weightGStep = channelsOut / group * channels / group * kernelH * kernelW;

    int32_t const m = channelsOut / group;
    int32_t const n = heightOut * widthOut;
    int32_t const k = channels / group * kernelH * kernelW;
    TScalar alpha = 1.;
    TScalar beta = 0.;

    for (int32_t b = 0; b < batch; b++)
    {
        TScalar const* inputStart = input + b * inputStep;
        TScalar const* offsetStart = offset + b * offsetStep;
        TScalar const* maskStart = mask + b * maskStep;
        trtModulatedDeformableIm2col<TScalar>(inputStart, offsetStart, maskStart, 1, channels, height, width, heightOut,
            widthOut, kernelH, kernelW, padH, padW, strideH, strideW, dilationH, dilationW, deformableGroup, columns,
            stream);

        for (int32_t g = 0; g < group; g++)
        {
            TScalar const* weightStart = weight + g * weightGStep;
            TScalar* colStart = columns + g * colGStep;
            TScalar* outBufferStart = output + b * outStep + g * outGroupStep;

            cublasGemmWrap<TScalar>(cublasHandle, stream, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, colStart, n, weightStart,
                k, &beta, outBufferStart, n);

            PLUGIN_CHECK_CUDA(cudaPeekAtLastError());
        }
    }

    if (withBias)
    {
        outputAddBias<TScalar>(output, bias, batch, channelsOut, heightOut, widthOut, stream);
    }

    return cudaPeekAtLastError();
}

void ModulatedDeformConvForwardCUDAKernelLauncherFloat(float const* input, float const* weight, float const* bias,
    float const* offset, float const* mask, float* output, void* workspace, int32_t batch, int32_t channels,
    int32_t height, int32_t width, int32_t channelsOut, int32_t kernelW, int32_t kernelH, int32_t strideW,
    int32_t strideH, int32_t padW, int32_t padH, int32_t dilationW, int32_t dilationH, int32_t group,
    int32_t deformableGroup, int32_t im2colStep, cublasHandle_t cublasHandle, cudaStream_t stream)
{
    ModulatedDeformConvForwardCUDAKernelLauncher<float>(input, weight, bias, offset, mask, output, workspace, batch,
        channels, height, width, channelsOut, kernelW, kernelH, strideW, strideH, padW, padH, dilationW, dilationH,
        group, deformableGroup, im2colStep, cublasHandle, stream);
}

void ModulatedDeformConvForwardCUDAKernelLauncherHalf(__half const* input, __half const* weight, __half const* bias,
    __half const* offset, __half const* mask, __half* output, void* workspace, int32_t batch, int32_t channels,
    int32_t height, int32_t width, int32_t channelsOut, int32_t kernelW, int32_t kernelH, int32_t strideW,
    int32_t strideH, int32_t padW, int32_t padH, int32_t dilationW, int32_t dilationH, int32_t group,
    int32_t deformableGroup, int32_t im2colStep, cublasHandle_t cublasHandle, cudaStream_t stream)
{
    ModulatedDeformConvForwardCUDAKernelLauncher<__half>(input, weight, bias, offset, mask, output, workspace, batch,
	channels, height, width, channelsOut, kernelW, kernelH, strideW, strideH, padW, padH, dilationW, dilationH,
	group, deformableGroup, im2colStep, cublasHandle, stream);
}
