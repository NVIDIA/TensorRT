/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_MODULATED_DEFORM_CONV_PLUGIN_KERNEL_H
#define TRT_MODULATED_DEFORM_CONV_PLUGIN_KERNEL_H

#include "commonCudaHelper.h"
#include "modulatedDeformConvCudaHelper.h"
#include <cstdint>
#include <float.h>
#include <cuda_fp16.h>

template <typename T>
__device__ __forceinline__ T dmcnIm2colBilinear(
    T const* input, int32_t const dataWidth, int32_t const height, int32_t const width, float h, float w);

template <>
__device__ __forceinline__ __half dmcnIm2colBilinear(
    __half const* input, int32_t const dataWidth, int32_t const height, int32_t const width, float h, float w);

template <typename T>
__global__ void modulatedDeformableIm2colGpuKernel(int32_t const n, T const* dataIm, T const* dataOffset,
    T const* dataMask, int32_t const height, int32_t const width, int32_t const kernelH, int32_t const kernelW,
    int32_t const padH, int32_t const padW, int32_t const strideH, int32_t const strideW, int32_t const dilationH,
    int32_t const dilationW, int32_t const channelPerDeformableGroup, int32_t const batchSize,
    int32_t const numChannels, int32_t const deformableGroup, int32_t const heightCol, int32_t const widthCol,
    T* dataCol);

template <typename TScalar>
__global__ void outputAddBiasKernel(
    TScalar* output, TScalar const* bias, int32_t stepBatch, int32_t stepChannel, int32_t n);
template <typename T>
cudaError_t trtModulatedDeformableIm2col(T const* dataIm, T const* dataOffset, T const* dataMask,
    int32_t const batchSize, int32_t const channels, int32_t const heightIm, int32_t const widthIm,
    int32_t const heightCol, int32_t const widthCol, int32_t const kernelH, int32_t const kernelW, int32_t const padH,
    int32_t const padW, int32_t const strideH, int32_t const strideW, int32_t const dilationH, int32_t const dilationW,
    int32_t const deformableGroup, T* dataCol, cudaStream_t stream);

template <typename TScalar>
void outputAddBias(TScalar* output, TScalar const* bias, int32_t batch, int32_t channel, int32_t height, int32_t width,
    cudaStream_t stream);

template <typename TScalar>
cudaError_t ModulatedDeformConvForwardCUDAKernelLauncher(TScalar const* input, TScalar const* weight,
    TScalar const* bias, TScalar const* offset, TScalar const* mask, TScalar* output, void* workspace, int32_t batch,
    int32_t channels, int32_t height, int32_t width, int32_t channelsOut, int32_t kernelW, int32_t kernelH,
    int32_t strideW, int32_t strideH, int32_t padW, int32_t padH, int32_t dilationW, int32_t dilationH, int32_t group,
    int32_t deformableGroup, int32_t im2colStep, nvinfer1::pluginInternal::cublasHandle_t cublasHandle,
    cudaStream_t stream);

void ModulatedDeformConvForwardCUDAKernelLauncherFloat(float const* input, float const* weight, float const* bias,
    float const* offset, float const* mask, float* output, void* workspace, int32_t batch, int32_t channels,
    int32_t height, int32_t width, int32_t channelsOut, int32_t kernelW, int32_t kernelH, int32_t strideW,
    int32_t strideH, int32_t padW, int32_t padH, int32_t dilationW, int32_t dilationH, int32_t group,
    int32_t deformableGroup, int32_t im2colStep, nvinfer1::pluginInternal::cublasHandle_t cublasHandle,
    cudaStream_t stream);

void ModulatedDeformConvForwardCUDAKernelLauncherHalf(half const* input, half const* weight, half const* bias,
    half const* offset, half const* mask, half* output, void* workspace, int32_t batch, int32_t channels,
    int32_t height, int32_t width, int32_t channelsOut, int32_t kernelW, int32_t kernelH, int32_t strideW,
    int32_t strideH, int32_t padW, int32_t padH, int32_t dilationW, int32_t dilationH, int32_t group,
    int32_t deformableGroup, int32_t im2colStep, nvinfer1::pluginInternal::cublasHandle_t cublasHandle,
    cudaStream_t stream);

#endif // TRT_MODULATED_DEFORM_CONV_PLUGIN_KERNEL_H
