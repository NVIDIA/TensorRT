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

#include "coordConvACPlugin.h"
#include <cuda_fp16.h>

namespace nvinfer1
{
namespace plugin
{
template <typename T_DATA>
__global__ void kernelCopy(int N, T_DATA* inputs, T_DATA* outputs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        outputs[index] = inputs[index];
    }
    __syncthreads();
}

template <typename T_DATA>
__global__ void kernelAC(int N, int iH, int iW, float stepACh, float stepACw, T_DATA* outputs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int channelLength = N / 2;

    if (index < channelLength)
    {
        outputs[index] = -1.0 + (float) (index / iW) * stepACw;
        outputs[index + channelLength] = -1.0 + (float) ((index + channelLength) % iH) * stepACh;
    }
    __syncthreads();
}

template <typename T>
int inferenceAC(
    int batchSize, int iC, int iH, int iW, int oC, int oH, int oW, T* inputs, T* outputs, cudaStream_t stream)
{
    // NCHW
    const float coordsRange = 2.0;
    const int nThreads = 512;
    int lenCopy = iC * iH * iW;
    int lenAC = (oC * oH * oW) - lenCopy;

    int nBlocksCopy = (int) ((float) lenCopy / nThreads) + 1;
    int nBlocksAC = (int) ((float) lenAC / nThreads) + 1;

    float stepACh = coordsRange / (float) (iH - 1);
    float stepACw = coordsRange / (float) (iW - 1);

    for (int i = 0; i < batchSize; ++i)
    {
        // NOTE: kernelCopy kernel can be replaced with cudaMemcpy function
        kernelCopy<<<nBlocksCopy, nThreads, 0, stream>>>(lenCopy, inputs, outputs);
        outputs += lenCopy;

        kernelAC<<<nBlocksAC, nThreads, 0, stream>>>(lenAC, iH, iW, stepACh, stepACw, outputs);
        outputs += lenAC;
        inputs += lenCopy;
    }

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int CoordConvACPlugin::enqueue(
    int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    switch (iType)
    {
    case DataType::kFLOAT:
        return inferenceAC(batchSize, iC, iH, iW, oC, oH, oW, (float*) inputs[0], (float*) outputs[0], stream);
    case DataType::kHALF:
        return inferenceAC(batchSize, iC, iH, iW, oC, oH, oW, (__half*) inputs[0], (__half*) outputs[0], stream);
    case DataType::kINT8:
    case DataType::kUINT8:
    case DataType::kINT32:
    case DataType::kBOOL:
        break;
    }
    return 1;
}
} // namespace plugin
} // namespace nvinfer1
