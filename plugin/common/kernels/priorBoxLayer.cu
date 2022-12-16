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
#include "NvInferPluginUtils.h"
#include "common/kernel.h"
#include "reducedMathPlugin.h"
#include <iostream>

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using nvinfer1::plugin::ReducedDivisor;

template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA) __global__ void priorBoxKernel(PriorBoxParameters param, const int H, const int W,
    const int numPriors, const int numAspectRatios, const float* minSize, const float* maxSize,
    const float* aspectRatios, float* outputData)
{
    // output dims: (H, W, param.numMinSize, (1+haveMaxSize+numAR-1), 4)
    const int dim = H * W * numPriors;
    const bool haveMaxSize = param.numMaxSize > 0;
    const int dimAR = (haveMaxSize ? 1 : 0) + numAspectRatios;
    for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x;
         i < dim; i += gridDim.x * nthdsPerCTA)
    {
        const int w = (i / numPriors) % W;
        const int h = (i / numPriors) / W;
        // Usually param.offset == 0.5
        // Calucate the center of prior box at the input image scale
        const float centerX = (w + param.offset) * param.stepW;
        const float centerY = (h + param.offset) * param.stepH;
        // Minimum size index
        const int minSizeId = (i / dimAR) % param.numMinSize;
        // Aspect ratio index
        const int arId = i % dimAR;
        // Generate square pior box of aspect ratio of 1.0, edge length of minSize[minSizeId]
        if (arId == 0)
        {
            const float boxW = minSize[minSizeId];
            const float boxH = boxW;
            float x, y, z, w;
            // Calculate [x_topleft, y_topleft, x_bottomright, y_bottomright]
            // Coordinates were scaled to [0, 1] against the width or height of the original input image
            x = (centerX - boxW / 2.0f) / param.imgW;
            y = (centerY - boxH / 2.0f) / param.imgH;
            z = (centerX + boxW / 2.0f) / param.imgW;
            w = (centerY + boxH / 2.0f) / param.imgH;
            // If we decided to clip the prior box make sure all the bounding box are inside the original input image
            if (param.clip)
            {
                x = min(max(x, 0.0f), 1.0f);
                y = min(max(y, 0.0f), 1.0f);
                z = min(max(z, 0.0f), 1.0f);
                w = min(max(w, 0.0f), 1.0f);
            }
            // Copy the bounding box coordinates to output
            outputData[i * 4] = x;
            outputData[i * 4 + 1] = y;
            outputData[i * 4 + 2] = z;
            outputData[i * 4 + 3] = w;
        }
        // If have maxSize
        // Generate square pior box for aspect ratio of 1.0, edge length of sqrt(minSize[minSizeId] * maxSize[minSizeId])
        // Described in SSD paper page 6
        else if (haveMaxSize && arId == 1)
        {
            const float boxW = sqrt(minSize[minSizeId] * maxSize[minSizeId]);
            const float boxH = boxW;
            float x, y, z, w;
            x = (centerX - boxW / 2.0f) / param.imgW;
            y = (centerY - boxH / 2.0f) / param.imgH;
            z = (centerX + boxW / 2.0f) / param.imgW;
            w = (centerY + boxH / 2.0f) / param.imgH;
            if (param.clip)
            {
                x = min(max(x, 0.0f), 1.0f);
                y = min(max(y, 0.0f), 1.0f);
                z = min(max(z, 0.0f), 1.0f);
                w = min(max(w, 0.0f), 1.0f);
            }
            outputData[i * 4] = x;
            outputData[i * 4 + 1] = y;
            outputData[i * 4 + 2] = z;
            outputData[i * 4 + 3] = w;
        }
        // Generate other bouding boxes with aspect ratios of not one.
        else
        {
            const int arOffset = haveMaxSize ? arId - 1 : arId; // skip aspectRatios[0] which is 1
            const float boxW = minSize[minSizeId] * sqrt(aspectRatios[arOffset]);
            const float boxH = minSize[minSizeId] / sqrt(aspectRatios[arOffset]);
            float x, y, z, w;
            x = (centerX - boxW / 2.0f) / param.imgW;
            y = (centerY - boxH / 2.0f) / param.imgH;
            z = (centerX + boxW / 2.0f) / param.imgW;
            w = (centerY + boxH / 2.0f) / param.imgH;
            if (param.clip)
            {
                x = min(max(x, 0.0f), 1.0f);
                y = min(max(y, 0.0f), 1.0f);
                z = min(max(z, 0.0f), 1.0f);
                w = min(max(w, 0.0f), 1.0f);
            }
            outputData[i * 4] = x;
            outputData[i * 4 + 1] = y;
            outputData[i * 4 + 2] = z;
            outputData[i * 4 + 3] = w;
        }
    }
    // Simply copy variance to from the parameter to output
    float* output = outputData + dim * 4;
    for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x;
         i < dim; i += gridDim.x * nthdsPerCTA)
    {
        float x, y, z, w;
        x = param.variance[0];
        y = param.variance[1];
        z = param.variance[2];
        w = param.variance[3];
        output[i * 4] = x;
        output[i * 4 + 1] = y;
        output[i * 4 + 2] = z;
        output[i * 4 + 3] = w;
    }
}

pluginStatus_t priorBoxGpu(
    cudaStream_t stream,
    const PriorBoxParameters param,
    const int H,
    const int W,
    const int numPriors,
    const int numAspectRatios,
    const void* minSize,
    const void* maxSize,
    const void* aspectRatios,
    void* outputData)
{
    const int dim = H * W * numPriors;
    if (dim > 5120)
    {
        const int BS = 128;
        const int GS = (dim + BS - 1) / BS;
        priorBoxKernel<BS><<<GS, BS, 0, stream>>>(param, H, W, numPriors, numAspectRatios,
                                                  (const float*) minSize, (const float*) maxSize,
                                                  (const float*) aspectRatios, (float*) outputData);
        CSC(cudaGetLastError(), STATUS_FAILURE);
        return STATUS_SUCCESS;
    }
    else
    {
        const int BS = 32;
        const int GS = (dim + BS - 1) / BS;
        priorBoxKernel<BS><<<GS, BS, 0, stream>>>(param, H, W, numPriors, numAspectRatios,
                                                  (const float*) minSize, (const float*) maxSize,
                                                  (const float*) aspectRatios, (float*) outputData);
        CSC(cudaGetLastError(), STATUS_FAILURE);
        return STATUS_SUCCESS;
    }
}

pluginStatus_t priorBoxInference(cudaStream_t stream, const PriorBoxParameters param, const int H, const int W,
    const int numPriors, const int numAspectRatios, const void* minSize, const void* maxSize, const void* aspectRatios,
    void* outputData)
{
    PLUGIN_ASSERT(param.numMaxSize >= 0);
    if (param.numMaxSize)
        return priorBoxGpu(stream, param, H, W, numPriors, numAspectRatios, minSize, maxSize, aspectRatios, outputData);
    else
        return priorBoxGpu(stream, param, H, W, numPriors, numAspectRatios,
                           minSize, nullptr, aspectRatios, outputData);
}

namespace nvinfer1
{
namespace plugin
{
pluginStatus_t priorBoxInference(cudaStream_t stream, const PriorBoxParameters param, const int H, const int W,
    const int numPriors, const int numAspectRatios, const void* minSize, const void* maxSize, const void* aspectRatios,
    void* outputData)
{
    PLUGIN_ASSERT(param.numMaxSize >= 0);
    if (param.numMaxSize)
        return priorBoxGpu(stream, param, H, W, numPriors, numAspectRatios, minSize, maxSize, aspectRatios, outputData);
    else
        return priorBoxGpu(stream, param, H, W, numPriors, numAspectRatios,
                           minSize, nullptr, aspectRatios, outputData);
}
} // namespace nvinfer1
} // namespace plugin
