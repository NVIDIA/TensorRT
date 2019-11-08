/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "kernel.h"
#include "reducedMath.h"
#include <iostream>

using nvinfer1::rt::reduced_divisor;
template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
__global__ void gridAnchorKernel(
    const GridAnchorParameters param,
    const int numAspectRatios,
    reduced_divisor divObj,
    const float* widths,
    const float* heights,
    float* outputData
    )
{
    // output dims: (H, W, param.numMinSize, (1+haveMaxSize+numAR-1), 4)
    const int dim = param.H * param.W * numAspectRatios;

    /*
     * Parameters used to calculate the bounding box coordinates back to input image scale
     * Normally we calculate the anchorStride = image_input_size (in pixel) / feature_map_size
     * Here we do not use image_input_size for the moment
     * Instead we use 1.0
     * The coordinates calculated are scaled by the input image size. 
     * Most of the coordinates will be in a range of [0, 1], except for the bounding box coordinates going outside of the image
     * Every coordinate will go back to the pixel coordinates in the input image if being multiplied by image_input_size
     * Here we implicitly assumes the image input and feature map are square
     */
    float anchorStride = (1.0 / param.H);
    float anchorOffset = 0.5 * anchorStride;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dim)
    {
        return;
    }

    int arId, currIndex;
    divObj.divmod(tid, currIndex, arId);

    const int w = currIndex % param.W;
    const int h = currIndex / param.W;

    // Center coordinates
    float yC = h * anchorStride + anchorOffset;
    float xC = w * anchorStride + anchorOffset;
            
    // x_min, y_min
    float xMin = xC - 0.5 * widths[arId];
    float yMin = yC - 0.5 * heights[arId];

    // x_max, y_max
    float xMax = xC + 0.5 * widths[arId];
    float yMax = yC + 0.5 * heights[arId];

    outputData[tid * 4] = xMin;
    outputData[tid * 4 + 1] = yMin;
    outputData[tid * 4 + 2] = xMax;
    outputData[tid * 4 + 3] = yMax;

    // Remember to move the output cursor
    float* output = outputData + dim * 4;

    // Simply copying the variance
    output[tid * 4] = param.variance[0];
    output[tid * 4 + 1] = param.variance[1];
    output[tid * 4 + 2] = param.variance[2];
    output[tid * 4 + 3] = param.variance[3];
}

using nvinfer1::rt::reduced_divisor;
template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
__global__ void gridAnchorRectKernel(
    const GridAnchorParameters param,
    const int numAspectRatios,
    reduced_divisor divObj,
    const float* widths,
    const float* heights,
    float* outputData
    )
{
    // output dims: (H, W, param.numMinSize, (1+haveMaxSize+numAR-1), 4)
    const int dim = param.H * param.W * numAspectRatios;

    /*
     * Parameters used to calculate the bounding box coordinates back to input image scale
     * Normally we calculate the anchorStride = image_input_size (in pixel) / feature_map_size
     * Here we do not use image_input_size for the moment
     * Instead we use 1.0
     * The coordinates calculated are scaled by the input image size. 
     * Most of the coordinates will be in a range of [0, 1], except for the bounding box coordinates going outside of the image
     * Every coordinate will go back to the pixel coordinates in the input image if being multiplied by image_input_size
     * Here we implicitly assumes the image input and feature map are square
     */
    float anchorStrideH = (1.0 / param.H);
    float anchorStrideW = (1.0 / param.W);
    float anchorOffsetH = 0.5 * anchorStrideH;
    float anchorOffsetW = 0.5 * anchorStrideW;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dim)
    {
        return;
    }

    int arId, currIndex;
    divObj.divmod(tid, currIndex, arId);

    const int w = currIndex % param.W;
    const int h = currIndex / param.W;
            
    // Center coordinates
    float yC = h * anchorStrideH + anchorOffsetH;
    float xC = w * anchorStrideW + anchorOffsetW;

    // x_min, y_min 
    float xMin = xC - 0.5 * widths[arId];
    float yMin = yC - 0.5 * heights[arId];

    // x_max, y_max 
    float xMax = xC + 0.5 * widths[arId];
    float yMax = yC + 0.5 * heights[arId];

    outputData[tid * 4] = xMin;
    outputData[tid * 4 + 1] = yMin;
    outputData[tid * 4 + 2] = xMax;
    outputData[tid * 4 + 3] = yMax;

    // Remember to move the output cursor
    float* output = outputData + dim * 4;

    // Simply copying the variance
    output[tid * 4] = param.variance[0];
    output[tid * 4 + 1] = param.variance[1];
    output[tid * 4 + 2] = param.variance[2];
    output[tid * 4 + 3] = param.variance[3];
}

pluginStatus_t anchorGridInference(
    cudaStream_t stream,
    const GridAnchorParameters param,
    const int numAspectRatios,
    const void* widths,
    const void* heights,
    void* outputData
    )
{
    const int dim = param.H * param.W * numAspectRatios;
    reduced_divisor divObj(numAspectRatios);
    if (dim > 5120)
    {
        const int BS = 128;
        const int GS = (dim + BS - 1) / BS;
        gridAnchorKernel<BS><<<GS, BS, 0, stream>>>(param, numAspectRatios, divObj,
            (const float*) widths, (const float*) heights,
            (float*) outputData);

    }
    else
    {
        const int BS = 32;
        const int GS = (dim + BS - 1) / BS;
        gridAnchorKernel<BS><<<GS, BS, 0, stream>>>(param, numAspectRatios, divObj,
            (const float*) widths, (const float*) heights,
            (float*) outputData);
    }
    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}
