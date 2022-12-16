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
#include "common/kernel.h"

template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA) __global__ void softmaxKernel(const float* input, const int n, const int batch,
    const int batchOffset, const int groups, const int groupOffset, const int stride, const float temp, float* output)
{
    int id = blockIdx.x * nthdsPerCTA + threadIdx.x;
    if (id < batch * groups)
    {
        int b = id / groups;
        int g = id % groups;
        float sum = 0.;
        // Initialze largest to be the smallest float number.
        float largest = -3.402823466e+38;
        int offset = b * batchOffset + g * groupOffset;
        // Find the largest digits before softmax
        for (int i = 0; i < n; ++i)
        {
            float val = input[i * stride + offset];
            largest = (val > largest) ? val : largest;
        }
        // Softmax for a group of candidate classes
        // Calculate exponentials
        for (int i = 0; i < n; ++i)
        {
            /*
             * Here we used a trick to prevent numeric overflow
             * xm = max{x_1, x_2, ..., x_n}
             * e^{x_1} / (e^{x_1} + e^{x_2} + e^{x_n}) = e^{x_1 - xm} / (e^{x_1 - xm} + e^{x_2 - xm} + e^{x_n - xm})
             */
            float e = exp(input[i * stride + offset] / temp - largest / temp);
            sum += e;
            output[i * stride + offset] = e;
        }
        // Normalize
        for (int i = 0; i < n; ++i)
            output[i * stride + offset] /= sum;
    }
}

template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
    __global__ void activateKernel(float* data,
                                   const int range)
{
    int i = blockIdx.x * nthdsPerCTA + threadIdx.x;
    // Sigmoid function
    if (i < range)
        data[i] = 1. / (1. + exp(-data[i]));
}

pluginStatus_t regionGPU(
    cudaStream_t stream,
    const int batch,
    const int C,
    const int H,
    const int W,
    const int num,
    const int coords,
    const int classes,
    const bool hasSoftmaxTree,
    const nvinfer1::plugin::softmaxTree* smTree,
    const float* input,
    float* output)
{
    const int BS = 512;
    const int GS1 = (2 * H * W + BS - 1) / BS;
    const int GS2 = (H * W + BS - 1) / BS;
    // Applying sigmoid activations
    for (int b = 0; b < batch; ++b)
    {
        for (int n = 0; n < num; ++n)
        {
            // Apply sigmoid activation for the encoded center coordinates t_x, and t_y
            int index = b * C * H * W + n * H * W * (coords + classes + 1);
            activateKernel<BS><<<GS1, BS, 0, stream>>>(output + index, 2 * H * W);
            /*
             * Apply sigmoid for the encoded objectness t_o
             * + 4 * H * W because we want to skip the first four coordinates t_x, t_y, t_w, t_h
             * Chaning 4 * H * W to + coords * H * W will not make this function more general
             * Since we already assumed the information layout in the channels of the input tensor
             */
            index = b * C * H * W + n * H * W * (coords + classes + 1) + 4 * H * W;
            activateKernel<BS><<<GS2, BS, 0, stream>>>(output + index, H * W);
        }
    }
    const int GS3 = (batch * num * H * W + BS - 1) / BS;
    // Applying softmax activations
    if (hasSoftmaxTree)
    {
        // Softmax for hierarchical classification
        // The first 5 elements are t_x, t_y, t_w, t_h, t_o which we don't need to apply softmax activation
        int count = 5;
        // Only groups and groupSize information is useful for this plugin
        // Applying softmax activation sequentially for each group of candidate classes
        for (int i = 0; i < smTree->groups; ++i)
        {
            int groupSize = smTree->groupSize[i];
            softmaxKernel<BS><<<GS3, BS, 0, stream>>>(input + count * H * W, groupSize, batch * num, (C * H * W / num), H * W, 1, H * W, 1., output + count * H * W);
            count += groupSize;
        }
    }
    else
    {
        // Softmax for non-hierarchical classificiation
        softmaxKernel<BS><<<GS3, BS, 0, stream>>>(input + 5 * H * W, classes, batch * num, (C * H * W / num), H * W, 1, H * W, 1., output + 5 * H * W);
    }

    return STATUS_SUCCESS;
}

pluginStatus_t regionInference(cudaStream_t stream, const int batch, const int C, const int H, const int W,
    const int num, const int coords, const int classes, const bool hasSoftmaxTree,
    const nvinfer1::plugin::softmaxTree* smTree, const void* input, void* output)
{
    PLUGIN_CHECK(cudaMemcpyAsync(output, input, batch * C * H * W * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    return regionGPU(
        stream, batch, C, H, W, num, coords, classes, hasSoftmaxTree, smTree, (const float*) input, (float*) output);
}
