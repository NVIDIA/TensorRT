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
#include "reducedMathPlugin.h"

using namespace nvinfer1::plugin; // for ReducedDivisor

template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
    __global__ void reorgKernel(
        const float* input, // input tensor of shape (batch, C, H, W)
        const int volume,   // note that volumes of input and output tensors are the same
        ReducedDivisor batch,
        ReducedDivisor C,
        ReducedDivisor H,
        ReducedDivisor W,
        ReducedDivisor C_out,
        ReducedDivisor stride,
        float* output) // output tensor of shape (batch, C * stride * stride, H / stride, W / stride)
{
    /*
     * Reference
     * https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/src/blas_kernels.cu#L370
     * https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/src/blas.c#L9
     */

    // outIndex is row-major position of input coordinates
    for (int outIndex = blockIdx.x * nthdsPerCTA + threadIdx.x; outIndex < volume; outIndex += nthdsPerCTA)
    {
        int i = outIndex;

        // calculate output coordinates from outIndex
        int outW, outH, outC;
        W.divmod(i, i, outW);
        H.divmod(i, i, outH);
        C.divmod(i, i, outC);
        int outN = i;

        // calculate input coordinates based on output coordinates
        // offset is [0, 1, ..., stride * stride - 1] = posH * stride + posW
        int offset, inC, posH, posW;
        C_out.divmod(outC, offset, inC);
        stride.divmod(offset, posH, posW);
        int inH = outH * stride.get() + posH;
        int inW = outW * stride.get() + posW;
        int inN = outN;

        // inIndex is row-major position of input coordinates
        int inIndex = inW + W.get() * stride.get() * (inH + H.get() * stride.get() * (inC + C_out.get() * inN));

        output[outIndex] = input[inIndex];
    }
}

pluginStatus_t reorgGPU(
    cudaStream_t stream,
    const int batch,
    const int C,
    const int H,
    const int W,
    const int stride,
    const float* input,
    float* output)
{
    const int BS = 512;                    // number of threads in one block
    const int volume = batch * C * H * W;  // size of input tensor
    const int GS = (volume + BS - 1) / BS; // number of blocks to launch, calculated so global number of threads is >= volume

    ReducedDivisor C_out(C / (stride * stride));
    reorgKernel<BS><<<GS, BS, 0, stream>>>(input, volume, ReducedDivisor(batch), ReducedDivisor(C), ReducedDivisor(H), ReducedDivisor(W), C_out, ReducedDivisor(stride), output);
    return STATUS_SUCCESS;
}

pluginStatus_t reorgInference(
    cudaStream_t stream,
    const int batch,
    const int C,
    const int H,
    const int W,
    const int stride,
    const void* input,
    void* output)
{
    return reorgGPU(stream, batch, C, H, W, stride, (const float*) input, (float*) output);
}
