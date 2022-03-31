/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * Modified from Deformable DETR
 * Copyright (c) 2020 SenseTime. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 [see LICENSE for details]
 * https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE
 **************************************************************************
 * Modified from DCN (https://github.com/msracver/Deformable-ConvNets)
 * Copyright (c) 2018 Microsoft
 **************************************************************************
*/

#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "multiscaleDeformableIm2ColCuda.cuh"

int ms_deform_attn_cuda_forward(cudaStream_t stream, const float* value, const int32_t* spatialShapes,
    const int32_t* levelStartIndex, const float* samplingLoc, const float* attnWeight, float* output, int batch,
    int mSpatialSize, int mNumHeads, int mChannels, int mNumLevels, int mNumQuery, int mNumPoint)
{
    auto perValueSize = mSpatialSize * mNumHeads * mChannels;
    auto perSampleLocSize = mNumQuery * mNumHeads * mNumLevels * mNumPoint * 2;
    auto perAttnWeightSize = mNumQuery * mNumHeads * mNumLevels * mNumPoint;

    int mIm2colStep = batch;

    for (int n = 0; n < batch / mIm2colStep; ++n)
    {
        auto columns = output + perValueSize * n * mIm2colStep;
        ms_deformable_im2col_cuda<float>(stream, value + n * mIm2colStep * perValueSize, spatialShapes, levelStartIndex,
            samplingLoc + n * mIm2colStep * perSampleLocSize, attnWeight + n * mIm2colStep * perAttnWeightSize, batch,
            mSpatialSize, mNumHeads, mChannels, mNumLevels, mNumQuery, mNumPoint, columns);
    }
    
    return 0;
}

#if __CUDA_ARCH__ >= 530

int ms_deform_attn_cuda_forward(cudaStream_t stream, const __half* value, const int32_t* spatialShapes,
    const int32_t* levelStartIndex, const __half* samplingLoc, const __half* attnWeight, __half* output, int batch,
    int mSpatialSize, int mNumHeads, int mChannels, int mNumLevels, int mNumQuery, int mNumPoint)
{
    auto perValueSize = mSpatialSize * mNumHeads * mChannels;
    auto perSampleLocSize = mNumQuery * mNumHeads * mNumLevels * mNumPoint * 2;
    auto perAttnWeightSize = mNumQuery * mNumHeads * mNumLevels * mNumPoint;

    int mIm2colStep = batch;
    for (int n = 0; n < batch / mIm2colStep; ++n)
    {
        auto columns = output + perValueSize * n * mIm2colStep;
        ms_deformable_im2col_cuda<__half>(stream, value + n * mIm2colStep * perValueSize, spatialShapes,
            levelStartIndex, samplingLoc + n * mIm2colStep * perSampleLocSize,
            attnWeight + n * mIm2colStep * perAttnWeightSize, batch, mSpatialSize, mNumHeads, mChannels, mNumLevels,
            mNumQuery, mNumPoint, columns);
    }

    return 0;
}

#endif
