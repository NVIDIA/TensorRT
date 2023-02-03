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

int32_t ms_deform_attn_cuda_forward(cudaStream_t stream, const float* value, const int32_t* spatialShapes,
    const int32_t* levelStartIndex, const float* samplingLoc, const float* attnWeight, float* output, int32_t batch,
    int32_t mSpatialSize, int32_t mNumHeads, int32_t mChannels, int32_t mNumLevels, int32_t mNumQuery, int32_t mNumPoint)
{
    auto perValueSize = mSpatialSize * mNumHeads * mChannels;
    auto perSampleLocSize = mNumQuery * mNumHeads * mNumLevels * mNumPoint * 2;
    auto perAttnWeightSize = mNumQuery * mNumHeads * mNumLevels * mNumPoint;

    int32_t mIm2colStep = batch;

    for (int32_t n = 0; n < batch / mIm2colStep; ++n)
    {
        auto columns = output + perValueSize * n * mIm2colStep;
        ms_deformable_im2col_cuda<float>(stream, value + n * mIm2colStep * perValueSize, spatialShapes, levelStartIndex,
            samplingLoc + n * mIm2colStep * perSampleLocSize, attnWeight + n * mIm2colStep * perAttnWeightSize, batch,
            mSpatialSize, mNumHeads, mChannels, mNumLevels, mNumQuery, mNumPoint, columns);
    }
    
    return 0;
}

int32_t ms_deform_attn_cuda_forward(cudaStream_t stream, const __half* value, const int32_t* spatialShapes,
    const int32_t* levelStartIndex, const __half* samplingLoc, const __half* attnWeight, __half* output, int32_t batch,
    int32_t mSpatialSize, int32_t mNumHeads, int32_t mChannels, int32_t mNumLevels, int32_t mNumQuery, int32_t mNumPoint)
{
    auto perValueSize = mSpatialSize * mNumHeads * mChannels;
    auto perSampleLocSize = mNumQuery * mNumHeads * mNumLevels * mNumPoint * 2;
    auto perAttnWeightSize = mNumQuery * mNumHeads * mNumLevels * mNumPoint;

    int32_t mIm2colStep = batch;
    for (int32_t n = 0; n < batch / mIm2colStep; ++n)
    {
        auto columns = output + perValueSize * n * mIm2colStep;
        ms_deformable_im2col_cuda<__half>(stream, value + n * mIm2colStep * perValueSize, spatialShapes,
            levelStartIndex, samplingLoc + n * mIm2colStep * perSampleLocSize,
            attnWeight + n * mIm2colStep * perAttnWeightSize, batch, mSpatialSize, mNumHeads, mChannels, mNumLevels,
            mNumQuery, mNumPoint, columns);
    }

    return 0;
}
