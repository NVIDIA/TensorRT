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
 * Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
 **************************************************************************************************
*/

#ifndef TRT_MULTISCALE_DEFORMABLE_ATTN_H
#define TRT_MULTISCALE_DEFORMABLE_ATTN_H

int32_t ms_deform_attn_cuda_forward(cudaStream_t stream, float const* value, int32_t const* spatialShapes,
    int32_t const* levelStartIndex, float const* samplingLoc, float const* attnWeight, float* output, int32_t batch,
    int32_t mSpatialSize, int32_t mNumHeads, int32_t mChannels, int32_t mNumLevels, int32_t mNumQuery, int32_t mNumPoint);


int32_t ms_deform_attn_cuda_forward(cudaStream_t stream, const __half* value, int32_t const* spatialShapes,
    int32_t const* levelStartIndex, const __half* samplingLoc, const __half* attnWeight, __half* output, int32_t batch,
    int32_t mSpatialSize, int32_t mNumHeads, int32_t mChannels, int32_t mNumLevels, int32_t mNumQuery, int32_t mNumPoint);

#endif
