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
#ifndef TRT_GROUPNORM_PLUGIN_COMMON_H
#define TRT_GROUPNORM_PLUGIN_COMMON_H

#include "cuda_fp16.h"
#include <cuda.h>
#include <stdint.h>

struct Group_norm_nhwc_params
{
    // The output buffer. Layout NHWC.
    __half* dst;
    // The input buffer. Layout NHWC.
    __half const* src;
    // The gamma scaling factor.
    float const* gamma;
    // The beta term to add in GN.
    float const* beta;
    // The temporary buffer to do the global parallel reduction. Size:
    // BLOCKS_PER_BATCH x C x 2.
    float* red_buffer;

    // The number of instances in the batch.
    int32_t n;
    // The height and width of each activation map.
    int32_t h, w;
    // The number of channels.
    int32_t c;
    // The number of groups.
    int32_t groups;
    // Do we apply the Swish activation function?
    bool with_swish;

    // Precomputed values and parameters to control the execution of the kernels.

    // The number of activations per instance (h * w) and the number of
    // activations per block.
    int32_t hw, hw_per_block;
    // The number of channels per group and blocks per activation in the C
    // dimension.
    int32_t c_per_block, c_per_group;

    // The precomputed stride between instances.
    int32_t hwc;
    // The inverse of hwc in floats (to compute mean/var).
    float inv_hwc;
    // The precomputed number of groups per block.
    int32_t groups_per_block;
};
#endif // TRT_GROUPNORM_PLUGIN_COMMON_H
