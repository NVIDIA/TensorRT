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

#ifndef TRT_COMMON_CUDA_HELPER_H
#define TRT_COMMON_CUDA_HELPER_H

#include <cstdint>
#include <cuda.h>

constexpr int32_t THREADS_PER_BLOCK{512};

inline int32_t get_blocks(int32_t const N, int32_t const numThreads = THREADS_PER_BLOCK)
{
    int32_t optimalBlockNum = (N + numThreads - 1) / numThreads;
    int32_t maxBlockNum = 4096;

    return min(optimalBlockNum, maxBlockNum);
}

#endif // TRT_COMMON_CUDA_HELPER_H
