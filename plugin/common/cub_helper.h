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

// TODO: Remove WAR once issue resolved in CUB (CUDA 12.6+?)

#ifndef CUDA_VERSION
#include <cuda.h>
#endif // CUDA_VERSION

#include "common/kernels/kernel.h"
#include <cub/cub.cuh>
template <typename KeyT, typename ValueT>
size_t cubSortPairsWorkspaceSize(int32_t num_items, int32_t num_segments)
{
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending((void*) NULL, temp_storage_bytes, (KeyT const*) NULL,
        (KeyT*) NULL, (ValueT const*) NULL, (ValueT*) NULL,
        num_items,    // # items
        num_segments, // # segments
        (int32_t const*) NULL, (int32_t const*) NULL);
    return temp_storage_bytes;
}
