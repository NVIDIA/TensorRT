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
#ifndef TRT_ROIALIGN_KERNEL_H
#define TRT_ROIALIGN_KERNEL_H

#include "common/checkMacrosPlugin.h"
#include <cuda_runtime.h>
#include <stdint.h>

template <typename T>
cudaError_t RoiAlignImpl(cudaStream_t stream, int32_t const maxThreadsPerBlock, T const* bottomData, T const spatialScale,
    int32_t const numRois, int32_t const channels, int32_t const height, int32_t const width, int32_t const pooledHeight,
    int32_t const pooledWidth, int32_t const samplingRatio, T const* bottomRois, T* topData, int32_t const isModeAvg,
    int32_t const* batchIndicesPtr, int32_t const aligned);

#endif // TRT_ROIALIGN_KERNEL_H
