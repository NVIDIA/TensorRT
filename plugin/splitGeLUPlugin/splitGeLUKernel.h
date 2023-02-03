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
#ifndef TRT_SPLITGELU_KERNEL_H
#define TRT_SPLITGELU_KERNEL_H

#include "common/checkMacrosPlugin.h"

#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

using half = __half;

template <typename T>
int32_t launchSplitGeLUKernel(cudaStream_t stream, int32_t gridSize, int32_t nHalfHiddenSize, T const* input, T* output,
    float const fDivRecip, float const fAdd, float const fMul);

#endif // TRT_SPLITGELU_KERNEL_H
