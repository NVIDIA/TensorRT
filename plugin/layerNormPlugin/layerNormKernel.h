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
#ifndef TRT_LAYERNORM_KERNEL_H
#define TRT_LAYERNORM_KERNEL_H

#include "common/checkMacrosPlugin.h"

#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

using half = __half;

template <typename T>
int32_t computeLayerNorm(int32_t const gridSize, int32_t const nHiddenDimension, T const* input, T const* gamma, T const* beta,
    T* output, float const epsilon, cudaStream_t stream);

int32_t computeLayerNormQDQ(int32_t const gridSize, int32_t const nHiddenDimension, int8_t const* input, __half const* gamma,
    __half const* beta, int8_t* output, float const dqScaleIn, float const qScale, float const epsilon,
    cudaStream_t stream);

#endif // TRT_LAYERNORM_KERNEL_H
