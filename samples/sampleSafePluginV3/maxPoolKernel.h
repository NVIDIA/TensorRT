/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef MAX_POOL_KERNEL_H
#define MAX_POOL_KERNEL_H

#include <cstdint>
#include <cuda_fp16.h>

int32_t maxPoolFloat(cudaStream_t stream, int32_t batch_size, int32_t C, int32_t H, int32_t W, void const* input,
    void* output, int32_t kernsize, int32_t stride, int32_t pad);
int32_t maxPoolHalf(cudaStream_t stream, int32_t batch_size, int32_t C, int32_t H, int32_t W, void const* input,
    void* output, int32_t kernsize, int32_t stride, int32_t pad);
int32_t maxPoolInt8(cudaStream_t stream, int32_t batch_size, int32_t C, int32_t H, int32_t W, void const* input,
    void* output, int32_t kernsize, int32_t stride, int32_t pad);

#endif // MAX_POOL_KERNEL_H
