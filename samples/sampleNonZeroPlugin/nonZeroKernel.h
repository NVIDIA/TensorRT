/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef SAMPLE_NONZERO_KERNEL_H
#define SAMPLE_NONZERO_KERNEL_H

#include <cuda_fp16.h>

#include <cstdint>

template <typename T>
void nonZeroIndicesImpl(T const* X, int32_t* indices, int64_t* count, int64_t const* K, int32_t R, int32_t C,
    bool rowOrder, cudaStream_t stream);

#endif // SAMPLE_NONZERO_KERNEL_H
