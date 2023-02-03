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

#ifndef TRT_FMHA_H
#define TRT_FMHA_H

#include "fmha_flash_attention/include/fmha_flash_attention.h"

namespace nvinfer1
{
namespace plugin
{
int32_t runFMHFAKernel(void const* devQKV, void* cuSeqlens, void* devOutput, size_t total, int32_t sm,
    FusedMultiHeadFlashAttentionKernel const* kernels, int32_t b = 2, int32_t h = 8, int32_t d = 64, int32_t s = 4096,
    cudaStream_t stream = 0);
}
} // namespace nvinfer1

#endif // TRT_FMHA_H
