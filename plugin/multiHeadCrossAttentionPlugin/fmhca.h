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

#pragma once
#ifndef _FMHCA_
#define _FMHCA_

#include "fmha_cross_attention/include/fmha_cross_attention.h"
#include <stdio.h>
#include <stdlib.h>

namespace nvinfer1
{
namespace plugin
{
int32_t run_fmhca_api(void* q_packed_d, void* kv_packed_d, void* cu_seqlens_q_d, void* cu_seqlens_kv_d,
    void* o_packed_d, int32_t sm, FusedMultiHeadCrossAttentionKernel const* kernels, size_t b = 2, size_t h = 8,
    size_t d = 64, size_t s_q = 4096, size_t s_kv = 77, cudaStream_t stream = 0);
}
} // namespace nvinfer1

#endif
