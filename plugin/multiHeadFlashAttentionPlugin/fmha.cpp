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

#if defined(ENABLE_SM75) || defined(ENABLE_SM80) || defined(ENABLE_SM86) || defined(ENABLE_SM89)
#include "fmha.h"

namespace nvinfer1
{
namespace plugin
{

int32_t runFMHFAKernel(void const* devQKV, void* cuSeqlens, void* devOutput, size_t total, int32_t sm,
    FusedMultiHeadFlashAttentionKernel const* kernels, int32_t b, int32_t h, int32_t d, int32_t s, cudaStream_t stream)
{
    Fused_multihead_flash_attention_params_v2 params
        = getMHFAParams(/* data_type */ DATA_TYPE_FP16, /* acc_type */ DATA_TYPE_FP16, b, s, h, d, total, devQKV,
            cuSeqlens, devOutput, /* p_d */ nullptr, /* s_d */ nullptr,
            /* scale_bmm1 */ 1.F / sqrtf(d), /* scale_softmax */ 1.F, /* scale_bmm2 */ 1.F,
            /* interleaved */ false,
            /* ignore_b1opt */ false,
            /* force_unroll */ true,
            /* use_int8_scale_max  */ false);

    kernels->run(params, stream);
    return 0;
}
} // namespace plugin
} // namespace nvinfer1
#endif

