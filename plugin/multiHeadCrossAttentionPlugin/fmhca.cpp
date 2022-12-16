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
#include "fmhca.h"

namespace nvinfer1
{
namespace plugin
{
int32_t runFMHCAKernel(void const* devQ, void const* devKV, void* cuSeqlensQ, void* cuSeqlensKV, void* devOutput,
    int32_t sm, FusedMultiHeadCrossAttentionKernel const* kernels, int32_t b, int32_t h, int32_t d, int32_t seqQ,
    int32_t seqKV, cudaStream_t stream)
{

    PLUGIN_VALIDATE(sm != kSM_75 || d < 160, "There are no fMHCA kernels for sm75 and d >= 160.");

    // Run kernel.
    Fused_multihead_attention_params_mhca params = getMHCAParams(/* dType */ DATA_TYPE_FP16,
        /* accType */ DATA_TYPE_FP16, b, seqQ, seqKV, h, d, /* total */ 0, devQ, devKV, cuSeqlensQ, cuSeqlensKV,
        devOutput, /* devP */ nullptr, /* devS */ nullptr, /* scaleBmm1 */ 1.F / sqrtf(d), /* scaleSoftmax */ 1.F,
        /* scaleBmm2 */ 1.F, /* interleaved */ false, /* ignoreB1Opt */ false,
        /* forceUnroll */ true, /* useInt8ScaleMax */ false, /* useTMA */ false);

    kernels->run(params, stream);
    return 0;
}
} // namespace plugin
} // namespace nvinfer1
#endif

