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
#ifndef _FMHA_FLASH_ATTENTION
#define _FMHA_FLASH_ATTENTION

#include "common/bertCommon.h"
#include "common/plugin.h"
#include "commonDatatype.h"
#include "sharedCubinLoader.h"

namespace
{
    static inline size_t get_size_in_bytes(size_t n, nvinfer1::plugin::Data_type dtype)
    {
        switch (dtype)
        {
        case nvinfer1::plugin::DATA_TYPE_E8M10: return n * 4;
        case nvinfer1::plugin::DATA_TYPE_FP32: return n * 4;
        case nvinfer1::plugin::DATA_TYPE_FP16: return n * 2;
        case nvinfer1::plugin::DATA_TYPE_INT32: return n * 4;
        case nvinfer1::plugin::DATA_TYPE_INT8: return n;
        case nvinfer1::plugin::DATA_TYPE_INT4: return n / 2U;
        case nvinfer1::plugin::DATA_TYPE_BOOL: return n / 8U;
        case nvinfer1::plugin::DATA_TYPE_E8M7: return n * 2;
        default: PLUGIN_ASSERT(false); return 0;
        }
    }
}

namespace nvinfer1
{
namespace plugin
{

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_attention_params_v2
{
    // The QKV matrices.
    void* qkv_ptr;
    // The mask to implement drop-out.
    void* packed_mask_ptr;
    // The O matrix (output).
    void* o_ptr;

    // The stride between rows of the Q, K and V matrices.
    int64_t qkv_stride_in_bytes;
    // The stride between matrices of packed mask.
    int64_t packed_mask_stride_in_bytes;
    // The stride between rows of O.
    int64_t o_stride_in_bytes;

#if defined(STORE_P)
    // The pointer to the P matrix (for debugging).
    void* p_ptr;
    // The stride between rows of the P matrix (for debugging).
    int64_t p_stride_in_bytes;
#endif // defined(STORE_P)

#if defined(STORE_S)
    // The pointer to the S matrix (for debugging).
    void* s_ptr;
    // The stride between rows of the S matrix (for debugging).
    int64_t s_stride_in_bytes;
#endif // defined(STORE_S)

    // The dimensions.
    int32_t b, h, s, d;
    // The scaling factors for the kernel.
    uint32_t scale_bmm1, scale_softmax, scale_bmm2;

    // Do we use Niall's trick to avoid I2F/F2I in the INT8 kernel.
    // See https://confluence.nvidia.com/pages/viewpage.action?pageId=302779721 for details.
    bool enable_i2f_trick;

    // array of length b+1 holding prefix sum of actual sequence lenghts
    int32_t* cu_seqlens;

    // use C/32 Format.
    bool interleaved = false;
    bool ignore_b1opt = false;
    bool force_unroll = false;
    bool use_int8_scale_max = false;

    void clear()
    {
        qkv_ptr = nullptr;
        packed_mask_ptr = nullptr;
        o_ptr = nullptr;

        qkv_stride_in_bytes = 0;
        packed_mask_stride_in_bytes = 0;
        o_stride_in_bytes = 0;
#if defined(STORE_P)
        p_ptr = nullptr;
        p_stride_in_bytes = 0
#endif // defined(STORE_P)

#if defined(STORE_S)
            s_ptr
            = nullptr;
        s_stride_in_bytes = 0;
#endif // defined(STORE_S)

        b = 0;
        h = 0;
        s = 0;
        d = 0;
        // The scaling factors for the kernel.
        scale_bmm1 = 0;
        scale_softmax = 0;
        scale_bmm2 = 0;

        enable_i2f_trick = false;

        cu_seqlens = nullptr;
        interleaved = false;
        ignore_b1opt = false;
        force_unroll = false;
        use_int8_scale_max = false;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_160_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_256_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_16_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_32_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_40_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_64_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_80_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_128_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_160_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_256_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_16_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_32_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_40_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_64_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_80_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_128_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_160_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_S_256_sm89_cu_cubin[];

extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_160_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_256_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_16_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_32_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_40_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_64_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_80_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_128_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_160_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_256_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_16_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_32_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_40_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_64_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_80_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_128_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_160_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_S_256_sm89_cu_cubin_len;

constexpr uint32_t S{0};

#if !(defined(ENABLE_SM80) || defined(ENABLE_SM86) || defined(ENABLE_SM89))
#error This file can only be included one of sm 80, 86 or 89 are defined.
#endif
static const struct FusedMultiHeadFlashAttentionKernelMetaInfoV2
{
    Data_type mDataType;
    uint32_t mS;
    uint32_t mD;
    uint32_t mSM;
    unsigned char const* mCubin;
    uint32_t mCubinSize;
    char const* mFuncName;
    uint32_t mSharedMemBytes;
    uint32_t mThreadsPerCTA;
    uint32_t mUnrollStep;
    bool mInterleaved;
} sMhaKernelMetaInfos[] = {
#if defined(ENABLE_SM80)
    { DATA_TYPE_FP16, S, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_16_sm80_kernel", 8192, 128, 0, false },
    { DATA_TYPE_FP16, S, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_16_sm80_kernel_nl", 8192, 128, 64, false },
    { DATA_TYPE_FP16, S, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_32_sm80_kernel", 16384, 128, 0, false },
    { DATA_TYPE_FP16, S, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_32_sm80_kernel_nl", 16384, 128, 64, false },
    { DATA_TYPE_FP16, S, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_40_sm80_kernel", 32768, 128, 0, false },
    { DATA_TYPE_FP16, S, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_40_sm80_kernel_nl", 32768, 128, 64, false },
    { DATA_TYPE_FP16, S, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_64_sm80_kernel", 32768, 128, 0, false },
    { DATA_TYPE_FP16, S, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_64_sm80_kernel_nl", 32768, 128, 64, false },
    { DATA_TYPE_FP16, S, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_80_sm80_kernel", 49152, 128, 0, false },
    { DATA_TYPE_FP16, S, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_80_sm80_kernel_nl", 49152, 128, 64, false },
    { DATA_TYPE_FP16, S, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_128_sm80_kernel", 49152, 128, 0, false },
    { DATA_TYPE_FP16, S, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_128_sm80_kernel_nl", 49152, 128, 64, false },
    { DATA_TYPE_FP16, S, 160, kSM_80,  cubin_fmha_v2_flash_attention_fp16_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_160_sm80_kernel", 98304, 128, 0, false },
    { DATA_TYPE_FP16, S, 160, kSM_80,  cubin_fmha_v2_flash_attention_fp16_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_160_sm80_kernel_nl", 98304, 128, 64, false },
    { DATA_TYPE_FP16, S, 256, kSM_80,  cubin_fmha_v2_flash_attention_fp16_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_256_sm80_kernel", 98304, 128, 0, false },
    { DATA_TYPE_FP16, S, 256, kSM_80,  cubin_fmha_v2_flash_attention_fp16_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_256_sm80_kernel_nl", 98304, 128, 64, false },
#endif // defined(ENABLE_SM80)
#if defined(ENABLE_SM86)
    { DATA_TYPE_FP16, S, 16, kSM_86,  cubin_fmha_v2_flash_attention_fp16_S_16_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_16_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_16_sm86_kernel", 8192, 128, 0, false },
    { DATA_TYPE_FP16, S, 16, kSM_86,  cubin_fmha_v2_flash_attention_fp16_S_16_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_16_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_16_sm86_kernel_nl", 8192, 128, 64, false },
    { DATA_TYPE_FP16, S, 32, kSM_86,  cubin_fmha_v2_flash_attention_fp16_S_32_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_32_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_32_sm86_kernel", 16384, 128, 0, false },
    { DATA_TYPE_FP16, S, 32, kSM_86,  cubin_fmha_v2_flash_attention_fp16_S_32_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_32_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_32_sm86_kernel_nl", 16384, 128, 64, false },
    { DATA_TYPE_FP16, S, 40, kSM_86,  cubin_fmha_v2_flash_attention_fp16_S_40_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_40_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_40_sm86_kernel", 32768, 128, 0, false },
    { DATA_TYPE_FP16, S, 40, kSM_86,  cubin_fmha_v2_flash_attention_fp16_S_40_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_40_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_40_sm86_kernel_nl", 32768, 128, 64, false },
    { DATA_TYPE_FP16, S, 64, kSM_86,  cubin_fmha_v2_flash_attention_fp16_S_64_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_64_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_64_sm86_kernel", 32768, 128, 0, false },
    { DATA_TYPE_FP16, S, 64, kSM_86,  cubin_fmha_v2_flash_attention_fp16_S_64_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_64_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_64_sm86_kernel_nl", 32768, 128, 64, false },
    { DATA_TYPE_FP16, S, 80, kSM_86,  cubin_fmha_v2_flash_attention_fp16_S_80_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_80_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_80_sm86_kernel", 49152, 128, 0, false },
    { DATA_TYPE_FP16, S, 80, kSM_86,  cubin_fmha_v2_flash_attention_fp16_S_80_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_80_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_80_sm86_kernel_nl", 49152, 128, 64, false },
    { DATA_TYPE_FP16, S, 128, kSM_86,  cubin_fmha_v2_flash_attention_fp16_S_128_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_128_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_128_sm86_kernel", 49152, 128, 0, false },
    { DATA_TYPE_FP16, S, 128, kSM_86,  cubin_fmha_v2_flash_attention_fp16_S_128_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_128_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_128_sm86_kernel_nl", 49152, 128, 64, false },
    { DATA_TYPE_FP16, S, 160, kSM_86,  cubin_fmha_v2_flash_attention_fp16_S_160_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_160_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_160_sm86_kernel", 98304, 128, 0, false },
    { DATA_TYPE_FP16, S, 160, kSM_86,  cubin_fmha_v2_flash_attention_fp16_S_160_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_160_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_160_sm86_kernel_nl", 98304, 128, 64, false },
    { DATA_TYPE_FP16, S, 256, kSM_86,  cubin_fmha_v2_flash_attention_fp16_S_256_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_256_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_256_sm86_kernel", 98304, 128, 0, false },
    { DATA_TYPE_FP16, S, 256, kSM_86,  cubin_fmha_v2_flash_attention_fp16_S_256_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_256_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_256_sm86_kernel_nl", 98304, 128, 64, false },
#endif
#if defined(ENABLE_SM89)
    { DATA_TYPE_FP16, S, 16, kSM_89,  cubin_fmha_v2_flash_attention_fp16_S_16_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_16_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_16_sm89_kernel", 8192, 128, 0, false },
    { DATA_TYPE_FP16, S, 16, kSM_89,  cubin_fmha_v2_flash_attention_fp16_S_16_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_16_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_16_sm89_kernel_nl", 8192, 128, 64, false },
    { DATA_TYPE_FP16, S, 32, kSM_89,  cubin_fmha_v2_flash_attention_fp16_S_32_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_32_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_32_sm89_kernel", 16384, 128, 0, false },
    { DATA_TYPE_FP16, S, 32, kSM_89,  cubin_fmha_v2_flash_attention_fp16_S_32_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_32_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_32_sm89_kernel_nl", 16384, 128, 64, false },
    { DATA_TYPE_FP16, S, 40, kSM_89,  cubin_fmha_v2_flash_attention_fp16_S_40_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_40_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_40_sm89_kernel", 32768, 128, 0, false },
    { DATA_TYPE_FP16, S, 40, kSM_89,  cubin_fmha_v2_flash_attention_fp16_S_40_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_40_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_40_sm89_kernel_nl", 32768, 128, 64, false },
    { DATA_TYPE_FP16, S, 64, kSM_89,  cubin_fmha_v2_flash_attention_fp16_S_64_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_64_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_64_sm89_kernel", 32768, 128, 0, false },
    { DATA_TYPE_FP16, S, 64, kSM_89,  cubin_fmha_v2_flash_attention_fp16_S_64_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_64_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_64_sm89_kernel_nl", 32768, 128, 64, false },
    { DATA_TYPE_FP16, S, 80, kSM_89,  cubin_fmha_v2_flash_attention_fp16_S_80_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_80_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_80_sm89_kernel", 49152, 128, 0, false },
    { DATA_TYPE_FP16, S, 80, kSM_89,  cubin_fmha_v2_flash_attention_fp16_S_80_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_80_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_80_sm89_kernel_nl", 49152, 128, 64, false },
    { DATA_TYPE_FP16, S, 128, kSM_89,  cubin_fmha_v2_flash_attention_fp16_S_128_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_128_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_128_sm89_kernel", 49152, 128, 0, false },
    { DATA_TYPE_FP16, S, 128, kSM_89,  cubin_fmha_v2_flash_attention_fp16_S_128_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_128_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_128_sm89_kernel_nl", 49152, 128, 64, false },
    { DATA_TYPE_FP16, S, 160, kSM_89,  cubin_fmha_v2_flash_attention_fp16_S_160_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_160_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_160_sm89_kernel", 98304, 128, 0, false },
    { DATA_TYPE_FP16, S, 160, kSM_89,  cubin_fmha_v2_flash_attention_fp16_S_160_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_160_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_160_sm89_kernel_nl", 98304, 128, 64, false },
    { DATA_TYPE_FP16, S, 256, kSM_89,  cubin_fmha_v2_flash_attention_fp16_S_256_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_256_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_256_sm89_kernel", 98304, 128, 0, false },
    { DATA_TYPE_FP16, S, 256, kSM_89,  cubin_fmha_v2_flash_attention_fp16_S_256_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_S_256_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_S_256_sm89_kernel_nl", 98304, 128, 64, false }

#endif // defined(ENABLE_SM89)
};

class FusedMultiHeadFlashAttentionKernel
    : public TSharedCubinKernel<FusedMultiHeadFlashAttentionKernelMetaInfoV2, Fused_multihead_attention_params_v2>
{
public:
    FusedMultiHeadFlashAttentionKernel(FusedMultiHeadFlashAttentionKernelMetaInfoV2 const* pMetaStart,
        uint32_t nMetaCount, Data_type type, uint32_t sm)
        : TSharedCubinKernel<FusedMultiHeadFlashAttentionKernelMetaInfoV2, Fused_multihead_attention_params_v2>(
            pMetaStart, nMetaCount, type, sm)
    {
    }

    uint64_t hashID(uint32_t headsize, bool interleaved, bool unroll) const
    {
        // we only have 30 bits room for head size
        PLUGIN_ASSERT(headsize <= 0x3FFFFFFF);
        return (headsize << 2) | (interleaved ? 2U : 0U) | (unroll ? 1U : 0U);
    }

    uint64_t hashID(Fused_multihead_attention_params_v2 const& param) const
    {
        return hashID(param.d, param.interleaved, param.force_unroll);
    }

    uint64_t hashID(KernelMeta const& kernelMeta) const
    {
        return hashID(kernelMeta.mD, kernelMeta.mInterleaved, kernelMeta.mUnrollStep > 0);
    }
};

using FusedMHAFlashKernelFactory = TSharedCubinKernelFactory<FusedMultiHeadFlashAttentionKernel>;

inline FusedMultiHeadFlashAttentionKernel const* getFMHACubinKernels(Data_type type, uint32_t sm)
{
    return FusedMHAFlashKernelFactory::Get().getCubinKernels(
        sMhaKernelMetaInfos, sizeof(sMhaKernelMetaInfos) / sizeof(sMhaKernelMetaInfos[0]), type, sm);
}

} // namespace plugin
} // namespace nvinfer1

#endif
