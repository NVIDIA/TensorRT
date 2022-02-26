/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "fused_multihead_attention.h"
#include "fused_multihead_attention_common.h"
#include <cassert>
#include <cstdint>

namespace bert
{
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
    int b, h, s, d;
    // The scaling factors for the kernel.
    uint32_t scale_bmm1, scale_softmax, scale_bmm2;

    // Do we use Niall's trick to avoid I2F/F2I in the INT8 kernel.
    // See https://confluence.nvidia.com/pages/viewpage.action?pageId=302779721 for details.
    bool enable_i2f_trick;

    // array of length b+1 holding prefix sum of actual sequence lenghts
    int* cu_seqlens;

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
extern unsigned char fused_multihead_attention_v2_fp16_128_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_128_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_128_64_kernel_sm86_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_256_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_256_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_256_64_kernel_sm86_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_384_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_384_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_384_64_kernel_sm86_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_64_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_64_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_64_64_kernel_sm86_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_96_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_96_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_fp16_96_64_kernel_sm86_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_128_64_kernel_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_128_64_kernel_sm86_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_192_64_kernel_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_192_64_kernel_sm86_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_256_64_kernel_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_256_64_kernel_sm86_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_384_64_kernel_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin[];
extern unsigned char fused_multihead_attention_v2_int8_384_64_kernel_sm86_cubin[];

extern unsigned char cubin_fmha_v2_int8_512_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_512_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_256_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_128_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_512_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_512_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_256_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_128_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_512_64_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_512_32_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_256_32_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_128_32_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_512_64_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_512_32_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_256_32_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_128_32_sm75_cu_cubin[];

extern uint32_t fused_multihead_attention_v2_fp16_128_64_kernel_sm75_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_128_64_kernel_sm80_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_128_64_kernel_sm86_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_256_64_kernel_sm75_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_256_64_kernel_sm80_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_256_64_kernel_sm86_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_384_64_kernel_sm75_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_384_64_kernel_sm80_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_384_64_kernel_sm86_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_64_64_kernel_sm75_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_64_64_kernel_sm80_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_64_64_kernel_sm86_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_96_64_kernel_sm75_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_96_64_kernel_sm80_cubin_len;
extern uint32_t fused_multihead_attention_v2_fp16_96_64_kernel_sm86_cubin_len;
extern uint32_t fused_multihead_attention_v2_int8_128_64_kernel_cubin_len;
extern uint32_t fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin_len;
extern uint32_t fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin_len;
extern uint32_t fused_multihead_attention_v2_int8_128_64_kernel_sm86_cubin_len;
extern uint32_t fused_multihead_attention_v2_int8_192_64_kernel_cubin_len;
extern uint32_t fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin_len;
extern uint32_t fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin_len;
extern uint32_t fused_multihead_attention_v2_int8_192_64_kernel_sm86_cubin_len;
extern uint32_t fused_multihead_attention_v2_int8_256_64_kernel_cubin_len;
extern uint32_t fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin_len;
extern uint32_t fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin_len;
extern uint32_t fused_multihead_attention_v2_int8_256_64_kernel_sm86_cubin_len;
extern uint32_t fused_multihead_attention_v2_int8_384_64_kernel_cubin_len;
extern uint32_t fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin_len;
extern uint32_t fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin_len;
extern uint32_t fused_multihead_attention_v2_int8_384_64_kernel_sm86_cubin_len;

extern uint32_t cubin_fmha_v2_int8_512_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_512_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_256_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_128_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_512_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_512_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_256_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_128_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_512_64_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_512_32_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_256_32_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_128_32_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_512_64_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_512_32_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_256_32_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_128_32_sm75_cu_cubin_len;

static const struct FusedMultiHeadAttentionKernelMetaInfoV2
{
    Data_type mDataType;
    uint32_t mS;
    uint32_t mD;
    uint32_t mSM;
    const unsigned char* mCubin;
    uint32_t mCubinSize;
    const char* mFuncName;
    uint32_t mSharedMemBytes;
    uint32_t mThreadsPerCTA;
    uint32_t mUnrollStep;
    bool mInterleaved;
} sMhaKernelMetaInfosV2[] = {
    // Xavier
    {DATA_TYPE_INT8, 128, 64, kSM_72, fused_multihead_attention_v2_int8_128_64_kernel_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm72_interleaved", 24576, 128, 0, true},
    {DATA_TYPE_INT8, 128, 64, kSM_72, fused_multihead_attention_v2_int8_128_64_kernel_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm72", 32768, 128, 0, false},
    {DATA_TYPE_INT8, 192, 64, kSM_72, fused_multihead_attention_v2_int8_192_64_kernel_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm72_interleaved", 28672, 128, 0, true},
    {DATA_TYPE_INT8, 192, 64, kSM_72, fused_multihead_attention_v2_int8_192_64_kernel_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm72", 45056, 128, 0, false},
    {DATA_TYPE_INT8, 256, 64, kSM_72, fused_multihead_attention_v2_int8_256_64_kernel_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm72_interleaved", 36864, 128, 0, true},
    {DATA_TYPE_INT8, 256, 64, kSM_72, fused_multihead_attention_v2_int8_256_64_kernel_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm72", 57344, 128, 0, false},
    {DATA_TYPE_INT8, 384, 64, kSM_72, fused_multihead_attention_v2_int8_384_64_kernel_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm72_interleaved", 51200, 128, 0, true},
    {DATA_TYPE_INT8, 384, 64, kSM_72, fused_multihead_attention_v2_int8_384_64_kernel_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm72", 77824, 128, 0, false},

    // Turing
    {DATA_TYPE_FP16, 64, 64, kSM_75, fused_multihead_attention_v2_fp16_64_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_fp16_64_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_fp16_64_64_kernel_sm75", 24576, 128, 0, false},
    {DATA_TYPE_FP16, 96, 64, kSM_75, fused_multihead_attention_v2_fp16_96_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_fp16_96_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_fp16_96_64_kernel_sm75", 24576, 128, 0, false},
    {DATA_TYPE_FP16, 128, 64, kSM_75, fused_multihead_attention_v2_fp16_128_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_fp16_128_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_fp16_128_64_kernel_sm75_noloop", 20480, 128, 32, false},
    {DATA_TYPE_FP16, 128, 64, kSM_75, fused_multihead_attention_v2_fp16_128_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_fp16_128_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_fp16_128_64_kernel_sm75", 32768, 128, 0, false},
    {DATA_TYPE_FP16, 256, 64, kSM_75, fused_multihead_attention_v2_fp16_256_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_fp16_256_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_fp16_256_64_kernel_sm75_noloop", 36864, 128, 32, false},
    {DATA_TYPE_FP16, 256, 64, kSM_75, fused_multihead_attention_v2_fp16_256_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_fp16_256_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_fp16_256_64_kernel_sm75", 36864, 128, 0, false},
    {DATA_TYPE_FP16, 384, 64, kSM_75, fused_multihead_attention_v2_fp16_384_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_fp16_384_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_fp16_384_64_kernel_sm75_noloop", 53248, 256, 32, false},
    {DATA_TYPE_FP16, 384, 64, kSM_75, fused_multihead_attention_v2_fp16_384_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_fp16_384_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_fp16_384_64_kernel_sm75", 53248, 256, 0, false},

    {DATA_TYPE_INT8, 128, 64, kSM_75, fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm75_interleaved_noloop", 18432, 128, 16, true},
    {DATA_TYPE_INT8, 128, 64, kSM_75, fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm75_noloop", 18432, 128, 16, false},
    {DATA_TYPE_INT8, 128, 64, kSM_75, fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm75_interleaved", 24576, 128, 0, true},
    {DATA_TYPE_INT8, 128, 64, kSM_75, fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm75", 24576, 128, 0, false},
    {DATA_TYPE_INT8, 192, 64, kSM_75, fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm75_interleaved_noloop", 28672, 128, 64, true},
    {DATA_TYPE_INT8, 192, 64, kSM_75, fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm75_noloop", 28672, 128, 64, false},
    {DATA_TYPE_INT8, 192, 64, kSM_75, fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm75_interleaved", 28672, 128, 0, true},
    {DATA_TYPE_INT8, 192, 64, kSM_75, fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm75", 28672, 128, 0, false},
    {DATA_TYPE_INT8, 256, 64, kSM_75, fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm75_interleaved_noloop", 34816, 128, 32, true},
    {DATA_TYPE_INT8, 256, 64, kSM_75, fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm75_noloop", 34816, 128, 32, false},
    {DATA_TYPE_INT8, 256, 64, kSM_75, fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm75_interleaved", 34816, 128, 0, true},
    {DATA_TYPE_INT8, 256, 64, kSM_75, fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm75", 34816, 128, 0, false},
    {DATA_TYPE_INT8, 384, 64, kSM_75, fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm75_interleaved_noloop", 51200, 128, 32, true},
    {DATA_TYPE_INT8, 384, 64, kSM_75, fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm75_noloop", 51200, 128, 32, false},
    {DATA_TYPE_INT8, 384, 64, kSM_75, fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm75_interleaved", 51200, 128, 0, true},
    {DATA_TYPE_INT8, 384, 64, kSM_75, fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm75_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm75", 51200, 128, 0, false},

    // {fp16} x {512} x {64, 32} x {sm75} x {normal, noloop}
    {DATA_TYPE_FP16, 512, 64, kSM_75, cubin_fmha_v2_fp16_512_64_sm75_cu_cubin,
        cubin_fmha_v2_fp16_512_64_sm75_cu_cubin_len, "fmha_v2_fp16_512_64_sm75_kernel", 69632, 256, 0, false},
    {DATA_TYPE_FP16, 512, 64, kSM_75, cubin_fmha_v2_fp16_512_64_sm75_cu_cubin,
        cubin_fmha_v2_fp16_512_64_sm75_cu_cubin_len, "fmha_v2_fp16_512_64_sm75_kernel_nl", 69632, 256, 32, false},
    {DATA_TYPE_FP16, 512, 32, kSM_75, cubin_fmha_v2_fp16_512_32_sm75_cu_cubin,
        cubin_fmha_v2_fp16_512_32_sm75_cu_cubin_len, "fmha_v2_fp16_512_32_sm75_kernel", 36864, 256, 0, false},
    {DATA_TYPE_FP16, 512, 32, kSM_75, cubin_fmha_v2_fp16_512_32_sm75_cu_cubin,
        cubin_fmha_v2_fp16_512_32_sm75_cu_cubin_len, "fmha_v2_fp16_512_32_sm75_kernel_nl", 36864, 256, 32, false},

    // {fp16, int8} x {128} x {32} x {sm75} x {normal, noloop}
    {DATA_TYPE_INT8, 128, 32, kSM_75, cubin_fmha_v2_int8_128_32_sm75_cu_cubin,
        cubin_fmha_v2_int8_128_32_sm75_cu_cubin_len, "fmha_v2_int8_128_32_sm75_kernel", 12288, 128, 0, false},
    {DATA_TYPE_INT8, 128, 32, kSM_75, cubin_fmha_v2_int8_128_32_sm75_cu_cubin,
        cubin_fmha_v2_int8_128_32_sm75_cu_cubin_len, "fmha_v2_int8_128_32_sm75_kernel_nl", 10240, 128, 32, false},
    {DATA_TYPE_FP16, 128, 32, kSM_75, cubin_fmha_v2_fp16_128_32_sm75_cu_cubin,
        cubin_fmha_v2_fp16_128_32_sm75_cu_cubin_len, "fmha_v2_fp16_128_32_sm75_kernel", 16384, 128, 0, false},
    {DATA_TYPE_FP16, 128, 32, kSM_75, cubin_fmha_v2_fp16_128_32_sm75_cu_cubin,
        cubin_fmha_v2_fp16_128_32_sm75_cu_cubin_len, "fmha_v2_fp16_128_32_sm75_kernel_nl", 10240, 128, 32, false},

    // {fp16, int8} x {256} x {32} x {sm75} x {normal, noloop}
    {DATA_TYPE_INT8, 256, 32, kSM_75, cubin_fmha_v2_int8_256_32_sm75_cu_cubin,
        cubin_fmha_v2_int8_256_32_sm75_cu_cubin_len, "fmha_v2_int8_256_32_sm75_kernel", 18432, 128, 0, false},
    {DATA_TYPE_INT8, 256, 32, kSM_75, cubin_fmha_v2_int8_256_32_sm75_cu_cubin,
        cubin_fmha_v2_int8_256_32_sm75_cu_cubin_len, "fmha_v2_int8_256_32_sm75_kernel_nl", 18432, 128, 32, false},
    {DATA_TYPE_FP16, 256, 32, kSM_75, cubin_fmha_v2_fp16_256_32_sm75_cu_cubin,
        cubin_fmha_v2_fp16_256_32_sm75_cu_cubin_len, "fmha_v2_fp16_256_32_sm75_kernel", 18432, 128, 0, false},
    {DATA_TYPE_FP16, 256, 32, kSM_75, cubin_fmha_v2_fp16_256_32_sm75_cu_cubin,
        cubin_fmha_v2_fp16_256_32_sm75_cu_cubin_len, "fmha_v2_fp16_256_32_sm75_kernel_nl", 18432, 128, 32, false},

    // {int8} x {512} x {64, 32} x {sm75} x {normal, noloop}
    {DATA_TYPE_INT8, 512, 64, kSM_75, cubin_fmha_v2_int8_512_64_sm75_cu_cubin,
        cubin_fmha_v2_int8_512_64_sm75_cu_cubin_len, "fmha_v2_int8_512_64_sm75_kernel", 69632, 256, 0, false},
    {DATA_TYPE_INT8, 512, 64, kSM_75, cubin_fmha_v2_int8_512_64_sm75_cu_cubin,
        cubin_fmha_v2_int8_512_64_sm75_cu_cubin_len, "fmha_v2_int8_512_64_sm75_kernel_nl", 69632, 256, 32, false},
    {DATA_TYPE_INT8, 512, 32, kSM_75, cubin_fmha_v2_int8_512_32_sm75_cu_cubin,
        cubin_fmha_v2_int8_512_32_sm75_cu_cubin_len, "fmha_v2_int8_512_32_sm75_kernel", 36864, 256, 0, false},
    {DATA_TYPE_INT8, 512, 32, kSM_75, cubin_fmha_v2_int8_512_32_sm75_cu_cubin,
        cubin_fmha_v2_int8_512_32_sm75_cu_cubin_len, "fmha_v2_int8_512_32_sm75_kernel_nl", 36864, 256, 32, false},

#if CUDA_VERSION >= 11000
    // {fp16} x {128} x {32} x {sm80} x {normal, noloop}
    {DATA_TYPE_FP16, 128, 32, kSM_80, cubin_fmha_v2_fp16_128_32_sm80_cu_cubin,
        cubin_fmha_v2_fp16_128_32_sm80_cu_cubin_len, "fmha_v2_fp16_128_32_sm80_kernel", 32768, 128, 0, false},
    {DATA_TYPE_FP16, 128, 32, kSM_80, cubin_fmha_v2_fp16_128_32_sm80_cu_cubin,
        cubin_fmha_v2_fp16_128_32_sm80_cu_cubin_len, "fmha_v2_fp16_128_32_sm80_kernel_nl", 20480, 128, 16, false},

    // {int8} x {128} x {32} x {sm80} x {normal, noloop, interleave, interleave_noloop}
    {DATA_TYPE_INT8, 128, 32, kSM_80, cubin_fmha_v2_int8_128_32_sm80_cu_cubin,
        cubin_fmha_v2_int8_128_32_sm80_cu_cubin_len, "fmha_v2_int8_128_32_sm80_kernel", 16384, 128, 0, false},
    {DATA_TYPE_INT8, 128, 32, kSM_80, cubin_fmha_v2_int8_128_32_sm80_cu_cubin,
        cubin_fmha_v2_int8_128_32_sm80_cu_cubin_len, "fmha_v2_int8_128_32_sm80_kernel_nl", 12288, 128, 16, false},
    {DATA_TYPE_INT8, 128, 32, kSM_80, cubin_fmha_v2_int8_128_32_sm80_cu_cubin,
        cubin_fmha_v2_int8_128_32_sm80_cu_cubin_len, "fmha_v2_int8_128_32_sm80_kernel", 12288, 128, 0, true},
    {DATA_TYPE_INT8, 128, 32, kSM_80, cubin_fmha_v2_int8_128_32_sm80_cu_cubin,
        cubin_fmha_v2_int8_128_32_sm80_cu_cubin_len, "fmha_v2_il_int8_128_32_sm80_kernel_nl", 10240, 128, 16, true},

    // {fp16, int8} x {256} x {32} x {sm80} x {normal, noloop}
    {DATA_TYPE_INT8, 256, 32, kSM_80, cubin_fmha_v2_int8_256_32_sm80_cu_cubin,
        cubin_fmha_v2_int8_256_32_sm80_cu_cubin_len, "fmha_v2_int8_256_32_sm80_kernel", 20480, 128, 0, false},
    {DATA_TYPE_INT8, 256, 32, kSM_80, cubin_fmha_v2_int8_256_32_sm80_cu_cubin,
        cubin_fmha_v2_int8_256_32_sm80_cu_cubin_len, "fmha_v2_int8_256_32_sm80_kernel_nl", 20480, 128, 32, false},
    {DATA_TYPE_FP16, 256, 32, kSM_80, cubin_fmha_v2_fp16_256_32_sm80_cu_cubin,
        cubin_fmha_v2_fp16_256_32_sm80_cu_cubin_len, "fmha_v2_fp16_256_32_sm80_kernel", 20480, 128, 0, false},
    {DATA_TYPE_FP16, 256, 32, kSM_80, cubin_fmha_v2_fp16_256_32_sm80_cu_cubin,
        cubin_fmha_v2_fp16_256_32_sm80_cu_cubin_len, "fmha_v2_fp16_256_32_sm80_kernel_nl", 20480, 128, 32, false},

    // {int8} x {512} x {64, 32} x {sm80} x {normal, noloop}
    {DATA_TYPE_INT8, 512, 64, kSM_80, cubin_fmha_v2_int8_512_64_sm80_cu_cubin,
        cubin_fmha_v2_int8_512_64_sm80_cu_cubin_len, "fmha_v2_int8_512_64_sm80_kernel", 73728, 256, 0, false},
    {DATA_TYPE_INT8, 512, 64, kSM_80, cubin_fmha_v2_int8_512_64_sm80_cu_cubin,
        cubin_fmha_v2_int8_512_64_sm80_cu_cubin_len, "fmha_v2_int8_512_64_sm80_kernel_nl", 73728, 256, 32, false},
    {DATA_TYPE_INT8, 512, 32, kSM_80, cubin_fmha_v2_int8_512_32_sm80_cu_cubin,
        cubin_fmha_v2_int8_512_32_sm80_cu_cubin_len, "fmha_v2_int8_512_32_sm80_kernel", 40960, 256, 0, false},
    {DATA_TYPE_INT8, 512, 32, kSM_80, cubin_fmha_v2_int8_512_32_sm80_cu_cubin,
        cubin_fmha_v2_int8_512_32_sm80_cu_cubin_len, "fmha_v2_int8_512_32_sm80_kernel_nl", 40960, 256, 32, false},

    // {fp16} x {512} x {64, 32} x {sm80} x {normal, noloop}
    {DATA_TYPE_FP16, 512, 64, kSM_80, cubin_fmha_v2_fp16_512_64_sm80_cu_cubin,
        cubin_fmha_v2_fp16_512_64_sm80_cu_cubin_len, "fmha_v2_fp16_512_64_sm80_kernel", 73728, 256, 0, false},
    {DATA_TYPE_FP16, 512, 64, kSM_80, cubin_fmha_v2_fp16_512_64_sm80_cu_cubin,
        cubin_fmha_v2_fp16_512_64_sm80_cu_cubin_len, "fmha_v2_fp16_512_64_sm80_kernel_nl", 73728, 256, 32, false},
    {DATA_TYPE_FP16, 512, 32, kSM_80, cubin_fmha_v2_fp16_512_32_sm80_cu_cubin,
        cubin_fmha_v2_fp16_512_32_sm80_cu_cubin_len, "fmha_v2_fp16_512_32_sm80_kernel", 40960, 256, 0, false},
    {DATA_TYPE_FP16, 512, 32, kSM_80, cubin_fmha_v2_fp16_512_32_sm80_cu_cubin,
        cubin_fmha_v2_fp16_512_32_sm80_cu_cubin_len, "fmha_v2_fp16_512_32_sm80_kernel_nl", 40960, 256, 32, false},

    // Ampere
    {DATA_TYPE_FP16, 64, 64, kSM_80, fused_multihead_attention_v2_fp16_64_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_fp16_64_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_fp16_64_64_kernel_sm80", 32768, 128, 0, false},
    {DATA_TYPE_FP16, 96, 64, kSM_80, fused_multihead_attention_v2_fp16_96_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_fp16_96_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_fp16_96_64_kernel_sm80", 49152, 128, 0, false},
    {DATA_TYPE_FP16, 128, 64, kSM_80, fused_multihead_attention_v2_fp16_128_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_fp16_128_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_fp16_128_64_kernel_sm80_noloop", 40960, 128, 32, false},
    {DATA_TYPE_FP16, 128, 64, kSM_80, fused_multihead_attention_v2_fp16_128_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_fp16_128_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_fp16_128_64_kernel_sm80", 65536, 128, 0, false},
    {DATA_TYPE_FP16, 256, 64, kSM_80, fused_multihead_attention_v2_fp16_256_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_fp16_256_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_fp16_256_64_kernel_sm80_noloop", 73728, 128, 32, false},
    {DATA_TYPE_FP16, 256, 64, kSM_80, fused_multihead_attention_v2_fp16_256_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_fp16_256_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_fp16_256_64_kernel_sm80", 73728, 128, 0, false},
    {DATA_TYPE_FP16, 384, 64, kSM_80, fused_multihead_attention_v2_fp16_384_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_fp16_384_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_fp16_384_64_kernel_sm80_noloop", 114688, 256, 48, false},
    {DATA_TYPE_FP16, 384, 64, kSM_80, fused_multihead_attention_v2_fp16_384_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_fp16_384_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_fp16_384_64_kernel_sm80", 114688, 256, 0, false},

    {DATA_TYPE_INT8, 128, 64, kSM_80, fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm80_interleaved_noloop", 20480, 128, 16, true},
    {DATA_TYPE_INT8, 128, 64, kSM_80, fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm80_noloop", 20480, 128, 16, false},
    {DATA_TYPE_INT8, 128, 64, kSM_80, fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm80_interleaved", 24576, 128, 0, true},
    {DATA_TYPE_INT8, 128, 64, kSM_80, fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm80", 32768, 128, 0, false},
    {DATA_TYPE_INT8, 192, 64, kSM_80, fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm80_interleaved_noloop", 28672, 128, 32, true},
    {DATA_TYPE_INT8, 192, 64, kSM_80, fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm80_noloop", 28672, 128, 32, false},
    {DATA_TYPE_INT8, 192, 64, kSM_80, fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm80_interleaved", 32768, 128, 0, true},
    {DATA_TYPE_INT8, 192, 64, kSM_80, fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm80", 32768, 128, 0, false},
    {DATA_TYPE_INT8, 256, 64, kSM_80, fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm80_interleaved_noloop", 36864, 128, 32, true},
    {DATA_TYPE_INT8, 256, 64, kSM_80, fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm80_noloop", 36864, 128, 32, false},
    {DATA_TYPE_INT8, 256, 64, kSM_80, fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm80_interleaved", 36864, 128, 0, true},
    {DATA_TYPE_INT8, 256, 64, kSM_80, fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm80", 36864, 128, 0, false},
    {DATA_TYPE_INT8, 384, 64, kSM_80, fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm80_interleaved_noloop", 53248, 128, 32, true},
    {DATA_TYPE_INT8, 384, 64, kSM_80, fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm80_noloop", 53248, 128, 32, false},
    {DATA_TYPE_INT8, 384, 64, kSM_80, fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm80_interleaved", 51200, 128, 0, true},
    {DATA_TYPE_INT8, 384, 64, kSM_80, fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm80_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm80", 53248, 128, 0, false},

    // GA10x
    // Note: For GA10X keep only kernels whose sharedMemBytes < 100KiB
    {DATA_TYPE_FP16, 64, 64, kSM_86, fused_multihead_attention_v2_fp16_64_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_fp16_64_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_fp16_64_64_kernel_sm80", 32768, 128, 0, false},
    {DATA_TYPE_FP16, 96, 64, kSM_86, fused_multihead_attention_v2_fp16_96_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_fp16_96_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_fp16_96_64_kernel_sm80", 49152, 128, 0, false},
    {DATA_TYPE_FP16, 128, 64, kSM_86, fused_multihead_attention_v2_fp16_128_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_fp16_128_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_fp16_128_64_kernel_sm80_noloop", 40960, 128, 32, false},
    {DATA_TYPE_FP16, 128, 64, kSM_86, fused_multihead_attention_v2_fp16_128_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_fp16_128_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_fp16_128_64_kernel_sm80", 65536, 128, 0, false},
    {DATA_TYPE_FP16, 256, 64, kSM_86, fused_multihead_attention_v2_fp16_256_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_fp16_256_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_fp16_256_64_kernel_sm80_noloop", 73728, 128, 32, false},
    {DATA_TYPE_FP16, 256, 64, kSM_86, fused_multihead_attention_v2_fp16_256_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_fp16_256_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_fp16_256_64_kernel_sm80", 73728, 128, 0, false},
    {DATA_TYPE_FP16, 384, 64, kSM_86, fused_multihead_attention_v2_fp16_384_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_fp16_384_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_fp16_384_64_kernel_sm80_noloop", 65536, 256, 48, false},
    {DATA_TYPE_FP16, 384, 64, kSM_86, fused_multihead_attention_v2_fp16_384_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_fp16_384_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_fp16_384_64_kernel_sm80", 65536, 256, 0, false},

    {DATA_TYPE_INT8, 128, 64, kSM_86, fused_multihead_attention_v2_int8_128_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm80_interleaved_noloop", 20480, 128, 16, true},
    {DATA_TYPE_INT8, 128, 64, kSM_86, fused_multihead_attention_v2_int8_128_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm80_noloop", 20480, 128, 16, false},
    {DATA_TYPE_INT8, 128, 64, kSM_86, fused_multihead_attention_v2_int8_128_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm80_interleaved", 24576, 128, 0, true},
    {DATA_TYPE_INT8, 128, 64, kSM_86, fused_multihead_attention_v2_int8_128_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_int8_128_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_int8_128_64_kernel_sm80", 32768, 128, 0, false},
    {DATA_TYPE_INT8, 192, 64, kSM_86, fused_multihead_attention_v2_int8_192_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm86_interleaved_noloop", 28672, 128, 32, true},
    {DATA_TYPE_INT8, 192, 64, kSM_86, fused_multihead_attention_v2_int8_192_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm86_noloop", 28672, 128, 32, false},
    {DATA_TYPE_INT8, 192, 64, kSM_86, fused_multihead_attention_v2_int8_192_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm86_interleaved", 32768, 128, 0, true},
    {DATA_TYPE_INT8, 192, 64, kSM_86, fused_multihead_attention_v2_int8_192_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_int8_192_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_int8_192_64_kernel_sm86", 32768, 128, 0, false},
    {DATA_TYPE_INT8, 256, 64, kSM_86, fused_multihead_attention_v2_int8_256_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm86_interleaved_noloop", 36864, 128, 32, true},
    {DATA_TYPE_INT8, 256, 64, kSM_86, fused_multihead_attention_v2_int8_256_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm86_noloop", 36864, 128, 32, false},
    {DATA_TYPE_INT8, 256, 64, kSM_86, fused_multihead_attention_v2_int8_256_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm86_interleaved", 36864, 128, 0, true},
    {DATA_TYPE_INT8, 256, 64, kSM_86, fused_multihead_attention_v2_int8_256_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_int8_256_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_int8_256_64_kernel_sm86", 36864, 128, 0, false},
    {DATA_TYPE_INT8, 384, 64, kSM_86, fused_multihead_attention_v2_int8_384_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm86_interleaved_noloop", 28672, 128, 32, true},
    {DATA_TYPE_INT8, 384, 64, kSM_86, fused_multihead_attention_v2_int8_384_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm86_noloop", 28672, 128, 32, false},
    {DATA_TYPE_INT8, 384, 64, kSM_86, fused_multihead_attention_v2_int8_384_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm86_interleaved", 28672, 128, 0, true},
    {DATA_TYPE_INT8, 384, 64, kSM_86, fused_multihead_attention_v2_int8_384_64_kernel_sm86_cubin,
        fused_multihead_attention_v2_int8_384_64_kernel_sm86_cubin_len,
        "fused_multihead_attention_v2_int8_384_64_kernel_sm86", 28672, 128, 0, false},
#endif
};

class FusedMultiHeadAttentionXMMAKernelV2
    : public TFusedMultiHeadAttentionXMMAKernel<FusedMultiHeadAttentionKernelMetaInfoV2,
          Fused_multihead_attention_params_v2>
{
public:
    FusedMultiHeadAttentionXMMAKernelV2(
        const FusedMultiHeadAttentionKernelMetaInfoV2* pMetaStart, uint32_t nMetaCount, Data_type type, uint32_t sm)
        : TFusedMultiHeadAttentionXMMAKernel<FusedMultiHeadAttentionKernelMetaInfoV2,
              Fused_multihead_attention_params_v2>(pMetaStart, nMetaCount, type, sm)
    {
    }

    inline uint64_t hashID(uint32_t s, uint32_t headsize, bool interleaved, bool unroll) const
    {
        // we only have 30 bits room for head size
        ASSERT(headsize <= 0x3FFFFFFF);
        return static_cast<uint64_t>(s) << 32 | (headsize << 2) | (interleaved ? 2U : 0U) | (unroll ? 1U : 0U);
    }

    virtual uint64_t hashID(const KernelMeta& kernelMeta) const
    {
        return hashID(kernelMeta.mS, kernelMeta.mD, kernelMeta.mInterleaved, kernelMeta.mUnrollStep);
    }

    virtual void run(Fused_multihead_attention_params_v2& params, cudaStream_t ss) const
    {
        if (params.interleaved)
        {
            assert(mDataType == bert::DATA_TYPE_INT8);
        }

        bool forceUnroll = params.force_unroll;
        if (!forceUnroll && !params.ignore_b1opt && mSM >= kSM_75)
        {
            const struct
            {
                uint32_t mSM;
                Data_type mDataType;
                int mS;
                int mMaxBatch;
            } unrollList[]
                = { {kSM_75, bert::DATA_TYPE_FP16, 256, 1},
                      {kSM_75, bert::DATA_TYPE_FP16, 384, 1},
                      {kSM_75, bert::DATA_TYPE_INT8, 128, 1},
                      {kSM_75, bert::DATA_TYPE_INT8, 192, 2},
                      {kSM_75, bert::DATA_TYPE_INT8, 256, 1},
                      {kSM_75, bert::DATA_TYPE_INT8, 384, 1},
#if CUDA_VERSION >= 11000
                      {kSM_80, bert::DATA_TYPE_FP16, 128, 4},
                      {kSM_80, bert::DATA_TYPE_FP16, 256, 4},
                      {kSM_80, bert::DATA_TYPE_FP16, 384, 4},
                      {kSM_80, bert::DATA_TYPE_INT8, 128, 4},
                      {kSM_80, bert::DATA_TYPE_INT8, 192, 16},
                      {kSM_80, bert::DATA_TYPE_INT8, 256, 8},
                      {kSM_80, bert::DATA_TYPE_INT8, 384, 8},

                      {kSM_86, bert::DATA_TYPE_FP16, 128, 4},
                      {kSM_86, bert::DATA_TYPE_FP16, 256, 4},
                      {kSM_86, bert::DATA_TYPE_INT8, 128, 4},
                      {kSM_86, bert::DATA_TYPE_INT8, 192, 16},
                      {kSM_86, bert::DATA_TYPE_INT8, 256, 8},
                      {kSM_86, bert::DATA_TYPE_INT8, 384, 8},
#endif
                  };
            for (uint32_t i = 0u; i < sizeof(unrollList) / sizeof(unrollList[0]); ++i)
            {
                if (mSM == unrollList[i].mSM && mDataType == unrollList[i].mDataType && params.s == unrollList[i].mS
                    && params.b <= unrollList[i].mMaxBatch)
                {
                    forceUnroll = true;
                    break;
                }
            }
        }

        const auto findIter = mFunctions.find(hashID(params.s, params.d, params.interleaved, forceUnroll));
        ASSERT(findIter != mFunctions.end());

        const auto& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        const CUfunction func = findIter->second.mDeviceFunction;

        void* kernelParams[] = {&params, nullptr};
        if (!forceUnroll)
        {
            cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, 1, kernelMeta.mThreadsPerCTA, 1, 1,
                           kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
                mDriver);
        }
        else
        {
            int unroll = kernelMeta.mS / kernelMeta.mUnrollStep;
            assert(kernelMeta.mS == kernelMeta.mUnrollStep * unroll);
            cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, unroll, kernelMeta.mThreadsPerCTA, 1, 1,
                           kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
                mDriver);
        }
    }
};

using FusedMHAKernelFactoryV2 = TFusedMHAKernelFactory<FusedMultiHeadAttentionXMMAKernelV2>;

inline const FusedMultiHeadAttentionXMMAKernelV2* getXMMAKernelsV2(Data_type type, uint32_t sm)
{
    return FusedMHAKernelFactoryV2::Get().getXMMAKernels(
        sMhaKernelMetaInfosV2, sizeof(sMhaKernelMetaInfosV2) / sizeof(sMhaKernelMetaInfosV2[0]), type, sm);
}

} // namespace bert
