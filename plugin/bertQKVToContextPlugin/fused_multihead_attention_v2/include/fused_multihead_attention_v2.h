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

#ifndef _BERT_FMHAV2_FMHAV2
#define _BERT_FMHAV2_FMHAV2
#include "bertQKVToContextPlugin/fused_multihead_attention/include/fused_multihead_attention.h"
#include "bertQKVToContextPlugin/fused_multihead_attention/include/fused_multihead_attention_common.h"
#include "common/bertCommon.h"
#include <cstdint>

namespace nvinfer1
{
namespace plugin
{
namespace bert
{
struct Fused_multihead_attention_params_v2
{
    // The QKV matrices.
    void* qkv_ptr{};
    // The mask to implement drop-out.
    void* packed_mask_ptr{};
    // The O matrix (output).
    void* o_ptr{};

    // The stride between rows of the Q, K and V matrices.
    int64_t qkv_stride_in_bytes{};
    // The stride between matrices of packed mask.
    int64_t packed_mask_stride_in_bytes{};
    // The stride between rows of O.
    int64_t o_stride_in_bytes;

#if defined(STORE_P)
    // The pointer to the P matrix (for debugging).
    void* p_ptr{};
    // The stride between rows of the P matrix (for debugging).
    int64_t p_stride_in_bytes{};
#endif // defined(STORE_P)

#if defined(STORE_S)
    // The pointer to the S matrix (for debugging).
    void* s_ptr{};
    // The stride between rows of the S matrix (for debugging).
    int64_t s_stride_in_bytes{};
#endif // defined(STORE_S)

    // The dimensions.
    int32_t b{};
    int32_t h{};
    int32_t s{};
    int32_t d{};
    // The scaling factors for the kernel.
    uint32_t scale_bmm1{};
    uint32_t scale_softmax{};
    uint32_t scale_bmm2{};

    // Do we use Niall's trick to avoid I2F/F2I in the INT8 kernel.
    // See https://confluence.nvidia.com/pages/viewpage.action?pageId=302779721 for details.
    bool enable_i2f_trick{};

    // array of length b+1 holding prefix sum of actual sequence lenghts
    int32_t* cu_seqlens{};

    // use C/32 Format.
    bool interleaved{};
    bool ignore_b1opt{};
    bool force_unroll{};
    bool use_int8_scale_max{};

    // The number of heads computed by one iteration of the wave.
    int32_t heads_per_wave{};
    // Buffers to perform a global sync and a critical section.
    int32_t* counters{};
    int32_t* max_barriers{};
    int32_t* sum_barriers{};
    int32_t* locks{};
    // Scratch buffers to finalize softmax.
    float* max_scratch_ptr{};
    float* sum_scratch_ptr{};
    // Scratch buffer to finalize the output (not needed for FP16).
    int* o_scratch_ptr{};

    void clear()
    {
        *this = Fused_multihead_attention_params_v2();
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

extern unsigned char cubin_fmha_v2_int8_64_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_il_int8_64_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_96_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_il_int8_96_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_512_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_512_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_256_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_128_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_il_int8_128_32_sm80_cu_cubin[];
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

extern unsigned char cubin_fmha_v2_int8_384_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_il_int8_384_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_256_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_il_int8_256_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_192_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_il_int8_192_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_128_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_il_int8_128_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_96_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_il_int8_96_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_64_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_il_int8_64_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_384_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_256_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_128_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_96_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_64_64_sm87_cu_cubin[];

extern unsigned char cubin_fmha_v2_int8_512_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_384_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_il_int8_384_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_256_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_il_int8_256_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_192_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_il_int8_192_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_128_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_il_int8_128_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_96_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_il_int8_96_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_int8_64_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_il_int8_64_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_512_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_384_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_256_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_128_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_96_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_64_64_sm90_cu_cubin[];

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

extern uint32_t cubin_fmha_v2_il_int8_96_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_il_int8_64_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_96_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_64_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_512_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_512_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_256_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_128_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_il_int8_128_32_sm80_cu_cubin_len;
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

extern uint32_t cubin_fmha_v2_int8_384_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_il_int8_384_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_256_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_il_int8_256_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_192_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_il_int8_192_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_128_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_il_int8_128_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_96_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_il_int8_96_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_64_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_il_int8_64_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_384_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_256_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_128_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_96_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_64_64_sm87_cu_cubin_len;

extern uint32_t cubin_fmha_v2_int8_512_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_384_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_il_int8_384_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_256_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_il_int8_256_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_192_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_il_int8_192_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_128_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_il_int8_128_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_96_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_il_int8_96_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_int8_64_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_il_int8_64_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_512_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_384_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_256_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_128_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_96_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_64_64_sm90_cu_cubin_len;

#if !(defined(ENABLE_SM72) || defined(ENABLE_SM75) || defined(ENABLE_SM80) || defined(ENABLE_SM86) || defined(ENABLE_SM87) || defined(ENABLE_SM89) || defined(ENABLE_SM90))
#error This file can only be included one of sm 72, 75, 80, 86, 87, 89, or 90 are defined.
#endif
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
#if defined(ENABLE_SM72)
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
#endif // defined(ENABLE_SM72)
#if defined(ENABLE_SM75)
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
#endif // defined(ENABLE_SM75)

#if defined(ENABLE_SM80) || defined(ENABLE_SM86) || defined(ENABLE_SM89)
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
    {DATA_TYPE_INT8, 128, 32, kSM_80, cubin_fmha_v2_il_int8_128_32_sm80_cu_cubin,
        cubin_fmha_v2_il_int8_128_32_sm80_cu_cubin_len, "fmha_v2_il_int8_128_32_sm80_kernel_nl", 10240, 128, 16, true},

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

    {DATA_TYPE_INT8, 64, 64, kSM_80, cubin_fmha_v2_int8_64_64_sm80_cu_cubin, cubin_fmha_v2_int8_64_64_sm80_cu_cubin_len,
        "fmha_v2_int8_64_64_sm80_kernel", 24576, 128, 0, false},
    {DATA_TYPE_INT8, 96, 64, kSM_80, cubin_fmha_v2_int8_96_64_sm80_cu_cubin, cubin_fmha_v2_int8_96_64_sm80_cu_cubin_len,
        "fmha_v2_int8_96_64_sm80_kernel", 28672, 128, 0, false},
    {DATA_TYPE_INT8, 64, 64, kSM_80, cubin_fmha_v2_il_int8_64_64_sm80_cu_cubin,
        cubin_fmha_v2_il_int8_64_64_sm80_cu_cubin_len, "fmha_v2_il_int8_64_64_sm80_kernel", 20480, 128, 0, true},
    {DATA_TYPE_INT8, 96, 64, kSM_80, cubin_fmha_v2_il_int8_96_64_sm80_cu_cubin,
        cubin_fmha_v2_il_int8_96_64_sm80_cu_cubin_len, "fmha_v2_il_int8_96_64_sm80_kernel", 22528, 128, 0, true},

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
#endif // defined(ENABLE_SM80) || defined(ENABLE_SM86) || defined(ENABLE_SM89)
#if defined(ENABLE_SM86)
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
#endif // defined(ENABLE_SM86)

#if defined(ENABLE_SM87)
    // GA10b (Orin-Auto)
    // Adding head {64} x seq_len {64,96,128,196,256,384} kernels
    {DATA_TYPE_INT8, 384, 64, kSM_87, cubin_fmha_v2_int8_384_64_sm87_cu_cubin,
        cubin_fmha_v2_int8_384_64_sm87_cu_cubin_len, "fmha_v2_int8_384_64_sm87_kernel", 53248, 128, 0, false},
    {DATA_TYPE_INT8, 384, 64, kSM_87, cubin_fmha_v2_int8_384_64_sm87_cu_cubin,
        cubin_fmha_v2_int8_384_64_sm87_cu_cubin_len, "fmha_v2_int8_384_64_sm87_kernel_nl", 53248, 128, 32, false},
    {DATA_TYPE_INT8, 384, 64, kSM_87, cubin_fmha_v2_il_int8_384_64_sm87_cu_cubin,
        cubin_fmha_v2_il_int8_384_64_sm87_cu_cubin_len, "fmha_v2_il_int8_384_64_sm87_kernel", 51200, 128, 0, true},
    {DATA_TYPE_INT8, 384, 64, kSM_87, cubin_fmha_v2_il_int8_384_64_sm87_cu_cubin,
        cubin_fmha_v2_il_int8_384_64_sm87_cu_cubin_len, "fmha_v2_il_int8_384_64_sm87_kernel_nl", 51200, 128, 32, true},
    {DATA_TYPE_INT8, 256, 64, kSM_87, cubin_fmha_v2_int8_256_64_sm87_cu_cubin,
        cubin_fmha_v2_int8_256_64_sm87_cu_cubin_len, "fmha_v2_int8_256_64_sm87_kernel", 36864, 128, 0, false},
    {DATA_TYPE_INT8, 256, 64, kSM_87, cubin_fmha_v2_int8_256_64_sm87_cu_cubin,
        cubin_fmha_v2_int8_256_64_sm87_cu_cubin_len, "fmha_v2_int8_256_64_sm87_kernel_nl", 36864, 128, 32, false},
    {DATA_TYPE_INT8, 256, 64, kSM_87, cubin_fmha_v2_il_int8_256_64_sm87_cu_cubin,
        cubin_fmha_v2_il_int8_256_64_sm87_cu_cubin_len, "fmha_v2_il_int8_256_64_sm87_kernel", 34816, 128, 0, true},
    {DATA_TYPE_INT8, 256, 64, kSM_87, cubin_fmha_v2_il_int8_256_64_sm87_cu_cubin,
        cubin_fmha_v2_il_int8_256_64_sm87_cu_cubin_len, "fmha_v2_il_int8_256_64_sm87_kernel_nl", 34816, 128, 32, true},
    {DATA_TYPE_INT8, 192, 64, kSM_87, cubin_fmha_v2_int8_192_64_sm87_cu_cubin,
        cubin_fmha_v2_int8_192_64_sm87_cu_cubin_len, "fmha_v2_int8_192_64_sm87_kernel", 28672, 128, 0, false},
    {DATA_TYPE_INT8, 192, 64, kSM_87, cubin_fmha_v2_int8_192_64_sm87_cu_cubin,
        cubin_fmha_v2_int8_192_64_sm87_cu_cubin_len, "fmha_v2_int8_192_64_sm87_kernel_nl", 28672, 128, 32, false},
    {DATA_TYPE_INT8, 192, 64, kSM_87, cubin_fmha_v2_il_int8_192_64_sm87_cu_cubin,
        cubin_fmha_v2_il_int8_192_64_sm87_cu_cubin_len, "fmha_v2_il_int8_192_64_sm87_kernel", 26624, 128, 0, true},
    {DATA_TYPE_INT8, 192, 64, kSM_87, cubin_fmha_v2_il_int8_192_64_sm87_cu_cubin,
        cubin_fmha_v2_il_int8_192_64_sm87_cu_cubin_len, "fmha_v2_il_int8_192_64_sm87_kernel_nl", 26624, 128, 32, true},
    {DATA_TYPE_INT8, 128, 64, kSM_87, cubin_fmha_v2_int8_128_64_sm87_cu_cubin,
        cubin_fmha_v2_int8_128_64_sm87_cu_cubin_len, "fmha_v2_int8_128_64_sm87_kernel", 32768, 128, 0, false},
    {DATA_TYPE_INT8, 128, 64, kSM_87, cubin_fmha_v2_int8_128_64_sm87_cu_cubin,
        cubin_fmha_v2_int8_128_64_sm87_cu_cubin_len, "fmha_v2_int8_128_64_sm87_kernel_nl", 20480, 128, 16, false},
    {DATA_TYPE_INT8, 128, 64, kSM_87, cubin_fmha_v2_il_int8_128_64_sm87_cu_cubin,
        cubin_fmha_v2_il_int8_128_64_sm87_cu_cubin_len, "fmha_v2_il_int8_128_64_sm87_kernel", 24576, 128, 0, true},
    {DATA_TYPE_INT8, 128, 64, kSM_87, cubin_fmha_v2_il_int8_128_64_sm87_cu_cubin,
        cubin_fmha_v2_il_int8_128_64_sm87_cu_cubin_len, "fmha_v2_il_int8_128_64_sm87_kernel_nl", 18432, 128, 16, true},
    {DATA_TYPE_INT8, 96, 64, kSM_87, cubin_fmha_v2_int8_96_64_sm87_cu_cubin,
        cubin_fmha_v2_int8_96_64_sm87_cu_cubin_len, "fmha_v2_int8_96_64_sm87_kernel", 28672, 128, 0, false},
    {DATA_TYPE_INT8, 96, 64, kSM_87, cubin_fmha_v2_int8_96_64_sm87_cu_cubin,
        cubin_fmha_v2_int8_96_64_sm87_cu_cubin_len, "fmha_v2_int8_96_64_sm87_kernel_nl", 20480, 128, 16, false},
    {DATA_TYPE_INT8, 96, 64, kSM_87, cubin_fmha_v2_il_int8_96_64_sm87_cu_cubin,
        cubin_fmha_v2_il_int8_96_64_sm87_cu_cubin_len, "fmha_v2_il_int8_96_64_sm87_kernel", 22528, 128, 0, true},
    {DATA_TYPE_INT8, 96, 64, kSM_87, cubin_fmha_v2_il_int8_96_64_sm87_cu_cubin,
        cubin_fmha_v2_il_int8_96_64_sm87_cu_cubin_len, "fmha_v2_il_int8_96_64_sm87_kernel_nl", 18432, 128, 16, true},
    {DATA_TYPE_INT8, 64, 64, kSM_87, cubin_fmha_v2_int8_64_64_sm87_cu_cubin,
        cubin_fmha_v2_int8_64_64_sm87_cu_cubin_len, "fmha_v2_int8_64_64_sm87_kernel", 24576, 128, 0, false},
    {DATA_TYPE_INT8, 64, 64, kSM_87, cubin_fmha_v2_il_int8_64_64_sm87_cu_cubin,
        cubin_fmha_v2_il_int8_64_64_sm87_cu_cubin_len, "fmha_v2_il_int8_64_64_sm87_kernel", 20480, 128, 0, true},
    {DATA_TYPE_FP16, 384, 64, kSM_87, cubin_fmha_v2_fp16_384_64_sm87_cu_cubin,
        cubin_fmha_v2_fp16_384_64_sm87_cu_cubin_len, "fmha_v2_fp16_384_64_sm87_kernel", 65536, 256, 0, false},
    {DATA_TYPE_FP16, 384, 64, kSM_87, cubin_fmha_v2_fp16_384_64_sm87_cu_cubin,
        cubin_fmha_v2_fp16_384_64_sm87_cu_cubin_len, "fmha_v2_fp16_384_64_sm87_kernel_nl", 57344, 256, 32, false},
    {DATA_TYPE_FP16, 256, 64, kSM_87, cubin_fmha_v2_fp16_256_64_sm87_cu_cubin,
        cubin_fmha_v2_fp16_256_64_sm87_cu_cubin_len, "fmha_v2_fp16_256_64_sm87_kernel", 40960, 128, 0, false},
    {DATA_TYPE_FP16, 256, 64, kSM_87, cubin_fmha_v2_fp16_256_64_sm87_cu_cubin,
        cubin_fmha_v2_fp16_256_64_sm87_cu_cubin_len, "fmha_v2_fp16_256_64_sm87_kernel_nl", 40960, 128, 32, false},
    {DATA_TYPE_FP16, 128, 64, kSM_87, cubin_fmha_v2_fp16_128_64_sm87_cu_cubin,
        cubin_fmha_v2_fp16_128_64_sm87_cu_cubin_len, "fmha_v2_fp16_128_64_sm87_kernel", 65536, 128, 0, false},
    {DATA_TYPE_FP16, 128, 64, kSM_87, cubin_fmha_v2_fp16_128_64_sm87_cu_cubin,
        cubin_fmha_v2_fp16_128_64_sm87_cu_cubin_len, "fmha_v2_fp16_128_64_sm87_kernel_nl", 36864, 128, 16, false},
    {DATA_TYPE_FP16, 96, 64, kSM_87, cubin_fmha_v2_fp16_96_64_sm87_cu_cubin, cubin_fmha_v2_fp16_96_64_sm87_cu_cubin_len,
        "fmha_v2_fp16_96_64_sm87_kernel", 49152, 128, 0, false},
    {DATA_TYPE_FP16, 64, 64, kSM_87, cubin_fmha_v2_fp16_64_64_sm87_cu_cubin, cubin_fmha_v2_fp16_64_64_sm87_cu_cubin_len,
        "fmha_v2_fp16_64_64_sm87_kernel", 32768, 128, 0, false},
    {DATA_TYPE_FP16, 64, 64, kSM_87, cubin_fmha_v2_fp16_64_64_sm87_cu_cubin, cubin_fmha_v2_fp16_64_64_sm87_cu_cubin_len,
        "fmha_v2_fp16_64_64_sm87_kernel_nl", 20480, 128, 16, false},
#endif // defined(ENABLE_SM87)

#if defined(ENABLE_SM90)
    // GH100 hopper
    {DATA_TYPE_INT8, 512, 64, kSM_90, cubin_fmha_v2_int8_512_64_sm90_cu_cubin,
        cubin_fmha_v2_int8_512_64_sm90_cu_cubin_len, "fmha_v2_int8_512_64_sm90_kernel", 73728, 256, 0, false},
    {DATA_TYPE_INT8, 512, 64, kSM_90, cubin_fmha_v2_int8_512_64_sm90_cu_cubin,
        cubin_fmha_v2_int8_512_64_sm90_cu_cubin_len, "fmha_v2_int8_512_64_sm90_kernel_nl", 73728, 256, 32, false},
    {DATA_TYPE_INT8, 384, 64, kSM_90, cubin_fmha_v2_int8_384_64_sm90_cu_cubin,
        cubin_fmha_v2_int8_384_64_sm90_cu_cubin_len, "fmha_v2_int8_384_64_sm90_kernel", 53248, 128, 0, false},
    {DATA_TYPE_INT8, 384, 64, kSM_90, cubin_fmha_v2_int8_384_64_sm90_cu_cubin,
        cubin_fmha_v2_int8_384_64_sm90_cu_cubin_len, "fmha_v2_int8_384_64_sm90_kernel_nl", 53248, 128, 32, false},
    {DATA_TYPE_INT8, 384, 64, kSM_90, cubin_fmha_v2_il_int8_384_64_sm90_cu_cubin,
        cubin_fmha_v2_il_int8_384_64_sm90_cu_cubin_len, "fmha_v2_il_int8_384_64_sm90_kernel", 51200, 128, 0, true},
    {DATA_TYPE_INT8, 384, 64, kSM_90, cubin_fmha_v2_il_int8_384_64_sm90_cu_cubin,
        cubin_fmha_v2_il_int8_384_64_sm90_cu_cubin_len, "fmha_v2_il_int8_384_64_sm90_kernel_nl", 51200, 128, 32, true},
    {DATA_TYPE_INT8, 256, 64, kSM_90, cubin_fmha_v2_int8_256_64_sm90_cu_cubin,
        cubin_fmha_v2_int8_256_64_sm90_cu_cubin_len, "fmha_v2_int8_256_64_sm90_kernel", 36864, 128, 0, false},
    {DATA_TYPE_INT8, 256, 64, kSM_90, cubin_fmha_v2_int8_256_64_sm90_cu_cubin,
        cubin_fmha_v2_int8_256_64_sm90_cu_cubin_len, "fmha_v2_int8_256_64_sm90_kernel_nl", 36864, 128, 32, false},
    {DATA_TYPE_INT8, 256, 64, kSM_90, cubin_fmha_v2_il_int8_256_64_sm90_cu_cubin,
        cubin_fmha_v2_il_int8_256_64_sm90_cu_cubin_len, "fmha_v2_il_int8_256_64_sm90_kernel", 34816, 128, 0, true},
    {DATA_TYPE_INT8, 256, 64, kSM_90, cubin_fmha_v2_il_int8_256_64_sm90_cu_cubin,
        cubin_fmha_v2_il_int8_256_64_sm90_cu_cubin_len, "fmha_v2_il_int8_256_64_sm90_kernel_nl", 34816, 128, 32, true},
    {DATA_TYPE_INT8, 192, 64, kSM_90, cubin_fmha_v2_int8_192_64_sm90_cu_cubin,
        cubin_fmha_v2_int8_192_64_sm90_cu_cubin_len, "fmha_v2_int8_192_64_sm90_kernel", 36864, 128, 0, false},
    {DATA_TYPE_INT8, 192, 64, kSM_90, cubin_fmha_v2_int8_192_64_sm90_cu_cubin,
        cubin_fmha_v2_int8_192_64_sm90_cu_cubin_len, "fmha_v2_int8_192_64_sm90_kernel_nl", 36864, 128, 32, false},
    {DATA_TYPE_INT8, 192, 64, kSM_90, cubin_fmha_v2_il_int8_192_64_sm90_cu_cubin,
        cubin_fmha_v2_il_int8_192_64_sm90_cu_cubin_len, "fmha_v2_il_int8_192_64_sm90_kernel", 26624, 128, 0, true},
    {DATA_TYPE_INT8, 192, 64, kSM_90, cubin_fmha_v2_il_int8_192_64_sm90_cu_cubin,
        cubin_fmha_v2_il_int8_192_64_sm90_cu_cubin_len, "fmha_v2_il_int8_192_64_sm90_kernel_nl", 26624, 128, 32, true},
    {DATA_TYPE_INT8, 128, 64, kSM_90, cubin_fmha_v2_int8_128_64_sm90_cu_cubin,
        cubin_fmha_v2_int8_128_64_sm90_cu_cubin_len, "fmha_v2_int8_128_64_sm90_kernel", 32768, 128, 0, false},
    {DATA_TYPE_INT8, 128, 64, kSM_90, cubin_fmha_v2_int8_128_64_sm90_cu_cubin,
        cubin_fmha_v2_int8_128_64_sm90_cu_cubin_len, "fmha_v2_int8_128_64_sm90_kernel_nl", 20480, 128, 16, false},
    {DATA_TYPE_INT8, 128, 64, kSM_90, cubin_fmha_v2_il_int8_128_64_sm90_cu_cubin,
        cubin_fmha_v2_il_int8_128_64_sm90_cu_cubin_len, "fmha_v2_il_int8_128_64_sm90_kernel", 24576, 128, 0, true},
    {DATA_TYPE_INT8, 128, 64, kSM_90, cubin_fmha_v2_il_int8_128_64_sm90_cu_cubin,
        cubin_fmha_v2_il_int8_128_64_sm90_cu_cubin_len, "fmha_v2_il_int8_128_64_sm90_kernel_nl", 18432, 128, 16, true},
    {DATA_TYPE_INT8, 96, 64, kSM_90, cubin_fmha_v2_int8_96_64_sm90_cu_cubin,
        cubin_fmha_v2_int8_96_64_sm90_cu_cubin_len, "fmha_v2_int8_96_64_sm90_kernel", 28672, 128, 0, false},
    {DATA_TYPE_INT8, 96, 64, kSM_90, cubin_fmha_v2_int8_96_64_sm90_cu_cubin,
        cubin_fmha_v2_int8_96_64_sm90_cu_cubin_len, "fmha_v2_int8_96_64_sm90_kernel_nl", 20480, 128, 16, false},
    {DATA_TYPE_INT8, 96, 64, kSM_90, cubin_fmha_v2_il_int8_96_64_sm90_cu_cubin,
        cubin_fmha_v2_il_int8_96_64_sm90_cu_cubin_len, "fmha_v2_il_int8_96_64_sm90_kernel", 22528, 128, 0, true},
    {DATA_TYPE_INT8, 96, 64, kSM_90, cubin_fmha_v2_il_int8_96_64_sm90_cu_cubin,
        cubin_fmha_v2_il_int8_96_64_sm90_cu_cubin_len, "fmha_v2_il_int8_96_64_sm90_kernel_nl", 18432, 128, 16, true},
    {DATA_TYPE_INT8, 64, 64, kSM_90, cubin_fmha_v2_int8_64_64_sm90_cu_cubin,
        cubin_fmha_v2_int8_64_64_sm90_cu_cubin_len, "fmha_v2_int8_64_64_sm90_kernel", 40960, 128, 0, false},
    {DATA_TYPE_INT8, 64, 64, kSM_90, cubin_fmha_v2_il_int8_64_64_sm90_cu_cubin,
        cubin_fmha_v2_il_int8_64_64_sm90_cu_cubin_len, "fmha_v2_il_int8_64_64_sm90_kernel", 20480, 128, 0, true},
    {DATA_TYPE_FP16, 512, 64, kSM_90, cubin_fmha_v2_fp16_512_64_sm90_cu_cubin,
        cubin_fmha_v2_fp16_512_64_sm90_cu_cubin_len, "fmha_v2_fp16_512_64_sm90_kernel", 73728, 256, 0, false},
    {DATA_TYPE_FP16, 512, 64, kSM_90, cubin_fmha_v2_fp16_512_64_sm90_cu_cubin,
        cubin_fmha_v2_fp16_512_64_sm90_cu_cubin_len, "fmha_v2_fp16_512_64_sm90_kernel_nl", 73728, 256, 32, false},
    {DATA_TYPE_FP16, 384, 64, kSM_90, cubin_fmha_v2_fp16_384_64_sm90_cu_cubin,
        cubin_fmha_v2_fp16_384_64_sm90_cu_cubin_len, "fmha_v2_fp16_384_64_sm90_kernel", 65536, 256, 0, false},
    {DATA_TYPE_FP16, 384, 64, kSM_90, cubin_fmha_v2_fp16_384_64_sm90_cu_cubin,
        cubin_fmha_v2_fp16_384_64_sm90_cu_cubin_len, "fmha_v2_fp16_384_64_sm90_kernel_nl", 57344, 256, 32, false},
    {DATA_TYPE_FP16, 256, 64, kSM_90, cubin_fmha_v2_fp16_256_64_sm90_cu_cubin,
        cubin_fmha_v2_fp16_256_64_sm90_cu_cubin_len, "fmha_v2_fp16_256_64_sm90_kernel", 40960, 128, 0, false},
    {DATA_TYPE_FP16, 256, 64, kSM_90, cubin_fmha_v2_fp16_256_64_sm90_cu_cubin,
        cubin_fmha_v2_fp16_256_64_sm90_cu_cubin_len, "fmha_v2_fp16_256_64_sm90_kernel_nl", 40960, 128, 32, false},
    {DATA_TYPE_FP16, 128, 64, kSM_90, cubin_fmha_v2_fp16_128_64_sm90_cu_cubin,
        cubin_fmha_v2_fp16_128_64_sm90_cu_cubin_len, "fmha_v2_fp16_128_64_sm90_kernel", 65536, 128, 0, false},
    {DATA_TYPE_FP16, 128, 64, kSM_90, cubin_fmha_v2_fp16_128_64_sm90_cu_cubin,
        cubin_fmha_v2_fp16_128_64_sm90_cu_cubin_len, "fmha_v2_fp16_128_64_sm90_kernel_nl", 36864, 128, 16, false},
    {DATA_TYPE_FP16, 96, 64, kSM_90, cubin_fmha_v2_fp16_96_64_sm90_cu_cubin,
        cubin_fmha_v2_fp16_96_64_sm90_cu_cubin_len, "fmha_v2_fp16_96_64_sm90_kernel", 49152, 128, 0, false},
    {DATA_TYPE_FP16, 64, 64, kSM_90, cubin_fmha_v2_fp16_64_64_sm90_cu_cubin,
        cubin_fmha_v2_fp16_64_64_sm90_cu_cubin_len, "fmha_v2_fp16_64_64_sm90_kernel", 32768, 128, 0, false},
    {DATA_TYPE_FP16, 64, 64, kSM_90, cubin_fmha_v2_fp16_64_64_sm90_cu_cubin,
        cubin_fmha_v2_fp16_64_64_sm90_cu_cubin_len, "fmha_v2_fp16_64_64_sm90_kernel_nl", 20480, 128, 16, false},
#endif // defined(ENABLE_SM90)
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
        PLUGIN_ASSERT(headsize <= 0x3FFFFFFF);
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
            PLUGIN_ASSERT(mDataType == bert::DATA_TYPE_INT8);
        }

        bool forceUnroll = params.force_unroll;
        if (!forceUnroll && !params.ignore_b1opt && mSM >= kSM_75)
        {
            const struct
            {
                uint32_t mSM;
                Data_type mDataType;
                int32_t mS;
                int32_t mMaxBatch;
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
#if CUDA_VERSION >= 11040
                      {kSM_87, bert::DATA_TYPE_FP16, 128, 4},
                      {kSM_87, bert::DATA_TYPE_FP16, 256, 4},
                      {kSM_87, bert::DATA_TYPE_FP16, 384, 4},
                      {kSM_87, bert::DATA_TYPE_INT8, 128, 4},
                      {kSM_87, bert::DATA_TYPE_INT8, 192, 16},
                      {kSM_87, bert::DATA_TYPE_INT8, 256, 8},
                      {kSM_87, bert::DATA_TYPE_INT8, 384, 8},
#endif
#if CUDA_VERSION >= 11080
                      {kSM_90, bert::DATA_TYPE_FP16, 128, 4},
                      {kSM_90, bert::DATA_TYPE_FP16, 256, 4},
                      {kSM_90, bert::DATA_TYPE_FP16, 384, 4},
                      {kSM_90, bert::DATA_TYPE_INT8, 128, 4},
                      {kSM_90, bert::DATA_TYPE_INT8, 192, 16},
                      {kSM_90, bert::DATA_TYPE_INT8, 256, 8},
                      {kSM_90, bert::DATA_TYPE_INT8, 384, 8},
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
        // Provide debug information if the kernel is missing in the pool.
        std::stringstream errMsg;
        errMsg << "Could not find kernel for:\n"
               << "\t s: " << params.s << "\n"
               << "\t d: " << params.d << "\n"
               << "\t interleaved: " << params.interleaved << "\n"
               << "\t forceUnroll: " << forceUnroll << "\n"
               << "Was the plugin compiled on a compatible CUDA and SM version?\n"
               << "\t Compiled on CUDA " << CUDA_VERSION << "\n"
               << "\t Current SM version: " << mSM << "\n"
               << "\t SM versions enabled during compilation: "
#if defined(ENABLE_SM72)
               << "72 "
#endif
#if defined(ENABLE_SM75)
               << "75 "
#endif
#if defined(ENABLE_SM80)
               << "80 "
#endif
#if defined(ENABLE_SM86)
               << "86 "
#endif
#if defined(ENABLE_SM87)
               << "87 "
#endif
#if defined(ENABLE_SM90)
               << "90 "
#endif
               << "\n";
        PLUGIN_VALIDATE(findIter != mFunctions.end(), errMsg.str().c_str());

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
            int32_t unroll = kernelMeta.mS / kernelMeta.mUnrollStep;
            PLUGIN_ASSERT(kernelMeta.mS == kernelMeta.mUnrollStep * unroll);
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
} // namespace plugin
} // namespace nvinfer1
#endif // _BERT_FMHAV2_FMHAV2
