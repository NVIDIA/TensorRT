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

#ifndef _BERT_FMHA_FMHA
#define _BERT_FMHA_FMHA
#include "common/bertCommon.h"
#include "common/cudaDriverWrapper.h"
#include "common/plugin.h"
#include "cuda_runtime_api.h"
#include "fused_multihead_attention_common.h"
#include <memory>
#include <mutex>
#include <set>
#include <stdint.h>
#include <unordered_map>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
namespace bert
{
static inline size_t get_size_in_bytes(size_t n, Data_type dtype)
{
    switch (dtype)
    {
    case DATA_TYPE_E8M10: return n * 4;
    case DATA_TYPE_FP32: return n * 4;
    case DATA_TYPE_FP16: return n * 2;
    case DATA_TYPE_INT32: return n * 4;
    case DATA_TYPE_INT8: return n;
    case DATA_TYPE_INT4: return n / 2U;
    case DATA_TYPE_BOOL: return n / 8U;
    case DATA_TYPE_E8M7: return n * 2;
    default: PLUGIN_ASSERT(false); return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_attention_params
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
    int64_t o_stride_in_bytes{};

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
};

////////////////////////////////////////////////////////////////////////////////////////////////////
extern unsigned char fused_multihead_attention_fp16_64_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_fp16_96_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_fp16_64_64_kernel_sm80_cu_o[];
extern unsigned char fused_multihead_attention_fp16_96_64_kernel_sm80_cu_o[];
extern unsigned char fused_multihead_attention_fp16_128_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_fp16_384_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_int8_128_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_int8_384_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_int8_384_64_kernel_sm80_cu_o[];
extern unsigned char fused_multihead_attention_int8_128_64_kernel_sm80_cu_o[];
extern unsigned char fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o[];
extern unsigned char fused_multihead_attention_fp16_384_64_kernel_sm80_cu_o[];
extern unsigned char fused_multihead_attention_fp16_384_64_kernel_sm86_cu_o[];

extern unsigned char cubin_fmha_v1_int8_64_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v1_int8_96_64_sm80_cu_cubin[];

extern unsigned char cubin_fmha_v1_int8_384_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v1_int8_128_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v1_fp16_384_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v1_fp16_128_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v1_fp16_96_64_sm87_cu_cubin[];
extern unsigned char cubin_fmha_v1_fp16_64_64_sm87_cu_cubin[];

extern unsigned char cubin_fmha_v1_int8_512_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v1_int8_384_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v1_int8_128_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v1_fp16_512_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v1_fp16_384_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v1_fp16_128_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v1_fp16_96_64_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v1_fp16_64_64_sm90_cu_cubin[];

extern uint32_t fused_multihead_attention_fp16_64_64_kernel_sm75_cu_o_len;
extern uint32_t fused_multihead_attention_fp16_96_64_kernel_sm75_cu_o_len;
extern uint32_t fused_multihead_attention_fp16_64_64_kernel_sm80_cu_o_len;
extern uint32_t fused_multihead_attention_fp16_96_64_kernel_sm80_cu_o_len;
extern uint32_t fused_multihead_attention_fp16_128_64_kernel_sm75_cu_o_len;
extern uint32_t fused_multihead_attention_fp16_384_64_kernel_sm75_cu_o_len;
extern uint32_t fused_multihead_attention_int8_128_64_kernel_sm75_cu_o_len;
extern uint32_t fused_multihead_attention_int8_384_64_kernel_sm75_cu_o_len;
extern uint32_t fused_multihead_attention_int8_384_64_kernel_sm80_cu_o_len;
extern uint32_t fused_multihead_attention_int8_128_64_kernel_sm80_cu_o_len;
extern uint32_t fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o_len;
extern uint32_t fused_multihead_attention_fp16_384_64_kernel_sm80_cu_o_len;
extern uint32_t fused_multihead_attention_fp16_384_64_kernel_sm86_cu_o_len;

extern uint32_t cubin_fmha_v1_int8_64_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v1_int8_96_64_sm80_cu_cubin_len;

extern uint32_t cubin_fmha_v1_int8_384_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v1_int8_128_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v1_fp16_384_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v1_fp16_128_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v1_fp16_96_64_sm87_cu_cubin_len;
extern uint32_t cubin_fmha_v1_fp16_64_64_sm87_cu_cubin_len;

extern uint32_t cubin_fmha_v1_int8_512_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v1_int8_384_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v1_int8_128_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v1_fp16_512_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v1_fp16_384_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v1_fp16_128_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v1_fp16_96_64_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v1_fp16_64_64_sm90_cu_cubin_len;
#if !(defined(ENABLE_SM72) || defined(ENABLE_SM75) || defined(ENABLE_SM80) || defined(ENABLE_SM86)                     \
    || defined(ENABLE_SM87) || defined(ENABLE_SM89) || defined(ENABLE_SM90))
// TRT-17573: Remove SM72 support from this file by factoring out the common logic required by the
// V2 headers into a separate header.
#error This file can only be included one of sm 72, 75, 80, 86, 87, 89, or 90 are defined.
#endif
static const struct FusedMultiHeadAttentionKernelMetaInfoV1
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
} sMhaKernelMetaInfos[] = {
#if defined(ENABLE_SM75)
    // Turing
    {DATA_TYPE_FP16, 64, 64, kSM_75, fused_multihead_attention_fp16_64_64_kernel_sm75_cu_o,
        fused_multihead_attention_fp16_64_64_kernel_sm75_cu_o_len,
        "fused_multihead_attention_v2_fp16_64_64_kernel_sm75", 24576, 128},
    {DATA_TYPE_FP16, 96, 64, kSM_75, fused_multihead_attention_fp16_96_64_kernel_sm75_cu_o,
        fused_multihead_attention_fp16_96_64_kernel_sm75_cu_o_len,
        "fused_multihead_attention_v2_fp16_96_64_kernel_sm75", 24576, 128},
    {DATA_TYPE_FP16, 128, 64, kSM_75, fused_multihead_attention_fp16_128_64_kernel_sm75_cu_o,
        fused_multihead_attention_fp16_128_64_kernel_sm75_cu_o_len, "fused_multihead_attention_fp16_128_64_kernel_sm75",
        32768, 128},
    {DATA_TYPE_FP16, 384, 64, kSM_75, fused_multihead_attention_fp16_384_64_kernel_sm75_cu_o,
        fused_multihead_attention_fp16_384_64_kernel_sm75_cu_o_len, "fused_multihead_attention_fp16_384_64_kernel_sm75",
        57344, 256},
    {DATA_TYPE_INT8, 128, 64, kSM_75, fused_multihead_attention_int8_128_64_kernel_sm75_cu_o,
        fused_multihead_attention_int8_128_64_kernel_sm75_cu_o_len, "fused_multihead_attention_int8_128_64_kernel_sm75",
        16384, 128},
    {DATA_TYPE_INT8, 384, 64, kSM_75, fused_multihead_attention_int8_384_64_kernel_sm75_cu_o,
        fused_multihead_attention_int8_384_64_kernel_sm75_cu_o_len, "fused_multihead_attention_int8_384_64_kernel_sm75",
        53284, 256},
#endif // defined(ENABLE_SM75)
#if defined(ENABLE_SM80) || defined(ENABLE_SM86) || defined(ENABLE_SM89)
    // Ampere
    {DATA_TYPE_FP16, 64, 64, kSM_80, fused_multihead_attention_fp16_64_64_kernel_sm80_cu_o,
        fused_multihead_attention_fp16_64_64_kernel_sm80_cu_o_len,
        "fused_multihead_attention_v2_fp16_64_64_kernel_sm80", 32768, 128},
    {DATA_TYPE_FP16, 96, 64, kSM_80, fused_multihead_attention_fp16_96_64_kernel_sm80_cu_o,
        fused_multihead_attention_fp16_96_64_kernel_sm80_cu_o_len,
        "fused_multihead_attention_v2_fp16_96_64_kernel_sm80", 49152, 128},
    {DATA_TYPE_FP16, 128, 64, kSM_80, fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o,
        fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o_len, "fused_multihead_attention_fp16_128_64_kernel_sm80",
        49152, 128},
    {DATA_TYPE_FP16, 384, 64, kSM_80, fused_multihead_attention_fp16_384_64_kernel_sm80_cu_o,
        fused_multihead_attention_fp16_384_64_kernel_sm80_cu_o_len, "fused_multihead_attention_fp16_384_64_kernel_sm80",
        114688, 256},
    {DATA_TYPE_INT8, 64, 64, kSM_80, cubin_fmha_v1_int8_64_64_sm80_cu_cubin, cubin_fmha_v1_int8_64_64_sm80_cu_cubin_len,
        "fmha_v1_int8_64_64_sm80_kernel", 24576, 128},
    {DATA_TYPE_INT8, 96, 64, kSM_80, cubin_fmha_v1_int8_96_64_sm80_cu_cubin, cubin_fmha_v1_int8_96_64_sm80_cu_cubin_len,
        "fmha_v1_int8_96_64_sm80_kernel", 28672, 128},
    {DATA_TYPE_INT8, 128, 64, kSM_80, fused_multihead_attention_int8_128_64_kernel_sm80_cu_o,
        fused_multihead_attention_int8_128_64_kernel_sm80_cu_o_len, "fused_multihead_attention_int8_128_64_kernel_sm80",
        24576, 128},
    {DATA_TYPE_INT8, 384, 64, kSM_80, fused_multihead_attention_int8_384_64_kernel_sm80_cu_o,
        fused_multihead_attention_int8_384_64_kernel_sm80_cu_o_len, "fused_multihead_attention_int8_384_64_kernel_sm80",
        57344, 256},
#endif // defined(ENABLE_SM80) || defined(SM86) || defined(ENABLE_SM89)
#if defined(ENABLE_SM86)
    // GA10x
    // Note: For GA10X keep only kernels whose sharedMemBytes < 100KiB
    {DATA_TYPE_FP16, 64, 64, kSM_86, fused_multihead_attention_fp16_64_64_kernel_sm80_cu_o,
        fused_multihead_attention_fp16_64_64_kernel_sm80_cu_o_len,
        "fused_multihead_attention_v2_fp16_64_64_kernel_sm80", 32768, 128},
    {DATA_TYPE_FP16, 96, 64, kSM_86, fused_multihead_attention_fp16_96_64_kernel_sm80_cu_o,
        fused_multihead_attention_fp16_96_64_kernel_sm80_cu_o_len,
        "fused_multihead_attention_v2_fp16_96_64_kernel_sm80", 49152, 128},
    {DATA_TYPE_FP16, 128, 64, kSM_86, fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o,
        fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o_len, "fused_multihead_attention_fp16_128_64_kernel_sm80",
        49152, 128},
    {DATA_TYPE_FP16, 384, 64, kSM_86, fused_multihead_attention_fp16_384_64_kernel_sm86_cu_o,
        fused_multihead_attention_fp16_384_64_kernel_sm86_cu_o_len, "fused_multihead_attention_fp16_384_64_kernel_sm80",
        65536, 256},
    {DATA_TYPE_INT8, 128, 64, kSM_86, fused_multihead_attention_int8_128_64_kernel_sm80_cu_o,
        fused_multihead_attention_int8_128_64_kernel_sm80_cu_o_len, "fused_multihead_attention_int8_128_64_kernel_sm80",
        24576, 128},
    {DATA_TYPE_INT8, 384, 64, kSM_86, fused_multihead_attention_int8_384_64_kernel_sm80_cu_o,
        fused_multihead_attention_int8_384_64_kernel_sm80_cu_o_len, "fused_multihead_attention_int8_384_64_kernel_sm80",
        57344, 256},
#endif // defined(ENABLE_SM86)
#if defined(ENABLE_SM87)
    // GA10b (Orin-Auto)
    {DATA_TYPE_INT8, 384, 64, kSM_87, cubin_fmha_v1_int8_384_64_sm87_cu_cubin,
        cubin_fmha_v1_int8_384_64_sm87_cu_cubin_len, "fmha_v1_int8_384_64_sm87_kernel", 40960, 256},
    {DATA_TYPE_INT8, 128, 64, kSM_87, cubin_fmha_v1_int8_128_64_sm87_cu_cubin,
        cubin_fmha_v1_int8_128_64_sm87_cu_cubin_len, "fmha_v1_int8_128_64_sm87_kernel", 24576, 128},
    {DATA_TYPE_FP16, 384, 64, kSM_87, cubin_fmha_v1_fp16_384_64_sm87_cu_cubin,
        cubin_fmha_v1_fp16_384_64_sm87_cu_cubin_len, "fmha_v1_fp16_384_64_sm87_kernel", 65536, 256},
    {DATA_TYPE_FP16, 128, 64, kSM_87, cubin_fmha_v1_fp16_128_64_sm87_cu_cubin,
        cubin_fmha_v1_fp16_128_64_sm87_cu_cubin_len, "fmha_v1_fp16_128_64_sm87_kernel", 49152, 128},
    {DATA_TYPE_FP16, 96, 64, kSM_87, cubin_fmha_v1_fp16_96_64_sm87_cu_cubin, cubin_fmha_v1_fp16_96_64_sm87_cu_cubin_len,
        "fmha_v1_fp16_96_64_sm87_kernel", 49152, 128},
    {DATA_TYPE_FP16, 64, 64, kSM_87, cubin_fmha_v1_fp16_64_64_sm87_cu_cubin, cubin_fmha_v1_fp16_64_64_sm87_cu_cubin_len,
        "fmha_v1_fp16_64_64_sm87_kernel", 32768, 128},
#endif // defined(ENABLE_SM87)
#if defined(ENABLE_SM90)
    // GH100 hopper
    {DATA_TYPE_INT8, 512, 64, kSM_90, cubin_fmha_v1_int8_512_64_sm90_cu_cubin,
        cubin_fmha_v1_int8_512_64_sm90_cu_cubin_len, "fmha_v1_int8_512_64_sm90_kernel", 73728, 256},
    {DATA_TYPE_INT8, 384, 64, kSM_90, cubin_fmha_v1_int8_384_64_sm90_cu_cubin,
        cubin_fmha_v1_int8_384_64_sm90_cu_cubin_len, "fmha_v1_int8_384_64_sm90_kernel", 73728, 256},
    {DATA_TYPE_INT8, 128, 64, kSM_90, cubin_fmha_v1_int8_128_64_sm90_cu_cubin,
        cubin_fmha_v1_int8_128_64_sm90_cu_cubin_len, "fmha_v1_int8_128_64_sm90_kernel", 24576, 128},
    {DATA_TYPE_FP16, 512, 64, kSM_90, cubin_fmha_v1_fp16_512_64_sm90_cu_cubin,
        cubin_fmha_v1_fp16_512_64_sm90_cu_cubin_len, "fmha_v1_fp16_512_64_sm90_kernel", 73728, 256},
    {DATA_TYPE_FP16, 384, 64, kSM_90, cubin_fmha_v1_fp16_384_64_sm90_cu_cubin,
        cubin_fmha_v1_fp16_384_64_sm90_cu_cubin_len, "fmha_v1_fp16_384_64_sm90_kernel", 65536, 256},
    {DATA_TYPE_FP16, 128, 64, kSM_90, cubin_fmha_v1_fp16_128_64_sm90_cu_cubin,
        cubin_fmha_v1_fp16_128_64_sm90_cu_cubin_len, "fmha_v1_fp16_128_64_sm90_kernel", 65536, 128},
    {DATA_TYPE_FP16, 96, 64, kSM_90, cubin_fmha_v1_fp16_96_64_sm90_cu_cubin, cubin_fmha_v1_fp16_96_64_sm90_cu_cubin_len,
        "fmha_v1_fp16_96_64_sm90_kernel", 49152, 128},
    {DATA_TYPE_FP16, 64, 64, kSM_90, cubin_fmha_v1_fp16_64_64_sm90_cu_cubin, cubin_fmha_v1_fp16_64_64_sm90_cu_cubin_len,
        "fmha_v1_fp16_64_64_sm90_kernel", 32768, 128},
#endif // defined(ENABLE_SM90)
};

template <typename TKernelMeta, typename TKernelParam>
class TFusedMultiHeadAttentionXMMAKernel
{
public:
    using KernelMeta = TKernelMeta;
    using KernelParam = TKernelParam;
    inline uint64_t hashID(uint32_t s, uint32_t d) const
    {
        return (uint64_t) s << 32 | d;
    }
    virtual uint64_t hashID(const KernelMeta& kernelMeta) const
    {
        return hashID(kernelMeta.mS, kernelMeta.mD);
    }

    TFusedMultiHeadAttentionXMMAKernel(const TKernelMeta* pMetaStart, uint32_t nMetaCount, Data_type type, uint32_t sm)
        : mDataType(type)
        , mKernelMeta(pMetaStart)
        , mKernelMetaCount(nMetaCount)
        , mSM(sm)
    {
        PLUGIN_ASSERT(mKernelMetaCount && "No kernels were loaded correctly.");
    }

    void loadXMMAKernels(uint32_t smVersion)
    {
        for (uint32_t i = 0; i < mKernelMetaCount; ++i)
        {
            const auto& kernelMeta = mKernelMeta[i];
            const auto kernelKey = hashID(kernelMeta);
            if (kernelMeta.mSM == smVersion && kernelMeta.mDataType == mDataType
                && mFunctions.find(kernelKey) == mFunctions.end())
            {
                const uint32_t DEFAULT_SMEM_SIZE{48 * 1024};
                if (kernelMeta.mSharedMemBytes >= DEFAULT_SMEM_SIZE)
                {
                    int32_t deviceID{0};
                    cudaGetDevice(&deviceID);
                    int32_t sharedMemPerMultiprocessor{0};
                    if (cudaDeviceGetAttribute(
                            &sharedMemPerMultiprocessor, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceID)
                            != cudaSuccess
                        || sharedMemPerMultiprocessor < static_cast<int32_t>(kernelMeta.mSharedMemBytes))
                    {
                        // skip load function because not enough shared memory to launch the kernel
                        continue;
                    }
                }

                CUmodule hmod{0};
                auto findModuleIter = mModules.find(kernelMeta.mCubin);
                if (findModuleIter != mModules.end())
                {
                    hmod = findModuleIter->second;
                }
                else
                {
                    cuErrCheck(mDriver.cuModuleLoadData(&hmod, kernelMeta.mCubin), mDriver);
                    mModules.insert(std::make_pair(kernelMeta.mCubin, hmod));
                }

                FusedMultiHeadAttentionKernelInfo funcInfo;
                funcInfo.mMetaInfoIndex = i;
                cuErrCheck(mDriver.cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod, kernelMeta.mFuncName), mDriver);
                if (kernelMeta.mSharedMemBytes >= DEFAULT_SMEM_SIZE)
                {
                    if (mDriver.cuFuncSetAttribute(funcInfo.mDeviceFunction,
                            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernelMeta.mSharedMemBytes)
                        != CUDA_SUCCESS)
                    {
                        // some chip may not have enough shared memory to launch the kernel
                        continue;
                    }
                }
                mFunctions.insert({kernelKey, funcInfo});
                const int s = static_cast<int>(kernelMeta.mS);
                if (mValidSequences.find(s) == mValidSequences.end())
                {
                    mValidSequences.insert(s);
                }
            }
        }
    }

    void loadXMMAKernels()
    {
        if (!mFunctions.empty())
        {
            return;
        }

        loadXMMAKernels(mSM);

        // sm_86 chips prefer sm_86 sass, but can also use sm_80 sass if sm_86 not exist.
        // sm_87 cannot run sm_80 sass
        // sm_89 will reuse sm_80 kernels
        if (mSM == kSM_86 || mSM == kSM_89)
        {
            loadXMMAKernels(kSM_80);
        }
    }

    bool isValid(int s) const
    {
        return (mValidSequences.find(s) != mValidSequences.end());
    }

    virtual void run(TKernelParam& params, cudaStream_t ss) const
    {
        const auto findIter = mFunctions.find(hashID(params.s, params.d));
        std::stringstream errMsg;
        errMsg << "Could not find kernel for:\n"
               << "\t s: " << params.s << "\n"
               << "\t d: " << params.d << "\n"
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
        cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, 1, kernelMeta.mThreadsPerCTA, 1, 1,
                       kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
            mDriver);
    }

    virtual ~TFusedMultiHeadAttentionXMMAKernel() = default;

protected:
    nvinfer1::CUDADriverWrapper mDriver;

    Data_type mDataType;
    const TKernelMeta* mKernelMeta;
    uint32_t mKernelMetaCount;
    uint32_t mSM;
    std::unordered_map<const unsigned char*, CUmodule> mModules;
    struct FusedMultiHeadAttentionKernelInfo
    {
        uint32_t mMetaInfoIndex;
        CUfunction mDeviceFunction;
    };
    std::unordered_map<uint64_t, FusedMultiHeadAttentionKernelInfo> mFunctions;
    std::set<int> mValidSequences;
};

template <typename TFusedMHAKernelList>
class TFusedMHAKernelFactory
{
public:
    const TFusedMHAKernelList* getXMMAKernels(
        const typename TFusedMHAKernelList::KernelMeta* pKernelList, uint32_t nbKernels, Data_type type, uint32_t sm)
    {
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lg(s_mutex);

        const auto id = hashID(type, sm);
        const auto findIter = mKernels.find(id);
        if (findIter == mKernels.end())
        {
            TFusedMHAKernelList* newKernel = new TFusedMHAKernelList{pKernelList, nbKernels, type, sm};
            newKernel->loadXMMAKernels();
            mKernels.insert(std::make_pair(id, std::unique_ptr<TFusedMHAKernelList>(newKernel)));
            return newKernel;
        }
        return findIter->second.get();
    }

    static TFusedMHAKernelFactory<TFusedMHAKernelList>& Get()
    {
        static TFusedMHAKernelFactory<TFusedMHAKernelList> s_factory;
        return s_factory;
    }

private:
    TFusedMHAKernelFactory() = default;

    inline uint64_t hashID(Data_type type, uint32_t sm) const
    {
        // use deviceID in hasID for multi GPU support before driver support context-less loading of cubin
        int32_t deviceID{0};
        CSC(cudaGetDevice(&deviceID), STATUS_FAILURE);

        PLUGIN_ASSERT((deviceID & 0xFFFF) == deviceID);
        PLUGIN_ASSERT((type & 0xFFFF) == type);
        PLUGIN_ASSERT((sm & 0xFFFFFFFF) == sm);
        return (uint64_t) type << 48 | (uint64_t) deviceID << 32 | sm;
    }

    std::unordered_map<uint64_t, const std::unique_ptr<TFusedMHAKernelList>> mKernels;
};

using FusedMultiHeadAttentionXMMAKernel
    = TFusedMultiHeadAttentionXMMAKernel<FusedMultiHeadAttentionKernelMetaInfoV1, Fused_multihead_attention_params>;
using FusedMHAKernelFactory = TFusedMHAKernelFactory<FusedMultiHeadAttentionXMMAKernel>;

inline const FusedMultiHeadAttentionXMMAKernel* getXMMAKernels(Data_type type, uint32_t sm)
{
    return FusedMHAKernelFactory::Get().getXMMAKernels(
        sMhaKernelMetaInfos, sizeof(sMhaKernelMetaInfos) / sizeof(sMhaKernelMetaInfos[0]), type, sm);
}

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // _BERT_FMHA_FMHA
