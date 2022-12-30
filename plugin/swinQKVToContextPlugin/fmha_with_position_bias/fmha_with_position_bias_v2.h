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

#ifndef TRT_FMHA_WITH_POSITION_BIAS_V2_H
#define TRT_FMHA_WITH_POSITION_BIAS_V2_H
#include "common/bertCommon.h"
#include "common/cudaDriverWrapper.h"
#include "fmha_with_position_bias_common.h"
#include <cassert>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
struct FusedMHAParams
{
    // The QKV matrices.
    void* qkvPtr{};
    // The mask to implement drop-out.
    void* packedMaskPtr{};

    // The O matrix (output).
    void* oPtr{};

    // The stride between rows of the Q, K and V matrices.
    int64_t qkvStrideInBytes{};
    // The stride between matrices of packed mask.
    int64_t packedMaskStrideInBytes{};
    // The stride between rows of O.
    int64_t oStrideInBytes{};

#if defined(STORE_P)
    // The pointer to the P matrix (for debugging).
    void* pPtr{};
    // The stride between rows of the P matrix (for debugging).
    int64_t pStrideInBytes{};
#endif // defined(STORE_P)

#if defined(STORE_S)
    // The pointer to the S matrix (for debugging).
    void* sPtr{};
    // The stride between rows of the S matrix (for debugging).
    int64_t sStrideInBytes{};
#endif // defined(STORE_S)

    // The dimensions.
    int32_t b{};
    int32_t h{};
    int32_t s{};
    int32_t d{};

    // The scaling factors for the kernel.
    uint32_t scaleQK{};
    uint32_t scaleSoftmax{};
    uint32_t scaleVAttn{};

    // A trick to avoid I2F/F2I in the INT8 kernel.
    bool enableI2fTrick{};

    // array of length b+1 holding prefix sum of actual sequence lengths
    int32_t* cuSeqlens{};

    // use C/32 Format.
    bool interleaved{};
    bool ignoreB1opt{};
    bool forceUnroll{};
    bool useInt8ScaleMax{};

    // The additional parameters for fused_mha_with_relPosBias kernels
    // The relative position bias.
    void* packedRelativePositionBiasPtr{};

    int32_t windowNum{};
    int32_t actualSeqlen{};

    void clear()
    {
        *this = FusedMHAParams{};
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Do not modify this, it is integrated from generated/fmha_cubin.h in fmha.
////////////////////////////////////////////////////////////////////////////////////////////////////
extern unsigned char fused_mha_with_relPosBias_fp16_64_32_kernel_sm75_cubin[];
extern unsigned char fused_mha_with_relPosBias_fp16_64_32_kernel_sm80_cubin[];
extern unsigned char fused_mha_with_relPosBias_fp16_64_32_kernel_sm86_cubin[];
extern unsigned char fused_mha_with_relPosBias_fp16_128_32_kernel_sm75_cubin[];
extern unsigned char fused_mha_with_relPosBias_fp16_128_32_kernel_sm80_cubin[];
extern unsigned char fused_mha_with_relPosBias_fp16_128_32_kernel_sm86_cubin[];
extern unsigned char fused_mha_with_relPosBias_fp16_256_32_kernel_sm75_cubin[];
extern unsigned char fused_mha_with_relPosBias_fp16_256_32_kernel_sm80_cubin[];
extern unsigned char fused_mha_with_relPosBias_fp16_256_32_kernel_sm86_cubin[];

extern unsigned char fused_mha_with_relPosBias_int8_64_32_kernel_sm75_cubin[];
extern unsigned char fused_mha_with_relPosBias_int8_64_32_kernel_sm80_cubin[];
extern unsigned char fused_mha_with_relPosBias_int8_256_32_kernel_sm75_cubin[];
extern unsigned char fused_mha_with_relPosBias_int8_256_32_kernel_sm80_cubin[];

extern uint32_t fused_mha_with_relPosBias_fp16_64_32_kernel_sm75_cubin_len;
extern uint32_t fused_mha_with_relPosBias_fp16_64_32_kernel_sm80_cubin_len;
extern uint32_t fused_mha_with_relPosBias_fp16_64_32_kernel_sm86_cubin_len;
extern uint32_t fused_mha_with_relPosBias_fp16_128_32_kernel_sm75_cubin_len;
extern uint32_t fused_mha_with_relPosBias_fp16_128_32_kernel_sm80_cubin_len;
extern uint32_t fused_mha_with_relPosBias_fp16_128_32_kernel_sm86_cubin_len;
extern uint32_t fused_mha_with_relPosBias_fp16_256_32_kernel_sm75_cubin_len;
extern uint32_t fused_mha_with_relPosBias_fp16_256_32_kernel_sm80_cubin_len;
extern uint32_t fused_mha_with_relPosBias_fp16_256_32_kernel_sm86_cubin_len;

extern uint32_t fused_mha_with_relPosBias_int8_64_32_kernel_sm75_cubin_len;
extern uint32_t fused_mha_with_relPosBias_int8_64_32_kernel_sm80_cubin_len;
extern uint32_t fused_mha_with_relPosBias_int8_256_32_kernel_sm75_cubin_len;
extern uint32_t fused_mha_with_relPosBias_int8_256_32_kernel_sm80_cubin_len;

static struct FusedMHAKernelMetaInfo
{
    DataType mDataType{};
    int32_t mS{};
    int32_t mD{};
    int32_t mSM{};
    unsigned char const* mCubin{};
    uint32_t mCubinSize{};
    char const* mFuncName{};
    int32_t mSharedMemBytes{};
    int32_t mThreadsPerCTA{};
    int32_t mUnrollStep{};
    bool mInterleaved{};
} sMhaKernelMetaInfosV2[] = {
#if defined(ENABLE_SM75)
    // Turing
    {DATA_TYPE_FP16, 64, 32, kSM_75, fused_mha_with_relPosBias_fp16_64_32_kernel_sm75_cubin,
        fused_mha_with_relPosBias_fp16_64_32_kernel_sm75_cubin_len, "fused_mha_with_relPosBias_fp16_64_32_kernel_sm75",
        12288, 128, 0, false},
    {DATA_TYPE_FP16, 64, 32, kSM_75, fused_mha_with_relPosBias_fp16_64_32_kernel_sm75_cubin,
        fused_mha_with_relPosBias_fp16_64_32_kernel_sm75_cubin_len,
        "fused_mha_with_relPosBias_fp16_64_32_kernel_sm75_noloop", 10240, 128, 32, false},
    {DATA_TYPE_FP16, 128, 32, kSM_75, fused_mha_with_relPosBias_fp16_128_32_kernel_sm75_cubin,
        fused_mha_with_relPosBias_fp16_128_32_kernel_sm75_cubin_len,
        "fused_mha_with_relPosBias_fp16_128_32_kernel_sm75", 16384, 128, 0, false},
    {DATA_TYPE_FP16, 128, 32, kSM_75, fused_mha_with_relPosBias_fp16_128_32_kernel_sm75_cubin,
        fused_mha_with_relPosBias_fp16_128_32_kernel_sm75_cubin_len,
        "fused_mha_with_relPosBias_fp16_128_32_kernel_sm75_noloop", 10240, 128, 32, false},
    {DATA_TYPE_FP16, 256, 32, kSM_75, fused_mha_with_relPosBias_fp16_256_32_kernel_sm75_cubin,
        fused_mha_with_relPosBias_fp16_256_32_kernel_sm75_cubin_len,
        "fused_mha_with_relPosBias_fp16_256_32_kernel_sm75", 18432, 128, 0, false},
    {DATA_TYPE_FP16, 256, 32, kSM_75, fused_mha_with_relPosBias_fp16_256_32_kernel_sm75_cubin,
        fused_mha_with_relPosBias_fp16_256_32_kernel_sm75_cubin_len,
        "fused_mha_with_relPosBias_fp16_256_32_kernel_sm75_noloop", 18432, 128, 32, false},

    {DATA_TYPE_INT8, 64, 32, kSM_75, fused_mha_with_relPosBias_int8_64_32_kernel_sm75_cubin,
        fused_mha_with_relPosBias_int8_64_32_kernel_sm75_cubin_len, "fused_mha_with_relPosBias_int8_64_32_kernel_sm75",
        10240, 128, 0, false},
    {DATA_TYPE_INT8, 256, 32, kSM_75, fused_mha_with_relPosBias_int8_256_32_kernel_sm75_cubin,
        fused_mha_with_relPosBias_int8_256_32_kernel_sm75_cubin_len,
        "fused_mha_with_relPosBias_int8_256_32_kernel_sm75", 10240, 128, 0, false},
#endif // defined(ENABLE_SM75)

#if defined(ENABLE_SM80) || defined(ENABLE_SM86)
    // Ampere
    {DATA_TYPE_FP16, 64, 32, kSM_80, fused_mha_with_relPosBias_fp16_64_32_kernel_sm80_cubin,
        fused_mha_with_relPosBias_fp16_64_32_kernel_sm80_cubin_len, "fused_mha_with_relPosBias_fp16_64_32_kernel_sm80",
        12288, 128, 0, false},
    {DATA_TYPE_FP16, 64, 32, kSM_80, fused_mha_with_relPosBias_fp16_64_32_kernel_sm80_cubin,
        fused_mha_with_relPosBias_fp16_64_32_kernel_sm80_cubin_len,
        "fused_mha_with_relPosBias_fp16_64_32_kernel_sm80_noloop", 10240, 128, 32, false},
    {DATA_TYPE_FP16, 128, 32, kSM_80, fused_mha_with_relPosBias_fp16_128_32_kernel_sm80_cubin,
        fused_mha_with_relPosBias_fp16_128_32_kernel_sm80_cubin_len,
        "fused_mha_with_relPosBias_fp16_128_32_kernel_sm80", 16384, 128, 0, false},
    {DATA_TYPE_FP16, 128, 32, kSM_80, fused_mha_with_relPosBias_fp16_128_32_kernel_sm80_cubin,
        fused_mha_with_relPosBias_fp16_128_32_kernel_sm80_cubin_len,
        "fused_mha_with_relPosBias_fp16_128_32_kernel_sm80_noloop", 10240, 128, 32, false},
    {DATA_TYPE_FP16, 256, 32, kSM_80, fused_mha_with_relPosBias_fp16_256_32_kernel_sm80_cubin,
        fused_mha_with_relPosBias_fp16_256_32_kernel_sm80_cubin_len,
        "fused_mha_with_relPosBias_fp16_256_32_kernel_sm80", 18432, 128, 0, false},
    {DATA_TYPE_FP16, 256, 32, kSM_80, fused_mha_with_relPosBias_fp16_256_32_kernel_sm80_cubin,
        fused_mha_with_relPosBias_fp16_256_32_kernel_sm80_cubin_len,
        "fused_mha_with_relPosBias_fp16_256_32_kernel_sm80_noloop", 18432, 128, 32, false},

    {DATA_TYPE_INT8, 64, 32, kSM_80, fused_mha_with_relPosBias_int8_64_32_kernel_sm80_cubin,
        fused_mha_with_relPosBias_int8_64_32_kernel_sm80_cubin_len, "fused_mha_with_relPosBias_int8_64_32_kernel_sm80",
        10240, 128, 0, false},
    {DATA_TYPE_INT8, 256, 32, kSM_80, fused_mha_with_relPosBias_int8_256_32_kernel_sm80_cubin,
        fused_mha_with_relPosBias_int8_256_32_kernel_sm80_cubin_len,
        "fused_mha_with_relPosBias_int8_256_32_kernel_sm80", 10240, 128, 0, false},
#endif // defined(ENABLE_SM80) || defined(ENABLE_SM86)

#if defined(ENABLE_SM86)
    // GA10x
    // Note: For GA10X keep only kernels whose sharedMemBytes < 100KiB
    {DATA_TYPE_FP16, 64, 32, kSM_86, fused_mha_with_relPosBias_fp16_64_32_kernel_sm86_cubin,
        fused_mha_with_relPosBias_fp16_64_32_kernel_sm86_cubin_len, "fused_mha_with_relPosBias_fp16_64_32_kernel_sm86",
        12288, 128, 0, false},
    {DATA_TYPE_FP16, 64, 32, kSM_86, fused_mha_with_relPosBias_fp16_64_32_kernel_sm86_cubin,
        fused_mha_with_relPosBias_fp16_64_32_kernel_sm86_cubin_len,
        "fused_mha_with_relPosBias_fp16_64_32_kernel_sm86_noloop", 10240, 128, 32, false},
    {DATA_TYPE_FP16, 128, 32, kSM_86, fused_mha_with_relPosBias_fp16_128_32_kernel_sm86_cubin,
        fused_mha_with_relPosBias_fp16_128_32_kernel_sm86_cubin_len,
        "fused_mha_with_relPosBias_fp16_128_32_kernel_sm86", 16384, 128, 0, false},
    {DATA_TYPE_FP16, 128, 32, kSM_86, fused_mha_with_relPosBias_fp16_128_32_kernel_sm86_cubin,
        fused_mha_with_relPosBias_fp16_128_32_kernel_sm86_cubin_len,
        "fused_mha_with_relPosBias_fp16_128_32_kernel_sm86_noloop", 10240, 128, 32, false},
    {DATA_TYPE_FP16, 256, 32, kSM_86, fused_mha_with_relPosBias_fp16_256_32_kernel_sm86_cubin,
        fused_mha_with_relPosBias_fp16_256_32_kernel_sm86_cubin_len,
        "fused_mha_with_relPosBias_fp16_256_32_kernel_sm86", 18432, 128, 0, false},
    {DATA_TYPE_FP16, 256, 32, kSM_86, fused_mha_with_relPosBias_fp16_256_32_kernel_sm86_cubin,
        fused_mha_with_relPosBias_fp16_256_32_kernel_sm86_cubin_len,
        "fused_mha_with_relPosBias_fp16_256_32_kernel_sm86_noloop", 18432, 128, 32, false},

    {DATA_TYPE_INT8, 64, 32, kSM_86, fused_mha_with_relPosBias_int8_64_32_kernel_sm80_cubin,
        fused_mha_with_relPosBias_int8_64_32_kernel_sm80_cubin_len, "fused_mha_with_relPosBias_int8_64_32_kernel_sm80",
        10240, 128, 0, false},
    {DATA_TYPE_INT8, 256, 32, kSM_86, fused_mha_with_relPosBias_int8_256_32_kernel_sm80_cubin,
        fused_mha_with_relPosBias_int8_256_32_kernel_sm80_cubin_len,
        "fused_mha_with_relPosBias_int8_256_32_kernel_sm80", 10240, 128, 0, false},
#endif // defined(ENABLE_SM86)
};

template <typename TKernelMeta, typename TKernelParam>
class TFusedMultiHeadAttentionXMMAKernel
{
public:
    using KernelMeta = TKernelMeta;
    using KernelParam = TKernelParam;
    uint64_t hashID(int32_t s, int32_t d) const
    {
        return (static_cast<uint64_t>(s) << 32) | d;
    }

    virtual uint64_t hashID(KernelMeta const& kernelMeta) const
    {
        return hashID(kernelMeta.mS, kernelMeta.mD);
    }

    TFusedMultiHeadAttentionXMMAKernel(TKernelMeta const* pMetaStart, int32_t nMetaCount, DataType type, int32_t sm)
        : mDataType(type)
        , mKernelMeta(pMetaStart)
        , mKernelMetaCount(nMetaCount)
        , mSM(sm)
    {
    }

    void loadXMMAKernels()
    {
        if (!mFunctions.empty())
        {
            return;
        }

        for (int32_t i = 0; i < mKernelMetaCount; ++i)
        {
            auto const& kernelMeta = mKernelMeta[i];
            if (kernelMeta.mSM == mSM && kernelMeta.mDataType == mDataType)
            {
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
                if (kernelMeta.mSharedMemBytes >= 48 * 1024)
                {
                    cuErrCheck(mDriver.cuFuncSetAttribute(funcInfo.mDeviceFunction,
                                   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernelMeta.mSharedMemBytes),
                        mDriver);
                }
                mFunctions.insert(std::make_pair(hashID(kernelMeta), funcInfo));
                auto s = kernelMeta.mS;
                if (mValidSequences.find(s) == mValidSequences.end())
                {
                    mValidSequences.insert(s);
                }
            }
        }
    }

    bool isValid(int32_t s) const
    {
        return mValidSequences.find(s) != mValidSequences.end();
    }

    virtual void run(TKernelParam& params, cudaStream_t ss) const
    {
        auto const findIter = mFunctions.find(hashID(params.s, params.d));
        PLUGIN_ASSERT(findIter != mFunctions.end());

        auto const& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        CUfunction func = findIter->second.mDeviceFunction;

        void* kernelParams[] = {&params, nullptr};
        cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, 1, kernelMeta.mThreadsPerCTA, 1, 1,
                       kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
            mDriver);
    }

    virtual ~TFusedMultiHeadAttentionXMMAKernel() = default;

protected:
    CUDADriverWrapper mDriver{};

    DataType mDataType{};
    TKernelMeta const* mKernelMeta{};
    int32_t mKernelMetaCount{};
    int32_t mSM{};
    std::unordered_map<uint8_t const*, CUmodule> mModules{};
    struct FusedMultiHeadAttentionKernelInfo
    {
        int32_t mMetaInfoIndex{};
        CUfunction mDeviceFunction{};
    };
    std::unordered_map<uint64_t, FusedMultiHeadAttentionKernelInfo> mFunctions{};
    std::set<int32_t> mValidSequences{};
};

using FusedMultiHeadAttentionXMMAKernel = TFusedMultiHeadAttentionXMMAKernel<FusedMHAKernelMetaInfo, FusedMHAParams>;

class FusedMultiHeadAttentionXMMAKernelV2 : public FusedMultiHeadAttentionXMMAKernel
{
public:
    FusedMultiHeadAttentionXMMAKernelV2(
        FusedMHAKernelMetaInfo const* pMetaStart, int32_t nMetaCount, DataType type, int32_t sm)
        : TFusedMultiHeadAttentionXMMAKernel<FusedMHAKernelMetaInfo, FusedMHAParams>(pMetaStart, nMetaCount, type, sm)
    {
    }

    uint64_t hashID(int32_t s, int32_t d, bool interleaved, bool unroll) const
    {
        return static_cast<uint64_t>(s) << 32 | d | (interleaved ? 2U : 0U) | (unroll ? 1U : 0U);
    }

    virtual uint64_t hashID(KernelMeta const& kernelMeta) const
    {
        PLUGIN_ASSERT(kernelMeta.mD == 32);
        return hashID(kernelMeta.mS, kernelMeta.mD, kernelMeta.mInterleaved, kernelMeta.mUnrollStep);
    }
    virtual void run(FusedMHAParams& params, cudaStream_t ss) const
    {
        PLUGIN_ASSERT(params.d == 32);
        if (params.interleaved)
        {
            PLUGIN_ASSERT(mDataType == DATA_TYPE_INT8);
        }

        bool forceUnroll = params.forceUnroll;
        if (!forceUnroll && !params.ignoreB1opt && mSM >= kSM_75)
        {
            const struct
            {
                int32_t mSM;
                DataType mDataType;
                int32_t mS;
                int32_t mD;
                int32_t mMaxBatch;
            } unrollList[]
                = { {kSM_75, DATA_TYPE_FP16, 256, 64, 1},
                      {kSM_75, DATA_TYPE_FP16, 384, 64, 1},
                      {kSM_75, DATA_TYPE_INT8, 128, 64, 1},
                      {kSM_75, DATA_TYPE_INT8, 192, 64, 2},
                      {kSM_75, DATA_TYPE_INT8, 256, 64, 1},
                      {kSM_75, DATA_TYPE_INT8, 384, 64, 1},

                      {kSM_75, DATA_TYPE_FP16, 256, 32, 4},
#if CUDA_VERSION >= 11000
                      {kSM_80, DATA_TYPE_FP16, 128, 64, 4},
                      {kSM_80, DATA_TYPE_FP16, 256, 64, 4},
                      {kSM_80, DATA_TYPE_FP16, 384, 64, 4},
                      {kSM_80, DATA_TYPE_INT8, 128, 64, 4},
                      {kSM_80, DATA_TYPE_INT8, 192, 64, 16},
                      {kSM_80, DATA_TYPE_INT8, 256, 64, 8},
                      {kSM_80, DATA_TYPE_INT8, 384, 64, 8},

                      {kSM_80, DATA_TYPE_FP16, 128, 32, 4},
                      {kSM_80, DATA_TYPE_FP16, 256, 32, 4},

                      {kSM_86, DATA_TYPE_FP16, 128, 64, 4},
                      {kSM_86, DATA_TYPE_FP16, 256, 64, 4},
                      {kSM_86, DATA_TYPE_INT8, 128, 64, 4},
                      {kSM_86, DATA_TYPE_INT8, 192, 64, 16},
                      {kSM_86, DATA_TYPE_INT8, 256, 64, 8},
                      {kSM_86, DATA_TYPE_INT8, 384, 64, 8},

                      {kSM_86, DATA_TYPE_FP16, 128, 32, 4},
                      {kSM_86, DATA_TYPE_FP16, 256, 32, 4},
#endif
                  };
            for (auto const& entry : unrollList)
            {
                if (mSM == entry.mSM && mDataType == entry.mDataType && params.s == entry.mS && params.d == entry.mD
                    && params.b <= entry.mMaxBatch)
                {
                    forceUnroll = true;
                    break;
                }
            }
        }

        auto const findIter = mFunctions.find(hashID(params.s, params.d, params.interleaved, forceUnroll));
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

        auto const& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        CUfunction func = findIter->second.mDeviceFunction;

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

template <typename TFusedMHAKernelList>
class TFusedMHAKernelFactory
{
public:
    TFusedMHAKernelList const* getXMMAKernels(
        typename TFusedMHAKernelList::KernelMeta const* pKernelList, int32_t nbKernels, DataType type, int32_t sm)
    {
        static std::mutex sMutex;
        std::lock_guard<std::mutex> lg(sMutex);

        auto const id = hashID(type, sm);
        auto const findIter = mKernels.find(id);
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
        static TFusedMHAKernelFactory<TFusedMHAKernelList> sFactory;
        return sFactory;
    }

private:
    TFusedMHAKernelFactory() = default;

    uint64_t hashID(DataType type, int32_t sm) const
    {
        return static_cast<uint64_t>(type) << 32 | sm;
    }

    std::unordered_map<uint64_t, const std::unique_ptr<TFusedMHAKernelList>> mKernels{};
};

using FusedMHAKernelFactoryV2 = TFusedMHAKernelFactory<FusedMultiHeadAttentionXMMAKernelV2>;

inline FusedMultiHeadAttentionXMMAKernelV2 const* getXMMAKernelsV2(DataType type, int32_t sm)
{
    return FusedMHAKernelFactoryV2::Get().getXMMAKernels(
        sMhaKernelMetaInfosV2, sizeof(sMhaKernelMetaInfosV2) / sizeof(sMhaKernelMetaInfosV2[0]), type, sm);
}

} // namespace plugin
} // namespace nvinfer1
#endif // TRT_FMHA_WITH_POSITION_BIAS_V2_H
