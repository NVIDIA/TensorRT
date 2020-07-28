/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#include "cudaDriverWrapper.h"
#include "cuda_runtime_api.h"
#include <assert.h>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <unordered_map>
#include <vector>

namespace bert
{

static constexpr int32_t kSM_TURING = 75;
static constexpr int32_t kSM_AMPERE = 80;

enum Data_type
{
    DATA_TYPE_BOOL,
    DATA_TYPE_E8M10,
    DATA_TYPE_E8M7,
    DATA_TYPE_FP16,
    DATA_TYPE_FP32,
    DATA_TYPE_INT4,
    DATA_TYPE_INT8,
    DATA_TYPE_INT32
};

static inline size_t get_size_in_bytes(size_t n, Data_type dtype)
{
    switch (dtype)
    {
    case DATA_TYPE_E8M10: return n * 4;
    case DATA_TYPE_FP32: return n * 4;
    case DATA_TYPE_FP16: return n * 2;
    case DATA_TYPE_INT32: return n * 4;
    case DATA_TYPE_INT8: return n;
    case DATA_TYPE_INT4: return n / 2;
    case DATA_TYPE_BOOL: return n / 8;
    case DATA_TYPE_E8M7: return n * 2;
    default: assert(false); return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_attention_params
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
};

////////////////////////////////////////////////////////////////////////////////////////////////////
extern unsigned char fused_multihead_attention_fp16_128_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_fp16_384_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_int8_128_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_int8_384_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_int8_384_64_kernel_sm80_cu_o[];
extern unsigned char fused_multihead_attention_int8_128_64_kernel_sm80_cu_o[];
extern unsigned char fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o[];
extern unsigned char fused_multihead_attention_fp16_384_64_kernel_sm80_cu_o[];

extern unsigned int fused_multihead_attention_fp16_128_64_kernel_sm75_cu_o_len;
extern unsigned int fused_multihead_attention_fp16_384_64_kernel_sm75_cu_o_len;
extern unsigned int fused_multihead_attention_int8_128_64_kernel_sm75_cu_o_len;
extern unsigned int fused_multihead_attention_int8_384_64_kernel_sm75_cu_o_len;
extern unsigned int fused_multihead_attention_int8_384_64_kernel_sm80_cu_o_len;
extern unsigned int fused_multihead_attention_int8_128_64_kernel_sm80_cu_o_len;
extern unsigned int fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o_len;
extern unsigned int fused_multihead_attention_fp16_384_64_kernel_sm80_cu_o_len;

static const struct FusedMultiHeadAttentionKernelMetaInfo
{
    Data_type mDataType;
    unsigned int mS;
    unsigned int mD;
    unsigned int mSM;
    const unsigned char* mCubin;
    unsigned int mCubinSize;
    const char* mFuncName;
    unsigned int mSharedMemBytes;
    unsigned int mThreadsPerCTA;
} sMhaKernelMetaInfos[] = {
    // Turing
    {DATA_TYPE_FP16, 128, 64, kSM_TURING, fused_multihead_attention_fp16_128_64_kernel_sm75_cu_o,
        fused_multihead_attention_fp16_128_64_kernel_sm75_cu_o_len, "fused_multihead_attention_fp16_128_64_kernel_sm75",
        32768, 128},
    {DATA_TYPE_FP16, 384, 64, kSM_TURING, fused_multihead_attention_fp16_384_64_kernel_sm75_cu_o,
        fused_multihead_attention_fp16_384_64_kernel_sm75_cu_o_len, "fused_multihead_attention_fp16_384_64_kernel_sm75",
        57344, 256},
    {DATA_TYPE_INT8, 128, 64, kSM_TURING, fused_multihead_attention_int8_128_64_kernel_sm75_cu_o,
        fused_multihead_attention_int8_128_64_kernel_sm75_cu_o_len, "fused_multihead_attention_int8_128_64_kernel_sm75",
        16384, 128},
    {DATA_TYPE_INT8, 384, 64, kSM_TURING, fused_multihead_attention_int8_384_64_kernel_sm75_cu_o,
        fused_multihead_attention_int8_384_64_kernel_sm75_cu_o_len, "fused_multihead_attention_int8_384_64_kernel_sm75",
        53284, 256},
#if CUDA_VERSION >= 11000
    // Ampere
    {DATA_TYPE_FP16, 128, 64, kSM_AMPERE, fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o,
        fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o_len, "fused_multihead_attention_fp16_128_64_kernel_sm80",
        49152, 128},
    {DATA_TYPE_FP16, 384, 64, kSM_AMPERE, fused_multihead_attention_fp16_384_64_kernel_sm80_cu_o,
        fused_multihead_attention_fp16_384_64_kernel_sm80_cu_o_len, "fused_multihead_attention_fp16_384_64_kernel_sm80",
        114688, 256},
    {DATA_TYPE_INT8, 128, 64, kSM_AMPERE, fused_multihead_attention_int8_128_64_kernel_sm80_cu_o,
        fused_multihead_attention_int8_128_64_kernel_sm80_cu_o_len, "fused_multihead_attention_int8_128_64_kernel_sm80",
        24576, 128},
    {DATA_TYPE_INT8, 384, 64, kSM_AMPERE, fused_multihead_attention_int8_384_64_kernel_sm80_cu_o,
        fused_multihead_attention_int8_384_64_kernel_sm80_cu_o_len, "fused_multihead_attention_int8_384_64_kernel_sm80",
        57344, 256},
#endif
};

struct FusedMultiHeadAttentionXMMAKernel
{
    inline uint64_t hashID(unsigned int s, unsigned int d) const
    {
        return (uint64_t) s << 32 | d;
    }

    FusedMultiHeadAttentionXMMAKernel(Data_type type, unsigned int sm)
        : mDataType(type)
        , mSM(sm)
    {
        for (unsigned int i = 0; i < sizeof(sMhaKernelMetaInfos) / sizeof(sMhaKernelMetaInfos[0]); ++i)
        {
            const auto& kernelMeta = sMhaKernelMetaInfos[i];
            if (kernelMeta.mSM == sm && kernelMeta.mDataType == type)
            {
                CUmodule hmod{0};
                cuErrCheck(mDriver.cuModuleLoadData(&hmod, kernelMeta.mCubin), mDriver);
                mModules.push_back(hmod);

                FusedMultiHeadAttentionKernelInfo funcInfo;
                funcInfo.mMetaInfoIndex = i;
                cuErrCheck(mDriver.cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod, kernelMeta.mFuncName), mDriver);
                if (kernelMeta.mSharedMemBytes >= 48 * 1024)
                {
                    cuErrCheck(mDriver.cuFuncSetAttribute(funcInfo.mDeviceFunction,
                                   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernelMeta.mSharedMemBytes),
                        mDriver);
                }
                mFunctions.insert(std::make_pair(hashID(kernelMeta.mS, kernelMeta.mD), funcInfo));
            }
        }
    }

    ~FusedMultiHeadAttentionXMMAKernel()
    {
        for (auto mod : mModules)
        {
            mDriver.cuModuleUnload(mod);
        }
        mFunctions.clear();
        mModules.clear();
    }

    bool isValid() const
    {
        return !mFunctions.empty();
    }

    void run(Fused_multihead_attention_params& params, size_t s, size_t d, cudaStream_t ss) const
    {
        const auto findIter = mFunctions.find(hashID(s, d));
        ASSERT(findIter != mFunctions.end());

        const auto& kernelMeta = sMhaKernelMetaInfos[findIter->second.mMetaInfoIndex];
        const CUfunction func = findIter->second.mDeviceFunction;

        void* kernelParams[] = {&params, nullptr};
        cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, 1, kernelMeta.mThreadsPerCTA, 1, 1,
                       kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
            mDriver);
    }

    nvinfer1::CUDADriverWrapper mDriver;

    Data_type mDataType;
    unsigned int mSM;
    std::vector<CUmodule> mModules;
    struct FusedMultiHeadAttentionKernelInfo
    {
        unsigned int mMetaInfoIndex;
        CUfunction mDeviceFunction;
    };
    std::unordered_map<uint64_t, FusedMultiHeadAttentionKernelInfo> mFunctions;
};

class FusedMHAKernelFactory
{
public:
    const FusedMultiHeadAttentionXMMAKernel* getXMMAKernels(Data_type type, unsigned int sm)
    {
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lg(s_mutex);

        const auto id = hashID(type, sm);
        const auto findIter = mKernels.find(id);
        if (findIter == mKernels.end())
        {
            FusedMultiHeadAttentionXMMAKernel* newKernel = new FusedMultiHeadAttentionXMMAKernel{type, sm};
            mKernels.insert(std::make_pair(id, std::unique_ptr<FusedMultiHeadAttentionXMMAKernel>(newKernel)));
            return newKernel;
        }
        return findIter->second.get();
    }

    static FusedMHAKernelFactory& Get()
    {
        static FusedMHAKernelFactory s_factory;
        return s_factory;
    }

private:
    FusedMHAKernelFactory() = default;

    inline uint64_t hashID(Data_type type, unsigned int sm) const
    {
        return (uint64_t) type << 32 | sm;
    }

    std::unordered_map<uint64_t, const std::unique_ptr<FusedMultiHeadAttentionXMMAKernel>> mKernels;
};

} // namespace bert
