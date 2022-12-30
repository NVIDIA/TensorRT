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

#include "NvInfer.h"
#include "common/bertCommon.h"
#include "common/common.cuh"
#include "common/plugin.h"
#include "common/serialize.hpp"
#include "qkvToContextWithPosBiasPlugin.h"
#include "swinQKVToContextPlugin/fmha_with_position_bias/fmha_with_position_bias_v2.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <tuple>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
template <typename T>
inline uint32_t asUInt32(T const& val)
{
    PLUGIN_VALIDATE(sizeof(T) == sizeof(uint32_t));
    return *reinterpret_cast<uint32_t const*>(reinterpret_cast<void const*>(&val));
}

size_t MHARunner::getSerializationSize() const noexcept
{
    return sizeof(mS) + sizeof(mB) + sizeof(mW);
}

void MHARunner::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mS);
    serialize_value(&buffer, mB);
    serialize_value(&buffer, mW);
}

void MHARunner::deserialize(void const* data, size_t length)
{
    deserialize_value(&data, &length, &mS);
    deserialize_value(&data, &length, &mB);
    deserialize_value(&data, &length, &mW);
    setup(mS, mB, mW);
}

static inline void setAlpha(uint32_t& alpha, float norm, DataType dtype)
{
    if (dtype == DATA_TYPE_FP16)
    {
        half2 h2 = __float2half2_rn(norm);
        alpha = asUInt32(h2);
    }
    else if (dtype == DATA_TYPE_FP32)
    {
        alpha = asUInt32(norm);
    }
    else if (dtype == DATA_TYPE_INT32)
    {
        int32_t inorm = static_cast<int32_t>(norm);
        alpha = asUInt32(inorm);
    }
    else
    {
        PLUGIN_ERROR("setAlpha: data type must be either FP16, FP32 or INT32");
    }
}

class FusedMHARunnerFP16v2::mhaImpl
{
public:
    mhaImpl(FusedMHARunnerFP16v2* interface)
        : interface(interface)
        , sm(interface->mSm)
        , xmmaKernel(getXMMAKernelsV2(DATA_TYPE_FP16, sm))
    {
        PLUGIN_VALIDATE(
            (sm == kSM_70 || sm == kSM_72 || sm == kSM_75 || sm == kSM_80 || sm == kSM_86), "Unsupported architecture");
        params.clear();
    }

    ~mhaImpl() {}

    size_t getPackedMaskSizeInBytes() const
    {
        // check that we initialized
        PLUGIN_VALIDATE(mXMMAS > 0);
        PLUGIN_VALIDATE(threadsPerCTA > 0);
        PLUGIN_VALIDATE(interface->mB > 0);
        return interface->mB * mXMMAS * threadsPerCTA * sizeof(uint32_t);
    }

    void setup(int32_t const S, int32_t const B, int32_t const windowNum)
    {
        // TODO these implementation details might be better centralized into the XMMA code, since they are needed in
        // several places (also outside of this plugin)
        size_t warpsM{};
        size_t warpsN{};
        size_t warpsK = 1U;
        if (S == 64 || S == 128)
        {
            warpsM = 2U;
            warpsN = 2U;
        }
        else if (S == 256)
        {
            warpsM = 1U;
            warpsN = 4U;
        }
        else if (S == 384)
        {
            warpsM = 1U;
            warpsN = 8U;
        }
        else
        {
            PLUGIN_VALIDATE(false, "Unsupported sequence length.");
        }
        // The number of threads per CTA.
        threadsPerCTA = warpsM * warpsN * warpsK * 32;
        // The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M dimension.
        mXMMAS = (S + 16 * warpsM - 1) / (16 * warpsM);
        // The number of xmmas in the N dimension.
        nXMMAS = (S + 16 * warpsN - 1) / (16 * warpsN);

        float const scaleBmm1 = interface->mQKVScale;
        float const scaleSoftmax = 1.F; // Seems to be only required for int8
        float const scaleBmm2 = 1.F;

        DataType scaleType = DATA_TYPE_FP16;
        setAlpha(params.scaleQK, scaleBmm1, scaleType);
        setAlpha(params.scaleSoftmax, scaleSoftmax, scaleType);
        setAlpha(params.scaleVAttn, scaleBmm2, scaleType);

        params.b = B;
        params.h = interface->mNumHeads;
        params.s = S;
        params.d = interface->mHeadSize;
        params.windowNum = windowNum;

        params.qkvStrideInBytes = 3 * interface->mNumHeads * interface->mHeadSize * sizeof(half);
        params.packedMaskStrideInBytes = S * sizeof(half);
        params.oStrideInBytes = interface->mNumHeads * interface->mHeadSize * sizeof(half);
    }

    void run(void const* qkvPtr, void const* maskPtr, void const* relativePositionBias, int32_t const actualSeqlen,
        void* output, void* workspace, cudaStream_t stream)
    {
        params.packedMaskPtr = interface->mHasMask ? const_cast<void*>(maskPtr) : nullptr;

        params.packedRelativePositionBiasPtr = const_cast<void*>(relativePositionBias);

        params.qkvPtr = const_cast<void*>(qkvPtr);

        params.oPtr = output;

        params.actualSeqlen = actualSeqlen;

        params.cuSeqlens = nullptr;
        xmmaKernel->run(params, stream);
        PLUGIN_CHECK(cudaPeekAtLastError());
    }

    bool isValid(int32_t s) const
    {
        return xmmaKernel->isValid(s);
    }

    int32_t getSFromMaxSeqLen(int32_t const maxSeqLen)
    {
        int32_t S = -1;
        if (maxSeqLen <= 64)
        {
            S = 64;
        }
        else if (maxSeqLen <= 128)
        {
            S = 128;
        }
        else if (maxSeqLen <= 256)
        {
            S = 256;
        }
        return S;
    }

private:
    FusedMHARunnerFP16v2* interface {
    };
    FusedMHAParams params{};
    int32_t sm{};
    FusedMultiHeadAttentionXMMAKernelV2 const* xmmaKernel{};
    size_t mXMMAS{};
    size_t nXMMAS{};
    size_t threadsPerCTA{};
};

FusedMHARunnerFP16v2::FusedMHARunnerFP16v2(int32_t const numHeads, int32_t const headSize, int32_t const sm,
    int32_t const hasMask, float const qkvscale, float const qScaling)
    : MHARunner(nvinfer1::DataType::kHALF, numHeads, headSize, hasMask, qkvscale)
    , mSm(sm)
    , pImpl(new mhaImpl(this))
{
}

void FusedMHARunnerFP16v2::setup(int32_t const S, int32_t const B, int32_t const windowNum)
{
    MHARunner::setup(S, B, windowNum);
    pImpl->setup(S, B, windowNum);
}

size_t FusedMHARunnerFP16v2::getWorkspaceSize() const
{
    return 0;
}

void FusedMHARunnerFP16v2::run(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{

    void const* qkvPtr = inputs[0];
    void const* maskPtr = inputs[1];
    void const* posBiasPtr = inputs[2];
    int32_t const actualSeqLen = inputDesc[0].dims.d[1]; // input is [BxnW, S, 3xEmbed] or [BxnW, S, 3xEmbed, 1, 1]
    pImpl->run(qkvPtr, maskPtr, posBiasPtr, actualSeqLen, outputs[0], workspace, stream);
}

bool FusedMHARunnerFP16v2::isValid(int32_t s) const
{
    return pImpl->isValid(s);
}

int32_t FusedMHARunnerFP16v2::getSFromMaxSeqLen(int32_t const maxSeqLen)
{
    return pImpl->getSFromMaxSeqLen(maxSeqLen);
}

// Int8 starts here: TODO refactor the duplicate stuff

class FusedMHARunnerInt8v2::mhaImpl
{

public:
    mhaImpl(FusedMHARunnerInt8v2* interface)
        : interface(interface)
        , sm(interface->mSm)
        , xmmaKernel(getXMMAKernelsV2(DATA_TYPE_INT8, sm))
        , mDqProbs(interface->mDqProbs)
    {
        PLUGIN_VALIDATE((sm == kSM_72 || sm == kSM_75 || sm == kSM_80 || sm == kSM_86), "Unsupported architecture");
        params.clear();
    }

    ~mhaImpl() {}

    size_t getPackedMaskSizeInBytes() const
    {
        PLUGIN_VALIDATE(mXMMAS > 0);
        PLUGIN_VALIDATE(threadsPerCTA > 0);
        PLUGIN_VALIDATE(interface->mB > 0);
        return interface->mB * mXMMAS * threadsPerCTA * sizeof(uint32_t);
    }

    void setup(int32_t const S, int32_t const B, int32_t const windowNum)
    {
        size_t warpsM{};
        size_t warpsN{};
        size_t warpsK = 1U;
        if (S == 64)
        {
            warpsM = 2U;
            warpsN = 2U;
        }
        else if (S == 128)
        {
            warpsM = 2U;
            warpsN = 2U;
        }
        else if (S == 256)
        {
            warpsM = 1U;
            warpsN = 4U;
        }
        else if (S == 384)
        {
            warpsM = 1U;
            warpsN = 8U;
        }
        else
        {
            PLUGIN_VALIDATE(false, "Unsupported sequence length.");
        }
        // The number of threads per CTA.
        threadsPerCTA = warpsM * warpsN * warpsK * 32;
        // The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M dimension.
        mXMMAS = (S + 16 * warpsM - 1) / (16 * warpsM);
        // The number of xmmas in the N dimension.
        nXMMAS = (S + 16 * warpsN - 1) / (16 * warpsN);

        params.b = B;
        params.h = interface->mNumHeads;
        params.s = S;
        params.d = interface->mHeadSize;
        params.windowNum = windowNum;
        params.useInt8ScaleMax = true;
        params.packedMaskStrideInBytes = S * sizeof(half);
        params.qkvStrideInBytes = 3 * interface->mNumHeads * interface->mHeadSize * sizeof(int8_t);
        params.oStrideInBytes = interface->mNumHeads * interface->mHeadSize * sizeof(int8_t);
    }

    void run(PluginTensorDesc const& inputDesc, PluginTensorDesc const& outputDesc, void const* qkvPtr,
        void const* maskPtr, void const* relativePositionBiasPtr, int32_t actualSeqlen, void* output, void* workspace,
        cudaStream_t stream)
    {
        float scaleQkv = inputDesc.scale;
        float scaleCtx = outputDesc.scale;

        float scaleBmm1 = scaleQkv * scaleQkv * interface->mQKVScale;
        float scaleBmm2 = mDqProbs * scaleQkv / scaleCtx;
        float scaleSoftmax = 1.F / mDqProbs;

        params.scaleQK = asUInt32(scaleBmm1);
        params.scaleVAttn = asUInt32(scaleBmm2);
        params.scaleSoftmax = asUInt32(scaleSoftmax);

        params.enableI2fTrick = -static_cast<double>(1 << 22) * static_cast<double>(scaleBmm2) <= -128.F
            && static_cast<double>(1 << 22) * static_cast<double>(scaleBmm2) >= 127.F;

        params.qkvPtr = const_cast<void*>(qkvPtr);
        params.packedMaskPtr = interface->mHasMask ? const_cast<void*>(maskPtr) : nullptr;
        params.packedRelativePositionBiasPtr = const_cast<void*>(relativePositionBiasPtr);

        params.actualSeqlen = actualSeqlen;

        params.oPtr = output;

        params.cuSeqlens = nullptr;

        xmmaKernel->run(params, stream);
    }

    bool isValid(int32_t s) const
    {
        return xmmaKernel->isValid(s);
    }

    int32_t getSFromMaxSeqLen(int32_t const maxSeqLen)
    {
        int32_t S = -1;
        if (maxSeqLen <= 64)
        {
            S = 64;
        }
        else if (maxSeqLen <= 256)
        {
            S = 256;
        }
        return S;
    }

private:
    float mDqProbs{};
    FusedMHARunnerInt8v2* interface{};
    FusedMHAParams params{};
    int32_t sm{};
    FusedMultiHeadAttentionXMMAKernelV2 const* xmmaKernel{};
    size_t mXMMAS{};
    size_t nXMMAS{};
    size_t threadsPerCTA{};
};

FusedMHARunnerInt8v2::FusedMHARunnerInt8v2(int32_t const numHeads, int32_t const headSize, int32_t const sm,
    int32_t const hasMask, float const qkvScale, float const qScaling)
    : MHARunner(nvinfer1::DataType::kINT8, numHeads, headSize, hasMask, qkvScale)
    , mSm(sm)
    , pImpl(new mhaImpl(this))
    , mDqProbs(qScaling)
{
}

void FusedMHARunnerInt8v2::setup(int32_t const S, int32_t const B, int32_t const windowNum)
{
    MHARunner::setup(S, B, windowNum);
    pImpl->setup(S, B, windowNum);
}

size_t FusedMHARunnerInt8v2::getWorkspaceSize() const
{
    return 0;
}

void FusedMHARunnerInt8v2::run(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{

    void const* qkvPtr = inputs[0];
    void const* maskPtr = inputs[1];
    void const* posBiasPtr = inputs[2];
    int32_t const actualSeqlen = inputDesc[0].dims.d[1]; // input is [BxnW, S, 3xEmbed] or [BxnW, S, 3xEmbed, 1, 1]
    pImpl->run(inputDesc[0], outputDesc[0], qkvPtr, maskPtr, posBiasPtr, actualSeqlen, outputs[0], workspace, stream);
}

bool FusedMHARunnerInt8v2::isValid(int32_t s) const
{
    return pImpl->isValid(s);
}

int32_t FusedMHARunnerInt8v2::getSFromMaxSeqLen(int32_t const maxSeqLen)
{
    return pImpl->getSFromMaxSeqLen(maxSeqLen);
}
} // namespace plugin
} // namespace nvinfer1
