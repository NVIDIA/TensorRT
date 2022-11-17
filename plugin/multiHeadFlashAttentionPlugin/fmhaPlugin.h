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
#ifndef _FMHA_PLUGIN_
#define _FMHA_PLUGIN_
#include "common/bertCommon.h"
#include "fmha_flash_attention/include/commonDatatype.h"
#include "fmha_flash_attention/include/fmha_flash_attention.h"

#include <NvInfer.h>
#include <cassert>
#include <string>
#include <vector>

namespace
{
static char const* PLUGIN_NAME{"fMHA_V2"};
static char const* PLUGIN_VERSION{"1"};
} // namespace

namespace nvinfer1
{
namespace plugin
{
class fmhaPlugin : public IPluginV2DynamicExt
{
private:
    std::string const mLayerName;
    std::string mNamespace;

    // scalar need copy
    struct
    {
        int32_t mOptBatchSize{};
        int32_t mOptSeqLen{};
        int32_t mMaxBatchSize{};
        DataType mDataType{DataType::kFLOAT};
    } m_;

public:
    fmhaPlugin(std::string const& name)
        : mLayerName(name)
    {
        init();
    }

    fmhaPlugin(std::string const& name, void const* data, size_t length)
        : mLayerName(name)
    {
        memcpy(&m_, data, sizeof(m_));
    }

    fmhaPlugin() = delete;
    ~fmhaPlugin() = default;

    void init(bool loadCubins = false)
    {
        try
        {
            mSM = bert::getSMVersion();

            // initialize seqlens buffer
            allocateSeqlens(m_.mMaxBatchSize);
            initializeSeqlens(m_.mOptBatchSize, m_.mOptSeqLen, mCuSeqLen.get());

            if (loadCubins)
            {
                createMHARunner();
            }
        }
        catch (std::exception const& e)
        {
            caughtError(e);
        }
    }

    void createMHARunner()
    {
        switch (m_.mDataType)
        {
        case DataType::kFLOAT: mKernels = getFMHACubinKernels(plugin::DATA_TYPE_FP32, mSM); break;
        case DataType::kHALF: mKernels = getFMHACubinKernels(plugin::DATA_TYPE_FP16, mSM); break;
        default: break;
        }
    }

    size_t getSerializationSize() const noexcept override
    {
        return sizeof(m_);
    }

    void serialize(void* buffer) const noexcept override
    {
        memcpy(buffer, &m_, sizeof(m_));
    }

    IPluginV2DynamicExt* clone() const noexcept override
    {
        try
        {
            std::vector<char> buff;
            buff.resize(getSerializationSize());
            serialize(buff.data());

            auto p = new fmhaPlugin(mLayerName, buff.data(), buff.size());
            p->mCuSeqLen = mCuSeqLen;
            p->setPluginNamespace(mNamespace.c_str());
            p->init(true);
            return p;
        }
        catch (std::exception const& e)
        {
            caughtError(e);
        }
        return nullptr;
    }

    int32_t getNbOutputs() const noexcept override
    {
        return 1;
    }

    // input0 qkv_packed in [b, s, h, 3, d] ??
    // input1 cu_seqlens in [b + 1]
    // output O in [b, s, h, d]
    DimsExprs getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override
    {
        DimsExprs out;
        out.nbDims = 4;
        out.d[0] = inputs[0].d[0];
        ;
        out.d[1] = inputs[0].d[1];
        out.d[2] = inputs[0].d[2];
        out.d[3] = inputs[0].d[4];
        return out;
    }

    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        bool res = false;
        try
        {
            // load kernel and check if we have any implementations.
            auto hasImplement = [this](DataType dt)
            {
                switch (dt)
                {
                case DataType::kFLOAT: return getFMHACubinKernels(plugin::DATA_TYPE_FP32, mSM)->isValid(/*dummy seq*/128);
                case DataType::kHALF: return getFMHACubinKernels(plugin::DATA_TYPE_FP16, mSM)->isValid(/*dummy seq*/128);
                default: break;
                }
                return false;
            };

            if (inOut[pos].format != TensorFormat::kLINEAR)
            {
                return false;
            }

            switch (pos)
            {
            case 0: res = hasImplement(inOut[pos].type) && inOut[pos].dims.nbDims == 5; break;
            case 1: res = inOut[pos].type == inOut[0].type && inOut[pos].dims.nbDims == 4; break;
            default: // should NOT be here
                break;
            }
        }
        catch (std::exception const& e)
        {
            caughtError(e);
        }
        return res;
    }

    DataType getOutputDataType(
        int32_t outputIndex, DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        return inputTypes[0];
    }

    void allocateSeqlens(int32_t maxBatchSize)
    {
        // allocate seqlens buffer
        if (!mCuSeqLen && maxBatchSize)
        {
            void* cudaMem{nullptr};
            PLUGIN_CHECK(cudaMalloc(&cudaMem, sizeof(int32_t) * (maxBatchSize + 1)));
            bert::make_cuda_shared(mCuSeqLen, cudaMem);
        }

        m_.mMaxBatchSize = maxBatchSize;
    }

    void initializeSeqlens(int32_t b, int32_t s, void* cu_seqlens_d, cudaStream_t stream = 0)
    {
        if (!b || !s)
        {
            return;
        }

        std::vector<int32_t> cuSeqLens(b + 1, 0);
        // Compute the prefix sum of the seqlen
        for (int32_t it = 0; it < b; it++)
        {
            cuSeqLens[it + 1] = cuSeqLens[it] + s;
        }

        PLUGIN_CUASSERT(cudaMemcpyAsync(
                                        cu_seqlens_d, cuSeqLens.data(), sizeof(int32_t) * cuSeqLens.size(), cudaMemcpyHostToDevice, stream));
        m_.mOptBatchSize = b;
        m_.mOptSeqLen = s;
    };

    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override
    {
        try
        {
            int32_t const batchSize = in[0].max.d[0];
            int32_t const seqLen = in[0].max.d[1];

            allocateSeqlens(batchSize);
            if (batchSize != m_.mOptBatchSize || seqLen != m_.mOptSeqLen)
            {
                initializeSeqlens(batchSize, seqLen, mCuSeqLen.get());
            }

            m_.mDataType = in[0].desc.type;
            createMHARunner();
        }
        catch (std::exception const& e)
        {
            caughtError(e);
        }
    }

    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override
    {
        return 0;
    }

    void setPluginNamespace(char const* szNamespace) noexcept override
    {
        mNamespace = szNamespace;
    }
    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }
    char const* getPluginType() const noexcept override
    {
        return PLUGIN_NAME;
    }
    char const* getPluginVersion() const noexcept override
    {
        return PLUGIN_VERSION;
    }
    int32_t initialize() noexcept override
    {
        return 0;
    }
    void terminate() noexcept override
    {
        return;
    }

    void destroy() noexcept override
    {
        delete this;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

private:
    bert::cuda_shared_ptr<void> mCuSeqLen;
    int32_t mSM{};
    FusedMultiHeadFlashAttentionKernel const* mKernels{};

}; // class fmhaPlugin

class fmhaPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection mFc;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;

public:
    fmhaPluginCreator()
    {
        mFc.nbFields = mPluginAttributes.size();
        mFc.fields = mPluginAttributes.data();
    }

    ~fmhaPluginCreator() {}

    IPluginV2* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override
    {
        try
        {
            return new fmhaPlugin(name);
        }
        catch (std::exception const& e)
        {
            caughtError(e);
        }
        return nullptr;
    }

    IPluginV2* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override
    {
        try
        {
            auto p = new fmhaPlugin(name, serialData, serialLength);
            p->init(true);
            return p;
        }
        catch (std::exception const& e)
        {
            caughtError(e);
        }
        return nullptr;
    }

    void setPluginNamespace(char const* szNamespace) noexcept override
    {
        mNamespace = szNamespace;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

    char const* getPluginName() const noexcept override
    {
        return PLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override
    {
        return PLUGIN_VERSION;
    }

    PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mFc;
    }
}; // class fmhaPluginCreator
} // namespace plugin
} // namespace nvinfer1

#endif
