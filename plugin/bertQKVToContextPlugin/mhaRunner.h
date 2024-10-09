/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_MHA_RUNNER_H
#define TRT_MHA_RUNNER_H

// Need 10.1 for cublasGemmStridedBatchedEx
#include <cuda.h>
#if CUDA_VERSION >= 10010

#include "NvInferPlugin.h"
#include "common/cublasWrapper.h"
#include "zeroPadding2d.h"
#include <math.h>
#include <string>
#include <vector>

using namespace nvinfer1::pluginInternal;

namespace nvinfer1
{
namespace plugin
{
namespace bert
{

// Multi Head Attention runner
class MHARunner
{
public:
    MHARunner(nvinfer1::DataType const type, int32_t const numHeads)
        : mType(type)
        , mS(0)
        , mB(0)
        , mOmatSize(0)
        , mNumMats(0)
        , mNumHeads(numHeads)
        , mHeadSize(0)
        , mWordSize(getElementSize(type))
        , mLdQKV(0)
        , mStrideQKV(0)
        , mLdOut(0)
        , mStrideOut(0)
        , mRsqrtHeadSize(0)
    {
    }

    virtual ~MHARunner() = default;

    virtual void setup(int32_t S, int32_t B, int32_t headSize)
    {
        PLUGIN_ASSERT(S);
        PLUGIN_ASSERT(B);
        mB = B;
        mS = S;
        mHeadSize = headSize;
        mRsqrtHeadSize = 1.F / std::sqrt(headSize);

        mLdQKV = 3 * B * mNumHeads * mHeadSize;
        mStrideQKV = 3 * mHeadSize;

        mLdOut = B * mNumHeads * mHeadSize;
        mStrideOut = mHeadSize;
        mOmatSize = S * S;
        mNumMats = B * mNumHeads;
    }

    virtual void run(nvinfer1::PluginTensorDesc const& inputDesc, nvinfer1::PluginTensorDesc const& outputDesc,
        void const* qkvPtr, void const* maskPtr, void* output, void* workspace, cudaStream_t stream,
        nvinfer1::pluginInternal::cublasHandle_t cublas)
        = 0;

    virtual void run(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream,
        nvinfer1::pluginInternal::cublasHandle_t cublas)
        = 0;

    virtual size_t getSerializationSize() const noexcept;
    virtual void serialize(void* buffer) const noexcept;
    virtual void deserialize(void const* data, size_t length);

    virtual size_t getWorkspaceSize() const = 0;

    virtual bool isValid(int32_t headSize, int32_t s) const = 0;

protected:
    nvinfer1::DataType mType;

    int32_t mS;
    int32_t mB;
    int32_t mOmatSize;
    int32_t mNumMats;
    int32_t mNumHeads;
    int32_t mHeadSize;
    int32_t mWordSize;
    int32_t mLdQKV;
    int32_t mStrideQKV;
    int32_t mLdOut;
    int32_t mStrideOut;

    float mRsqrtHeadSize;
};

class UnfusedMHARunner : public MHARunner
{
public:
    UnfusedMHARunner(nvinfer1::DataType const type, int32_t const numHeads, int32_t const smVersion);
    virtual ~UnfusedMHARunner();

    virtual void setup(int32_t S, int32_t B, int32_t headSize) override;

    void run(nvinfer1::PluginTensorDesc const& inputDesc, nvinfer1::PluginTensorDesc const& outputDesc,
        void const* qkvPtr, void const* maskPtr, void* output, void* workspace, cudaStream_t stream,
        nvinfer1::pluginInternal::cublasHandle_t cublas) override;

    void run(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream,
        nvinfer1::pluginInternal::cublasHandle_t cublas) override;

    size_t getWorkspaceSize() const override;

    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void deserialize(void const* data, size_t length) override;
    bool isValid(int32_t headSize, int32_t s) const override;

private:
    bool mIsBestAlgoFound{};
    int32_t mAlgoBatchedEx1{};
    int32_t mAlgoBatchedEx2{};
    int32_t mSm{};
};

class FusedMHARunnerFP16 : public MHARunner
{
public:
    FusedMHARunnerFP16(int32_t const numHeads, int32_t const sm);
    ~FusedMHARunnerFP16() = default; // for pimpl

    virtual void setup(int32_t S, int32_t B, int32_t headSize) override;

    void run(nvinfer1::PluginTensorDesc const& inputDesc, nvinfer1::PluginTensorDesc const& outputDesc,
        void const* qkvPtr, void const* maskPtr, void* output, void* workspace, cudaStream_t stream,
        nvinfer1::pluginInternal::cublasHandle_t cublas) override;

    void run(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream,
        nvinfer1::pluginInternal::cublasHandle_t cublas) override;

    size_t getWorkspaceSize() const override;

    void deserialize(void const* data, size_t length) override;

    bool isValid(int32_t headSize, int32_t s) const override;

private:
    int32_t mSm;
    class mhaImpl;
    std::unique_ptr<mhaImpl> pimpl;
};

class FusedMHARunnerInt8 : public MHARunner
{
public:
    FusedMHARunnerInt8(int32_t const numHeads, int32_t const sm, float const dqProbs);
    ~FusedMHARunnerInt8() = default; // for pimpl

    virtual void setup(int32_t S, int32_t B, int32_t headSize) override;

    void run(nvinfer1::PluginTensorDesc const& inputDesc, nvinfer1::PluginTensorDesc const& outputDesc,
        void const* qkvPtr, void const* maskPtr, void* output, void* workspace, cudaStream_t stream,
        nvinfer1::pluginInternal::cublasHandle_t cublas) override;

    void run(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream,
        nvinfer1::pluginInternal::cublasHandle_t cublas) override;

    size_t getWorkspaceSize() const override;

    void deserialize(void const* data, size_t length) override;

    bool isValid(int32_t headSize, int32_t s) const override;

private:
    float mDqProbs;
    int32_t mSm;
    class mhaImpl;
    std::unique_ptr<mhaImpl> pimpl;
};

class FusedMHARunnerFP16v2 : public MHARunner
{
public:
    FusedMHARunnerFP16v2(int32_t const numHeads, int32_t const sm);
    ~FusedMHARunnerFP16v2() = default; // for pimpl

    virtual void setup(int32_t S, int32_t B, int32_t headSize) override;

    void run(nvinfer1::PluginTensorDesc const& inputDesc, nvinfer1::PluginTensorDesc const& outputDesc,
        void const* qkvPtr, void const* maskPtr, void* output, void* workspace, cudaStream_t stream,
        nvinfer1::pluginInternal::cublasHandle_t cublas) override;

    void run(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream,
        nvinfer1::pluginInternal::cublasHandle_t cublas) override;

    size_t getWorkspaceSize() const override;

    void deserialize(void const* data, size_t length) override;

    bool isValid(int32_t headSize, int32_t s) const override;

private:
    int32_t mSm;
    class mhaImpl;
    std::unique_ptr<mhaImpl> pimpl;
};

class FusedMHARunnerInt8v2 : public MHARunner
{
public:
    FusedMHARunnerInt8v2(int32_t const numHeads, int32_t const sm, float const dqProbs, bool const useInt8ScaleMax);
    ~FusedMHARunnerInt8v2() = default; // for pimpl

    virtual void setup(int32_t S, int32_t B, int32_t headSize) override;

    void run(nvinfer1::PluginTensorDesc const& inputDesc, nvinfer1::PluginTensorDesc const& outputDesc,
        void const* qkvPtr, void const* maskPtr, void* output, void* workspace, cudaStream_t stream,
        nvinfer1::pluginInternal::cublasHandle_t cublas) override;

    void run(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream,
        nvinfer1::pluginInternal::cublasHandle_t cublas) override;

    size_t getWorkspaceSize() const override;

    void deserialize(void const* data, size_t length) override;

    bool isValid(int32_t headSize, int32_t s) const override;

private:
    float mDqProbs;
    int32_t mSm;
    class mhaImpl;
    std::unique_ptr<mhaImpl> pimpl;
    bool mUseInt8ScaleMax{true};
};

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_MHA_RUNNER_H

#endif // CUDA_VERSION >= 10010
