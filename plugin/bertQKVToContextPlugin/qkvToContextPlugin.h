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

// Need 10.1 for cublasGemmStridedBatchedEx
#include <cuda.h>
#if CUDA_VERSION >= 10010

#ifndef TRT_QKV_TO_CONTEXT_PLUGIN_H
#define TRT_QKV_TO_CONTEXT_PLUGIN_H

#include "NvInferPlugin.h"
#include "cublas_v2.h"
#include "zeroPadding2d.h"
#include <string>
#include <vector>

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
    MHARunner(const nvinfer1::DataType type, const int32_t numHeads, const int32_t headSize)
        : mType(type)
        , mS(0)
        , mB(0)
        , mOmatSize(0)
        , mNumMats(0)
        , mNumHeads(numHeads)
        , mHeadSize(headSize)
        , mWordSize(getElementSize(type))
        , mLdQKV(0)
        , mStrideQKV(0)
        , mLdOut(0)
        , mStrideOut(0)
        , mRsqrtHeadSize(1.F / sqrtf(headSize))
    {
    }

    virtual ~MHARunner() = default;

    virtual void setup(const int32_t S, const int32_t B)
    {
        PLUGIN_ASSERT(S);
        PLUGIN_ASSERT(B);
        mB = B;
        mS = S;

        mLdQKV = 3 * B * mNumHeads * mHeadSize;
        mStrideQKV = 3 * mHeadSize;

        mLdOut = B * mNumHeads * mHeadSize;
        mStrideOut = mHeadSize;
        mOmatSize = S * S;
        mNumMats = B * mNumHeads;
    }

    virtual void run(const nvinfer1::PluginTensorDesc& inputDesc, const nvinfer1::PluginTensorDesc& outputDesc,
        const void* qkvPtr, const void* maskPtr, void* output, void* workspace, cudaStream_t stream)
        = 0;

    virtual void run(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
        = 0;

    virtual size_t getSerializationSize() const noexcept;
    virtual void serialize(void* buffer) const noexcept;
    virtual void deserialize(const void* data, size_t length);

    virtual size_t getWorkspaceSize() const = 0;

    virtual bool isValid(int32_t s) const = 0;

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

std::pair<int, int> tuneBatchedGemm(const int32_t B, const int32_t S, const int32_t numHeads, const int32_t headSize);

template <typename T>
int32_t computeScaledSoftmax(cudaStream_t stream, const int32_t ld, const int32_t B, const int32_t N,
    const float rsqrtHeadSize, const T* input, T* output);

template <typename T>
int32_t computeMaskedScaledSoftmax(cudaStream_t stream, const int32_t ld, const int32_t B, const int32_t N,
    const float rsqrtHeadSize, const int* maskIdx, const T* input, T* output);

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class QKVToContextPluginDynamic : public nvinfer1::IPluginV2DynamicExt
{
public:
    QKVToContextPluginDynamic(const std::string name, const nvinfer1::DataType type, const int32_t hiddenSize,
        const int32_t numHeads, const float dqProbs, bool hasImask = false);

    QKVToContextPluginDynamic(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make QKVToContextPluginDynamic without arguments, so we
    // delete default constructor.
    QKVToContextPluginDynamic() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const
        noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

protected:
    void createMHARunner();

private:
    const std::string mLayerName;
    std::string mNamespace;

    // used for sequence len 128, 384, precision int8
    // used for sequence len 64, 96, 128, 384, precision fp16
    std::unique_ptr<MHARunner> fusedDispatcher;
    // used for other sequence, precision fp32 and fp16
    std::unique_ptr<MHARunner> unfusedDispatcher;

    int32_t mS;
    int32_t mB;
    int32_t mSM;
    int32_t mHeadSize;
    int32_t mHiddenSize;
    int32_t mNumHeads;
    bool mHasImask;
    nvinfer1::DataType mType;
    float mDqProbs;

    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2::enqueue;
    using IPluginV2Ext::configurePlugin;
};

class QKVToContextPluginDynamicCreator : public nvinfer1::IPluginCreator
{
public:
    QKVToContextPluginDynamicCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

class QKVToContextVarSeqlenPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    QKVToContextVarSeqlenPlugin(std::string const name, nvinfer1::DataType const type, int32_t const hiddenSize,
        int32_t const numHeads, float const dqProbs, bool hasImask = false, bool varSeqlen = false, bool const useInt8ScaleMax = true);

    QKVToContextVarSeqlenPlugin(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make QKVToContextVarSeqlenPlugin without arguments, so we
    // delete default constructor.
    QKVToContextVarSeqlenPlugin() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const
        noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

protected:
    void createMHARunner();

private:
    const std::string mLayerName;
    std::string mNamespace;

    std::unique_ptr<MHARunner> dispatcher;
    std::unique_ptr<QkvPaddingRunner> patcher;

    int32_t mS;
    int32_t mB;
    int32_t mSM;
    int32_t mHeadSize;
    int32_t mHiddenSize;
    int32_t mNumHeads;
    bool mHasImask;
    nvinfer1::DataType mType;

    float mDqProbs;

    int32_t mHdim;
    bool mUseVarSeqlen;
    bool mUseInt8ScaleMax{true};

    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2::enqueue;
    using IPluginV2Ext::configurePlugin;
};

class QKVToContextVarSeqlenPluginCreator : public nvinfer1::IPluginCreator
{
public:
    QKVToContextVarSeqlenPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

class UnfusedMHARunner : public MHARunner
{
public:
    UnfusedMHARunner(
        const nvinfer1::DataType type, const int32_t numHeads, const int32_t headSize, const int32_t smVersion);
    virtual ~UnfusedMHARunner();

    virtual void setup(const int32_t S, const int32_t B) override;

    void run(const nvinfer1::PluginTensorDesc& inputDesc, const nvinfer1::PluginTensorDesc& outputDesc,
        const void* qkvPtr, const void* maskPtr, void* output, void* workspace, cudaStream_t stream) override;

    void run(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    size_t getWorkspaceSize() const override;

    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void deserialize(const void* data, size_t length) override;
    bool isValid(int32_t s) const override;

private:
    bool mIsBestAlgoFound;
    int32_t mAlgoBatchedEx1;
    int32_t mAlgoBatchedEx2;
    cublasHandle_t mCublas;
    int32_t mSm;
};

class FusedMHARunnerFP16 : public MHARunner
{
public:
    FusedMHARunnerFP16(const int32_t numHeads, const int32_t headSize, const int32_t sm);
    ~FusedMHARunnerFP16() = default; // for pimpl

    virtual void setup(const int32_t S, const int32_t B) override;

    void run(const nvinfer1::PluginTensorDesc& inputDesc, const nvinfer1::PluginTensorDesc& outputDesc,
        const void* qkvPtr, const void* maskPtr, void* output, void* workspace, cudaStream_t stream) override;

    void run(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    size_t getWorkspaceSize() const override;

    void deserialize(const void* data, size_t length) override;

    bool isValid(int32_t s) const override;

private:
    int32_t mSm;
    class mhaImpl;
    std::unique_ptr<mhaImpl> pimpl;
};

class FusedMHARunnerInt8 : public MHARunner
{
public:
    FusedMHARunnerInt8(const int32_t numHeads, const int32_t headSize, const int32_t sm, const float dqProbs);
    ~FusedMHARunnerInt8() = default; // for pimpl

    virtual void setup(const int32_t S, const int32_t B) override;

    void run(const nvinfer1::PluginTensorDesc& inputDesc, const nvinfer1::PluginTensorDesc& outputDesc,
        const void* qkvPtr, const void* maskPtr, void* output, void* workspace, cudaStream_t stream) override;

    void run(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    size_t getWorkspaceSize() const override;

    void deserialize(const void* data, size_t length) override;

    bool isValid(int32_t s) const override;

private:
    float mDqProbs;
    int32_t mSm;
    class mhaImpl;
    std::unique_ptr<mhaImpl> pimpl;
};

class FusedMHARunnerFP16v2 : public MHARunner
{
public:
    FusedMHARunnerFP16v2(const int32_t numHeads, const int32_t headSize, const int32_t sm);
    ~FusedMHARunnerFP16v2() = default; // for pimpl

    virtual void setup(const int32_t S, const int32_t B) override;

    void run(const nvinfer1::PluginTensorDesc& inputDesc, const nvinfer1::PluginTensorDesc& outputDesc,
        const void* qkvPtr, const void* maskPtr, void* output, void* workspace, cudaStream_t stream) override;

    void run(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    size_t getWorkspaceSize() const override;

    void deserialize(const void* data, size_t length) override;

    bool isValid(int32_t s) const override;

private:
    int32_t mSm;
    class mhaImpl;
    std::unique_ptr<mhaImpl> pimpl;
};

class FusedMHARunnerInt8v2 : public MHARunner
{
public:
    FusedMHARunnerInt8v2(int32_t const numHeads, int32_t const headSize, int32_t const sm, float const dqProbs, bool const useInt8ScaleMax);
    ~FusedMHARunnerInt8v2() = default; // for pimpl

    virtual void setup(const int32_t S, const int32_t B) override;

    void run(const nvinfer1::PluginTensorDesc& inputDesc, const nvinfer1::PluginTensorDesc& outputDesc,
        const void* qkvPtr, const void* maskPtr, void* output, void* workspace, cudaStream_t stream) override;

    void run(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    size_t getWorkspaceSize() const override;

    void deserialize(const void* data, size_t length) override;

    bool isValid(int32_t s) const override;

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
#endif // TRT_QKV_TO_CONTEXT_PLUGIN_H

#endif // CUDA_VERSION >= 10010
