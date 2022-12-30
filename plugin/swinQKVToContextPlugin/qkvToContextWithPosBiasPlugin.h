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

#ifndef TRT_QKV_TO_CONTEXT_WITH_POS_BIAS_PLUGIN_H
#define TRT_QKV_TO_CONTEXT_WITH_POS_BIAS_PLUGIN_H

#include "NvInferPlugin.h"
#include "common/bertCommon.h"
#include "common/checkMacrosPlugin.h"
#include <cublas_v2.h>
#include <cuda.h>
#include <memory>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
// Multi Head Attention runner
class MHARunner
{
public:
    MHARunner() = default;

    MHARunner(nvinfer1::DataType const type, int32_t const numHeads, int32_t const headSize, int32_t const hasMask,
        float const qkvScale)
        : mType(type)
        , mNumHeads(numHeads)
        , mHeadSize(headSize)
        , mHasMask(hasMask)
        , mWordSize(bert::getElementSize(type))
        , mQKVScale(qkvScale)
    {
    }

    virtual ~MHARunner() = default;

    virtual void setup(int32_t const S, int32_t const B, int32_t const windowNum)
    {
        PLUGIN_VALIDATE(S > 0);
        PLUGIN_VALIDATE(B);
        PLUGIN_VALIDATE(windowNum);
        mB = B;
        mS = S;
        mW = windowNum;

        mLdQKV = 3 * B * mNumHeads * mHeadSize;
        mStrideQKV = 3 * mHeadSize;

        mLdOut = B * mNumHeads * mHeadSize;
        mStrideOut = mHeadSize;
        mOmatSize = S * S;
        mNumMats = B * mNumHeads;
    }

    virtual void run(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
        = 0;

    virtual size_t getSerializationSize() const noexcept;
    virtual void serialize(void* buffer) const noexcept;
    virtual void deserialize(void const* data, size_t length);

    virtual size_t getWorkspaceSize() const = 0;

    virtual bool isValid(int32_t s) const = 0;

    virtual int32_t getSFromMaxSeqLen(int32_t const maxSeqLen) = 0;

protected:
    nvinfer1::DataType mType{};

    int32_t mS{};
    int32_t mB{};
    int32_t mW{};
    int32_t mOmatSize{};
    int32_t mNumMats{};
    int32_t mNumHeads{};
    int32_t mHeadSize{};
    int32_t mHasMask{};
    int32_t mWordSize{};
    int32_t mLdQKV{};
    int32_t mStrideQKV{};
    int32_t mLdOut{};
    int32_t mStrideOut{};

    float mQKVScale{};
};

class FusedMHARunnerFP16v2 : public MHARunner
{
public:
    FusedMHARunnerFP16v2() = default;
    FusedMHARunnerFP16v2(int32_t const numHeads, int32_t const headSize, int32_t const sm, int32_t const hasMask,
        float const qkvScale, float const qScaling);
    ~FusedMHARunnerFP16v2() = default; // for pImpl

    virtual void setup(int32_t const S, int32_t const B, int32_t const windowNum) override;

    void run(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    size_t getWorkspaceSize() const override;

    bool isValid(int32_t s) const override;

    int32_t getSFromMaxSeqLen(int32_t const maxSeqLen) override;

private:
    int32_t mSm{};
    class mhaImpl;
    std::unique_ptr<mhaImpl> pImpl;
};

class FusedMHARunnerInt8v2 : public MHARunner
{
public:
    FusedMHARunnerInt8v2() = default;
    FusedMHARunnerInt8v2(int32_t const numHeads, int32_t const headSize, int32_t const sm, int32_t const hasMask,
        float const qkvScale, float const qScaling);
    ~FusedMHARunnerInt8v2() = default; // for pImpl

    virtual void setup(int32_t const S, int32_t const B, int32_t const windowNum) override;

    void run(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    size_t getWorkspaceSize() const override;

    bool isValid(int32_t s) const override;

    int32_t getSFromMaxSeqLen(int32_t const maxSeqLen) override;

private:
    float mDqProbs{};
    float mScaleQkv{};
    float mScaleCtx{};
    int32_t mSm{};
    class mhaImpl;
    std::unique_ptr<mhaImpl> pImpl;
};

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class QKVToContextWithPosBiasPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    QKVToContextWithPosBiasPlugin(std::string name, nvinfer1::DataType type, int32_t hiddenSize, int32_t numHeads,
        int32_t hasMask, float qkvScale, float dqProbs);

    QKVToContextWithPosBiasPlugin(std::string const& name, void const* data, size_t length);

    // It doesn't make sense to make QKVToContextWithPosBiasPlugin without arguments, so we
    // delete default constructor.
    QKVToContextWithPosBiasPlugin() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    // IPluginV2 Methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

protected:
    void createMHARunner();

private:
    std::string const mLayerName{};
    std::string mNamespace{};

    std::unique_ptr<MHARunner> dispatcher;

    nvinfer1::DataType mType{};

    int32_t mS{};
    int32_t mB{};
    int32_t mW{};
    int32_t mSM{};
    int32_t mHeadSize{};
    int32_t mHiddenSize{};
    int32_t mNumHeads{};
    int32_t mHasMask{};
    float mQKVScale{};
    float mDqProbs{};
    int32_t mHdim{};
};

class QKVToContextWithPosBiasPluginCreator : public nvinfer1::IPluginCreator
{
public:
    QKVToContextWithPosBiasPluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace{};
};
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_QKV_TO_CONTEXT_WITH_POS_BIAS_PLUGIN_H
