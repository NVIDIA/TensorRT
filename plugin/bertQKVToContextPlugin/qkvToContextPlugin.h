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

// Need 10.1 for cublasGemmStridedBatchedEx
#include <cuda.h>
#if CUDA_VERSION >= 10010

#ifndef TRT_QKV_TO_CONTEXT_PLUGIN_H
#define TRT_QKV_TO_CONTEXT_PLUGIN_H

#include "NvInferPlugin.h"
#include "cublas_v2.h"
#include <string>
#include <vector>

namespace bert
{

// Multi Head Attention runner
class MHARunner
{
public:
    MHARunner(const nvinfer1::DataType type, const int numHeads, const int headSize)
        : mType(type)
        , mNumHeads(numHeads)
        , mHeadSize(headSize)
        , mWordSize(getElementSize(type))
        , mRsqrtHeadSize(1.f / sqrtf(headSize))
    {
    }

    virtual ~MHARunner() = default;

    virtual void setup(const int S, const int B)
    {
        assert(S);
        assert(B);
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

    virtual size_t getSerializationSize() const;
    virtual void serialize(void* buffer) const;
    virtual void deserialize(const void* data, size_t length);

    virtual size_t getWorkspaceSize() const = 0;

    virtual bool isValid() const = 0;

protected:
    nvinfer1::DataType mType;

    int mS;
    int mB;
    int mOmatSize;
    int mNumMats;
    int mNumHeads;
    int mHeadSize;
    int mWordSize;
    int mLdQKV;
    int mStrideQKV;
    int mLdOut;
    int mStrideOut;

    float mRsqrtHeadSize;
};

std::pair<int, int> tuneBatchedGemm(const int B, const int S, const int numHeads, const int headSize);

template <typename T>
int computeScaledSoftmax(
    cudaStream_t stream, const int ld, const int B, const int N, const float rsqrtHeadSize, const T* input, T* output);

template <typename T>
int computeMaskedScaledSoftmax(cudaStream_t stream, const int ld, const int B, const int N, const float rsqrtHeadSize,
    const int* maskIdx, const T* input, T* output);

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class QKVToContextPluginDynamic : public nvinfer1::IPluginV2DynamicExt
{
public:
    QKVToContextPluginDynamic(const std::string name, const nvinfer1::DataType type, const int hiddenSize,
        const int numHeads, const float dqProbs, bool hasImask = false);

    QKVToContextPluginDynamic(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make QKVToContextPluginDynamic without arguments, so we
    // delete default constructor.
    QKVToContextPluginDynamic() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const override;
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    // IPluginV2 Methods
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    int getNbOutputs() const override;
    int initialize() override;
    void terminate() override;
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void destroy() override;
    void setPluginNamespace(const char* pluginNamespace) override;
    const char* getPluginNamespace() const override;

protected:
    void createMHARunner();
    int getSMVersion() const;

private:
    const std::string mLayerName;
    std::string mNamespace;

    std::unique_ptr<MHARunner> dispatcher;

    int mS;
    int mB;
    int mSM;
    int mHeadSize;
    int mHiddenSize;
    int mNumHeads;
    bool mHasImask;
    nvinfer1::DataType mType;

    float mDqProbs;

protected:
    // To prevent compiler warnings.
    using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::configurePlugin;
    using nvinfer1::IPluginV2DynamicExt::enqueue;
    using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
    using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
    using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::supportsFormat;
};

class QKVToContextPluginDynamicCreator : public nvinfer1::IPluginCreator
{
public:
    QKVToContextPluginDynamicCreator();

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const nvinfer1::PluginFieldCollection* getFieldNames() override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

class UnfusedMHARunner : public MHARunner
{
public:
    UnfusedMHARunner(const nvinfer1::DataType type, const int numHeads, const int headSize);
    virtual ~UnfusedMHARunner();

    virtual void setup(const int S, const int B) override;

    void run(const nvinfer1::PluginTensorDesc& inputDesc, const nvinfer1::PluginTensorDesc& outputDesc,
        const void* qkvPtr, const void* maskPtr, void* output, void* workspace, cudaStream_t stream) override;

    size_t getWorkspaceSize() const override;

    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void deserialize(const void* data, size_t length) override;
    bool isValid() const override;

private:
    int mAlgoBatchedEx1;
    int mAlgoBatchedEx2;
    cublasHandle_t mCublas;
};

class FusedMHARunnerFP16 : public MHARunner
{
public:
    FusedMHARunnerFP16(const int numHeads, const int headSize, const int sm);
    ~FusedMHARunnerFP16() = default; // for pimpl

    virtual void setup(const int S, const int B) override;

    void run(const nvinfer1::PluginTensorDesc& inputDesc, const nvinfer1::PluginTensorDesc& outputDesc,
        const void* qkvPtr, const void* maskPtr, void* output, void* workspace, cudaStream_t stream) override;

    size_t getWorkspaceSize() const override;

    void deserialize(const void* data, size_t length) override;

    bool isValid() const override;

private:
    int mSm;
    class mhaImpl;
    std::unique_ptr<mhaImpl> pimpl;
};

class FusedMHARunnerInt8 : public MHARunner
{
public:
    FusedMHARunnerInt8(const int numHeads, const int headSize, const int sm, const float dqProbs);
    ~FusedMHARunnerInt8() = default; // for pimpl

    virtual void setup(const int S, const int B) override;

    void run(const nvinfer1::PluginTensorDesc& inputDesc, const nvinfer1::PluginTensorDesc& outputDesc,
        const void* qkvPtr, const void* maskPtr, void* output, void* workspace, cudaStream_t stream) override;

    size_t getWorkspaceSize() const override;

    void deserialize(const void* data, size_t length) override;

    bool isValid() const override;

private:
    float mDqProbs;
    int mSm;
    class mhaImpl;
    std::unique_ptr<mhaImpl> pimpl;
};

} // namespace bert
#endif // TRT_QKV_TO_CONTEXT_PLUGIN_H

#endif // CUDA_VERSION >= 10010
