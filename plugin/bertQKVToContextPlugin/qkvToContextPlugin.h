/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "common/cublasWrapper.h"
#include "mhaRunner.h"
#include "zeroPadding2d.h"
#include <math.h>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
namespace bert
{

std::pair<int32_t, int32_t> tuneBatchedGemm(
    const int32_t B, const int32_t S, const int32_t numHeads, const int32_t headSize);

template <typename T>
int32_t computeScaledSoftmax(cudaStream_t stream, const int32_t ld, const int32_t B, const int32_t N,
    float const rsqrtHeadSize, T const* input, T* output);

template <typename T>
int32_t computeMaskedScaledSoftmax(cudaStream_t stream, const int32_t ld, const int32_t B, const int32_t N,
    float const rsqrtHeadSize, int32_t const* maskIdx, T const* input, T* output);

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class QKVToContextPluginDynamic : public IPluginV3,
                                  public IPluginV3OneCore,
                                  public IPluginV3OneBuild,
                                  public IPluginV3OneRuntime
{
public:
    QKVToContextPluginDynamic(const std::string name, const nvinfer1::DataType type, const int32_t hiddenSize,
        const int32_t numHeads, float const dqProbs, bool hasImask = false);

    // constructor that also takes in MHARunner state
    // this constructor should only be called during runtime plugin creation after engine deserialization
    QKVToContextPluginDynamic(const std::string name, const DataType type, const int32_t S, const int32_t B,
        const int32_t SM, const int32_t hiddenSize, const int32_t numHeads, float const dqProbs, bool hasImask,
        bool hasUnfusedDispatcher, void const* runnerStateBuffer);

    // It doesn't make sense to make QKVToContextPluginDynamic without arguments, so we
    // delete default constructor.
    QKVToContextPluginDynamic() = delete;

    ~QKVToContextPluginDynamic() override;

    // IPluginV3 Methods
    // NOTE: since this is itself is an abstract class, the rest of virtual methods defined in its children classes
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;
    // end of IPluginV3 Methods

    // IPluginV3OneCore Methods
    char const* getPluginName() const noexcept override;

    char const* getPluginNamespace() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

    char const* getPluginVersion() const noexcept override;
    // end of IPluginV3OneCore Methods

    // IPluginV3Build Methods
    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getNbOutputs() const noexcept override;

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;
    // end IPluginV3Build Methods

    // IPluginV3Runtime Methods
    IPluginV3* clone() noexcept override;

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;

    PluginFieldCollection const* getFieldsToSerialize() noexcept override;
    // end IPluginV3Runtime Methods

    // setter for mCublasWrapper and mCublasHandle
    void setCublasResources(std::shared_ptr<CublasWrapper>);

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

    int32_t mS{};
    int32_t mB{};
    int32_t mSM{};
    int32_t mHeadSize{};
    int32_t mHiddenSize;
    int32_t mNumHeads{};
    int32_t mHasImask{};
    int32_t mHasUnfusedDispatcher{};
    nvinfer1::DataType mType{};
    float mDqProbs{};
    nvinfer1::pluginInternal::cublasHandle_t mCublasHandle{};
    // the wrapper pointer is shared among all plugins attached to the same context.
    std::shared_ptr<nvinfer1::pluginInternal::CublasWrapper> mCublasWrapper;

    // IPluginV3 serialization related
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
    std::vector<char> mRunnerStateBuffer; // memory management of this is handled automatically by class destructor
};

class QKVToContextPluginDynamicCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    QKVToContextPluginDynamicCreator();
    ~QKVToContextPluginDynamicCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;

    void setPluginNamespace(char const* libNamespace) noexcept;

    char const* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

class QKVToContextVarSeqlenPlugin : public IPluginV3,
                                    public IPluginV3OneCore,
                                    public IPluginV3OneBuild,
                                    public IPluginV3OneRuntime
{
public:
    QKVToContextVarSeqlenPlugin(std::string const name, nvinfer1::DataType const type, int32_t const hiddenSize,
        int32_t const numHeads, float const dqProbs, bool hasImask = false, bool varSeqlen = false,
        bool const useInt8ScaleMax = true);

    QKVToContextVarSeqlenPlugin(std::string const name, int32_t const s, int32_t b, nvinfer1::DataType const type,
        int32_t const hiddenSize, int32_t const numHeads, float const dqProbs, bool hasImask, bool varSeqlen,
        bool const useInt8ScaleMax, void const* runnerStateBuffer);

    // It doesn't make sense to make QKVToContextVarSeqlenPlugin without arguments, so we
    // delete default constructor.
    QKVToContextVarSeqlenPlugin() = delete;

    ~QKVToContextVarSeqlenPlugin() override;

    // IPluginV3 Methods
    // NOTE: since this is itself is an abstract class, the rest of virtual methods defined in its children classes
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;
    // end of IPluginV3 Methods

    // IPluginV3OneCore Methods
    char const* getPluginName() const noexcept override;

    char const* getPluginNamespace() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

    char const* getPluginVersion() const noexcept override;
    // end of IPluginV3OneCore Methods

    // IPluginV3Build Methods
    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getNbOutputs() const noexcept override;

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;
    // end IPluginV3Build Methods

    // IPluginV3Runtime Methods
    IPluginV3* clone() noexcept override;

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;

    PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    // setter for mCublasWrapper and mCublasHandle
    void setCublasResources(std::shared_ptr<CublasWrapper>);

protected:
    void createMHARunner();

private:
    const std::string mLayerName;
    std::string mNamespace;

    // Used for kernels with header size equals to 32.
    std::unique_ptr<MHARunner> mDispatcher;
    std::unique_ptr<QkvPaddingRunner> mPatcher;

    int32_t mS{};
    int32_t mB{};
    int32_t mSM{};
    int32_t mHeadSize{};
    int32_t mHiddenSize{};
    int32_t mNumHeads{};
    int32_t mHasImask{};
    nvinfer1::DataType mType{};

    float mDqProbs{};

    int32_t mHdim{};
    int32_t mUseVarSeqlen{};
    int32_t mUseInt8ScaleMax{true};
    nvinfer1::pluginInternal::cublasHandle_t mCublasHandle{};
    // the wrapper pointer is shared among all plugins attached to the same context.
    std::shared_ptr<nvinfer1::pluginInternal::CublasWrapper> mCublasWrapper;

    // serialization data structures
    std::vector<char> mRunnerStateBuffer; // memory management of this is handled automatically by class destructor
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class QKVToContextVarSeqlenPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    QKVToContextVarSeqlenPluginCreator();
    ~QKVToContextVarSeqlenPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;

    void setPluginNamespace(char const* libNamespace) noexcept;

    char const* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_QKV_TO_CONTEXT_PLUGIN_H

#endif // CUDA_VERSION >= 10010
