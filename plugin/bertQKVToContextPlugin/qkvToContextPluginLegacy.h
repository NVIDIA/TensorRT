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
    int32_t const B, int32_t const S, int32_t const numHeads, int32_t const headSize);

template <typename T>
int32_t computeScaledSoftmax(cudaStream_t stream, int32_t const ld, int32_t const B, int32_t const N,
    float const rsqrtHeadSize, T const* input, T* output);

template <typename T>
int32_t computeMaskedScaledSoftmax(cudaStream_t stream, int32_t const ld, int32_t const B, int32_t const N,
    float const rsqrtHeadSize, int32_t const* maskIdx, T const* input, T* output);

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class QKVToContextPluginDynamicLegacy : public nvinfer1::IPluginV2DynamicExt
{
public:
    QKVToContextPluginDynamicLegacy(std::string const name, nvinfer1::DataType const type, int32_t const hiddenSize,
        int32_t const numHeads, float const dqProbs, bool hasImask = false);

    QKVToContextPluginDynamicLegacy(std::string const name, void const* data, size_t length);

    // It doesn't make sense to make QKVToContextPluginDynamicLegacy without arguments, so we
    // delete default constructor.
    QKVToContextPluginDynamicLegacy() = delete;

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
    void attachToContext(
        cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept override;

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
    std::string const mLayerName;
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
    bool mHasImask{};
    nvinfer1::DataType mType{};
    float mDqProbs{};
    nvinfer1::pluginInternal::cublasHandle_t mCublas{};
    // the wrapper pointer is shared among all plugins attached to the same context.
    std::shared_ptr<nvinfer1::pluginInternal::CublasWrapper> mCublasWrapper;

    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2::enqueue;
    using IPluginV2Ext::configurePlugin;
};

class QKVToContextPluginDynamicLegacyCreator : public nvinfer1::IPluginCreator
{
public:
    QKVToContextPluginDynamicLegacyCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

class QKVToContextVarSeqlenPluginLegacy : public nvinfer1::IPluginV2DynamicExt
{
public:
    QKVToContextVarSeqlenPluginLegacy(std::string const name, nvinfer1::DataType const type, int32_t const hiddenSize,
        int32_t const numHeads, float const dqProbs, bool hasImask = false, bool varSeqlen = false,
        bool const useInt8ScaleMax = true);

    QKVToContextVarSeqlenPluginLegacy(std::string const name, void const* data, size_t length);

    // It doesn't make sense to make QKVToContextVarSeqlenPluginLegacy without arguments, so we
    // delete default constructor.
    QKVToContextVarSeqlenPluginLegacy() = delete;

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
    void attachToContext(
        cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept override;

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
    std::string const mLayerName;
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
    bool mHasImask{};
    nvinfer1::DataType mType{};

    float mDqProbs{};

    int32_t mHdim{};
    bool mUseVarSeqlen{};
    bool mUseInt8ScaleMax{true};
    nvinfer1::pluginInternal::cublasHandle_t mCublas{};
    // the wrapper pointer is shared among all plugins attached to the same context.
    std::shared_ptr<nvinfer1::pluginInternal::CublasWrapper> mCublasWrapper;

    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2::enqueue;
    using IPluginV2Ext::configurePlugin;
};

class QKVToContextVarSeqlenPluginLegacyCreator : public nvinfer1::IPluginCreator
{
public:
    QKVToContextVarSeqlenPluginLegacyCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

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
