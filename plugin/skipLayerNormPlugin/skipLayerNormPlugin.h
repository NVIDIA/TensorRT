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

#include <cuda.h>
#if CUDA_VERSION >= 10010

#ifndef TRT_SKIP_LAYER_NORM_PLUGIN_H
#define TRT_SKIP_LAYER_NORM_PLUGIN_H

#include "NvInferPlugin.h"

#include "common/bertCommon.h"
#include <memory>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
namespace bert
{
template <bool hasBias>
int computeSkipLayerNormDQQ(cudaStream_t stream, const int ld, const int n, const int8_t* input, const int8_t* skip,
    const __half* beta, const __half* gamma, int8_t* output, const __half* bias, const float dqScaleIn,
    const float dqScaleSkip, const float qScale);

template <typename T, bool hasBias>
int computeSkipLayerNorm(cudaStream_t stream, const int ld, const int n, const T* input, const T* skip, const T* beta,
    const T* gamma, T* output, const T* bias);


class SkipLayerNormPluginDynamic : public nvinfer1::IPluginV2DynamicExt
{
public:
    SkipLayerNormPluginDynamic(const std::string name, const nvinfer1::DataType type, const int ld, const nvinfer1::Weights& beta,
        const nvinfer1::Weights& gamma, const nvinfer1::Weights& bias);

    SkipLayerNormPluginDynamic(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make SkipLayerNormPluginDynamic without arguments, so we
    // delete default constructor.
    SkipLayerNormPluginDynamic() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
        noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    const std::string mLayerName;
    std::string mNamespace;

    bert::cuda_unique_ptr<void> mGammaDev;
    bert::cuda_unique_ptr<void> mBetaDev;
    size_t mLd; // leading dim
    bert::WeightsWithOwnership mGamma;
    bert::WeightsWithOwnership mBeta;
    nvinfer1::DataType mType;
    nvinfer1::DataType mCfgType;

    bool mHasBias;
    bert::cuda_unique_ptr<void> mBiasDev;
    bert::WeightsWithOwnership mBias;

    size_t mParamWordsize;

    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2::enqueue;
    using IPluginV2Ext::configurePlugin;
};

class SkipLayerNormPluginDynamicCreator : public nvinfer1::IPluginCreator
{
public:
    SkipLayerNormPluginDynamicCreator();

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

class SkipLayerNormVarSeqlenPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    SkipLayerNormVarSeqlenPlugin(const std::string name, const nvinfer1::DataType type, const nvinfer1::Weights& beta,
        const nvinfer1::Weights& gamma, const nvinfer1::Weights& bias);

    SkipLayerNormVarSeqlenPlugin(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make SkipLayerNormVarSeqlenPlugin without arguments, so we
    // delete default constructor.
    SkipLayerNormVarSeqlenPlugin() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
        noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    const std::string mLayerName;
    std::string mNamespace;

    bert::cuda_unique_ptr<void> mGammaDev;
    bert::cuda_unique_ptr<void> mBetaDev;
    size_t mLd; // leading dim
    bert::WeightsWithOwnership mGamma;
    bert::WeightsWithOwnership mBeta;
    nvinfer1::DataType mType;
    nvinfer1::DataType mCfgType;

    bool mHasBias;
    bert::cuda_unique_ptr<void> mBiasDev;
    bert::WeightsWithOwnership mBias;

    size_t mParamWordsize;

    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2::enqueue;
    using IPluginV2Ext::configurePlugin;
};

class SkipLayerNormVarSeqlenPluginCreator : public nvinfer1::IPluginCreator
{
public:
    SkipLayerNormVarSeqlenPluginCreator();

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

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_SKIP_LAYER_NORM_PLUGIN_H

#endif // CUDA_VERSION >= 10010
