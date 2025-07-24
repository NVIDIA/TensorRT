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

#ifndef TRT_GROUP_NORM_PLUGIN_H
#define TRT_GROUP_NORM_PLUGIN_H

#include "common/plugin.h"
#include <string>
#include <vector>

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.
namespace nvinfer1
{
namespace plugin
{

template <typename T>
cudaError_t scaleShiftChannelsInplace(T* inOut, int32_t const B, int32_t const C, int32_t const channelVolume,
    float const* beta, float const* gamma, cudaStream_t stream);

class GroupNormalizationPlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    GroupNormalizationPlugin(float epsilon, int32_t const nbGroups);

    GroupNormalizationPlugin(void const* data, size_t length);

    // It doesn't make sense to make GroupNormalizationPlugin without arguments, so we
    // delete default constructor.
    GroupNormalizationPlugin() = delete;

    int32_t getNbOutputs() const noexcept override;

    // DynamicExt plugins returns DimsExprs class instead of Dims
    DimsExprs getOutputDimensions(int32_t index, nvinfer1::DimsExprs const* inputs, int32_t nbInputDims,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void destroy() noexcept override;

    DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    void attachToContext(
        cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept override;

    void detachFromContext() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

private:
    std::string mNamespace;

    float mEpsilon;
    int32_t mNbGroups;
    int32_t mChannelVolume;

    nvinfer1::pluginInternal::cudnnHandle_t mCudnnHandle{};
    // the wrapper pointer is shared among all plugins attached to the same context.
    std::shared_ptr<nvinfer1::pluginInternal::CudnnWrapper> mCudnnWrapper;

    // Describes input and output.
    nvinfer1::pluginInternal::cudnnTensorDescriptor_t mTensorDesc{};
    nvinfer1::pluginInternal::cudnnTensorDescriptor_t mBNTensorDesc{};

    // These are buffers initialized to 1 and 0 respectively
    std::shared_ptr<CudaBind<float>> mBnScales{};
    std::shared_ptr<CudaBind<float>> mBnBias{};
    size_t mNbScaleBias{};

    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2::enqueue;
    using IPluginV2Ext::configurePlugin;
};

class GroupNormalizationPluginCreator : public IPluginCreator
{
public:
    GroupNormalizationPluginCreator();

    ~GroupNormalizationPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

private:
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_GROUP_NORM_PLUGIN_H
