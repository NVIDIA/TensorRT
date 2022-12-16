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

#ifndef TRT_GROUP_NORM_PLUGIN_H
#define TRT_GROUP_NORM_PLUGIN_H

#include "common/plugin.h"
#include "common/serialize.hpp"
#include <cudnn.h>
#include <iostream>
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
cudaError_t scaleShiftChannelsInplace(T* inOut, const int B, const int C, const int channelVolume, const float* beta,
    const float* gamma, cudaStream_t stream);

class GroupNormalizationPlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    GroupNormalizationPlugin(float epsilon, const int nbGroups);

    GroupNormalizationPlugin(const void* data, size_t length);

    // It doesn't make sense to make GroupNormalizationPlugin without arguments, so we
    // delete default constructor.
    GroupNormalizationPlugin() = delete;

    int getNbOutputs() const noexcept override;

    // DynamicExt plugins returns DimsExprs class instead of Dims
    DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs, int nbInputDims,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void destroy() noexcept override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    void attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept override;

    void detachFromContext() noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

private:
    const char* mPluginNamespace;
    std::string mNamespace;

    float mEpsilon;
    int mNbGroups;
    int mChannelVolume;

    cudnnHandle_t _cudnn_handle;
    // Describes input and output.
    cudnnTensorDescriptor_t desc;
    cudnnTensorDescriptor_t bnDesc;
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

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_GROUP_NORM_PLUGIN_H
