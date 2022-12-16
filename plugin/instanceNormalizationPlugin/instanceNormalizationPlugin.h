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
#ifndef TRT_INSTANCE_NORMALIZATION_PLUGIN_H
#define TRT_INSTANCE_NORMALIZATION_PLUGIN_H
#include "common/plugin.h"
#include "common/serialize.hpp"
#include "instanceNormalizationPlugin/instanceNormFwd.h"
#include <cuda_fp16.h>
#include <cudnn.h>
#include <iostream>
#include <string>
#include <vector>

typedef unsigned short half_type;

namespace nvinfer1
{
namespace plugin
{
class InstanceNormalizationPlugin : public nvinfer1::IPluginV2DynamicExt
{

public:
    InstanceNormalizationPlugin(float epsilon, nvinfer1::Weights const& scale, nvinfer1::Weights const& bias,
        int32_t relu = 0, float alpha = 0.f);
    InstanceNormalizationPlugin(float epsilon, std::vector<float> const& scale, std::vector<float> const& bias,
        int32_t relu = 0, float alpha = 0.f);
    InstanceNormalizationPlugin(void const* serialData, size_t serialLength);

    InstanceNormalizationPlugin() = delete;

    ~InstanceNormalizationPlugin() override;

    int32_t getNbOutputs() const noexcept override;

    // DynamicExt plugins returns DimsExprs class instead of Dims
    using nvinfer1::IPluginV2::getOutputDimensions;
    DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    using nvinfer1::IPluginV2::getWorkspaceSize;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    using nvinfer1::IPluginV2::enqueue;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    // DynamicExt plugin supportsFormat update.
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const
        noexcept override;

    void attachToContext(
        cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept override;

    void detachFromContext() noexcept override;

    using nvinfer1::IPluginV2Ext::configurePlugin;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

protected:
    template <class PluginType>
    nvinfer1::IPluginV2DynamicExt* cloneBase() const noexcept;

private:
    float mEpsilon;
    float mAlpha;
    int32_t mRelu;
    int32_t mNchan;
    std::vector<float> mHostScale;
    std::vector<float> mHostBias;
    float* mDeviceScale{nullptr};
    float* mDeviceBias{nullptr};
    cudnnHandle_t mCudnnHandle{nullptr};
    cudnnTensorDescriptor_t mXDescriptor{nullptr};
    cudnnTensorDescriptor_t mYDescriptor{nullptr};
    cudnnTensorDescriptor_t mBDescriptor{nullptr};
    std::string mPluginNamespace;
    std::string mNamespace;
    bool mInitialized{false};
    int32_t mCudaDriverVersion{-1};

    // NDHWC implementation
    instance_norm_impl::InstanceNormFwdContext mContext;
};

class InstanceNormalizationPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    InstanceNormalizationPluginCreator();

    ~InstanceNormalizationPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(char const* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

protected:
    template <class PluginType>
    IPluginV2DynamicExt* createPluginBase(char const* name, const nvinfer1::PluginFieldCollection* fc) noexcept;

    template <class PluginType>
    IPluginV2DynamicExt* deserializePluginBase(char const* name, void const* serialData, size_t serialLength) noexcept;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

// For backward compatibility, create version "2" of the identical plugin.
// Background: in TRT 8.0, we added 3D InstanceNorm plugin as the version 2 of the "InstanceNormalization_TRT" plugin.
// However, in TRT 8.2, we have fused it into version 1, so a separate version 2 is no longer needed, but is only kept
// for backward compatibility.
class InstanceNormalizationPluginV2 final : public InstanceNormalizationPlugin
{
public:
    InstanceNormalizationPluginV2(float epsilon, nvinfer1::Weights const& scale, nvinfer1::Weights const& bias,
        int32_t relu = 0, float alpha = 0.f)
        : InstanceNormalizationPlugin(epsilon, scale, bias, relu, alpha)
    {
    }
    InstanceNormalizationPluginV2(float epsilon, std::vector<float> const& scale, std::vector<float> const& bias,
        int32_t relu = 0, float alpha = 0.f)
        : InstanceNormalizationPlugin(epsilon, scale, bias, relu, alpha)
    {
    }
    InstanceNormalizationPluginV2(void const* serialData, size_t serialLength)
        : InstanceNormalizationPlugin(serialData, serialLength)
    {
    }
    InstanceNormalizationPluginV2() = delete;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
};

class InstanceNormalizationPluginCreatorV2 final : public InstanceNormalizationPluginCreator
{
public:
    char const* getPluginVersion() const noexcept override;
    IPluginV2DynamicExt* createPlugin(char const* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    IPluginV2DynamicExt* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_INSTANCE_NORMALIZATION_PLUGIN_H
