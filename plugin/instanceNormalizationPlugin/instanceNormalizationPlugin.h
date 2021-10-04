/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#ifndef TRT_INSTANCE_NORMALIZATION_PLUGIN_H
#define TRT_INSTANCE_NORMALIZATION_PLUGIN_H
#include "instanceNormFwd.h"
#include "plugin.h"
#include "serialize.hpp"
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
using namespace instance_norm_impl;
class InstanceNormalizationPlugin final : public nvinfer1::IPluginV2DynamicExt
{

public:
    InstanceNormalizationPlugin(float epsilon, nvinfer1::Weights const& scale, nvinfer1::Weights const& bias,
        int32_t relu = 0, float alpha = 0.f);
    InstanceNormalizationPlugin(float epsilon, const std::vector<float>& scale, const std::vector<float>& bias,
        int32_t relu = 0, float alpha = 0.f);
    InstanceNormalizationPlugin(void const* serialData, size_t serialLength);

    InstanceNormalizationPlugin() = delete;

    ~InstanceNormalizationPlugin() override;

    int32_t getNbOutputs() const noexcept override;

    // DynamicExt plugins returns DimsExprs class instead of Dims
    using nvinfer1::IPluginV2::getOutputDimensions;
    DimsExprs getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    using nvinfer1::IPluginV2::getWorkspaceSize;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override;

    using nvinfer1::IPluginV2::enqueue;
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    // DynamicExt plugin supportsFormat update.
    bool supportsFormatCombination(
        int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const
        noexcept override;

    void attachToContext(
        cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept override;

    void detachFromContext() noexcept override;

    using nvinfer1::IPluginV2Ext::configurePlugin;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override;

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

    // NDHWC implementation
    InstanceNormFwdContext mContext;
};

class InstanceNormalizationPluginCreator : public BaseCreator
{
public:
    InstanceNormalizationPluginCreator();

    ~InstanceNormalizationPluginCreator() override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_INSTANCE_NORMALIZATION_PLUGIN_H
