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

#ifndef TRT_SKIP_LAYER_NORM_INTERLEAVED_PLUGIN_H
#define TRT_SKIP_LAYER_NORM_INTERLEAVED_PLUGIN_H
#include <cuda.h>
#include "NvInferPlugin.h"

#include "bertCommon.h"
#include <memory>
#include <string>
#include <vector>

namespace bert
{

class SkipLayerNormInterleavedPluginBase : public nvinfer1::IPluginV2DynamicExt
{
public:
    SkipLayerNormInterleavedPluginBase(
        const std::string name, const nvinfer1::Weights& beta, const nvinfer1::Weights& gamma);

    SkipLayerNormInterleavedPluginBase(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make SkipLayerNormInterleavedPlugin without arguments, so we
    // delete default constructor.
    SkipLayerNormInterleavedPluginBase() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const
        noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

protected:
    const std::string mLayerName;
    std::string mNamespace;

    bert::cuda_unique_ptr<void> mGammaDev;
    bert::cuda_unique_ptr<void> mBetaDev;
    size_t mLd; // leading dim
    bert::WeightsWithOwnership mGamma;
    bert::WeightsWithOwnership mBeta;

    size_t mParamWordsize;
    bool mParamsOnDevice;
};

class SkipLayerNormInterleavedPluginHFace : public SkipLayerNormInterleavedPluginBase
{
public:
    SkipLayerNormInterleavedPluginHFace(
        const std::string name, const nvinfer1::Weights& beta, const nvinfer1::Weights& gamma);

    SkipLayerNormInterleavedPluginHFace(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make SkipLayerNormInterleavedPlugin without arguments, so we
    // delete default constructor.
    SkipLayerNormInterleavedPluginHFace() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2 Methods
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;
    const char* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
};

class SkipLayerNormInterleavedPluginMTron : public SkipLayerNormInterleavedPluginBase
{
public:
    SkipLayerNormInterleavedPluginMTron(
        const std::string name, const nvinfer1::Weights& beta, const nvinfer1::Weights& gamma);

    SkipLayerNormInterleavedPluginMTron(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make SkipLayerNormInterleavedPlugin without arguments, so we
    // delete default constructor.
    SkipLayerNormInterleavedPluginMTron() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2 Methods
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;
    const char* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
};

class SkipLayerNormInterleavedPluginBaseCreator : public nvinfer1::IPluginCreator
{
public:
    SkipLayerNormInterleavedPluginBaseCreator();

    const char* getPluginName() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

class SkipLayerNormInterleavedPluginHFaceCreator : public SkipLayerNormInterleavedPluginBaseCreator
{
public:
    SkipLayerNormInterleavedPluginHFaceCreator();

    const char* getPluginVersion() const noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;
};

class SkipLayerNormInterleavedPluginMTronCreator : public SkipLayerNormInterleavedPluginBaseCreator
{
public:
    SkipLayerNormInterleavedPluginMTronCreator();

    const char* getPluginVersion() const noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;
};

} // namespace bert
#endif // TRT_SKIP_LAYER_NORM_INTERLEAVED_PLUGIN_H
