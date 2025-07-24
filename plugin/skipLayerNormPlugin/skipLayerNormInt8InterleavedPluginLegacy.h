/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef TRT_SKIP_LAYER_NORM_INTERLEAVED_PLUGIN_LEGACY_H
#define TRT_SKIP_LAYER_NORM_INTERLEAVED_PLUGIN_LEGACY_H
#include "NvInferPlugin.h"
#include <cuda.h>

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

int32_t launch_small_hface(cudaStream_t stream, int32_t const ld, int32_t const total, int8_t const* input,
    int8_t const* skip, half const* beta, half const* gamma, int8_t* output, float const dqScaleIn,
    float const dqScaleSkip, float const qScale);

int32_t launch_large_hface(cudaStream_t stream, int32_t const ld, int32_t const total, int8_t const* input,
    int8_t const* skip, half const* beta, half const* gamma, int8_t* output, float const dqScaleIn,
    float const dqScaleSkip, float const qScale);

int32_t launch_small_mtron(cudaStream_t stream, int32_t const ld, int32_t const total, int8_t const* input,
    int8_t const* skip, half const* beta, half const* gamma, int8_t* output, int8_t* preln, float const dqScaleIn,
    float const dqScaleSkip, float const qScale, float const qSkipScale);

int32_t launch_large_mtron(cudaStream_t stream, int32_t const ld, int32_t const total, int8_t const* input,
    int8_t const* skip, half const* beta, half const* gamma, int8_t* output, int8_t* preln, float const dqScaleIn,
    float const dqScaleSkip, float const qScale, float const qSkipScale);

class SkipLayerNormInterleavedPluginBaseLegacy : public nvinfer1::IPluginV2DynamicExt
{
public:
    SkipLayerNormInterleavedPluginBaseLegacy(
        std::string const& name, nvinfer1::Weights const& beta, nvinfer1::Weights const& gamma);

    SkipLayerNormInterleavedPluginBaseLegacy(std::string const& name, void const* data, size_t length);

    // It doesn't make sense to make SkipLayerNormInterleavedPlugin without
    // arguments, so we delete default constructor.
    SkipLayerNormInterleavedPluginBaseLegacy() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    // IPluginV2 Methods
    char const* getPluginType() const noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

protected:
    std::string const& mLayerName;
    std::string mNamespace;

    bert::cuda_unique_ptr<void> mGammaDev;
    bert::cuda_unique_ptr<void> mBetaDev;
    size_t mLd{}; // leading dim
    bert::WeightsWithOwnership mGamma;
    bert::WeightsWithOwnership mBeta;

    size_t mParamWordsize{};
    bool mParamsOnDevice{};
};

class SkipLayerNormInterleavedPluginHFaceLegacy : public SkipLayerNormInterleavedPluginBaseLegacy
{
public:
    SkipLayerNormInterleavedPluginHFaceLegacy(
        std::string const& name, nvinfer1::Weights const& beta, nvinfer1::Weights const& gamma);

    SkipLayerNormInterleavedPluginHFaceLegacy(std::string const& name, void const* data, size_t length);

    // It doesn't make sense to make SkipLayerNormInterleavedPlugin without
    // arguments, so we delete default constructor.
    SkipLayerNormInterleavedPluginHFaceLegacy() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2 Methods
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
};

class SkipLayerNormInterleavedPluginMTronLegacy : public SkipLayerNormInterleavedPluginBaseLegacy
{
public:
    SkipLayerNormInterleavedPluginMTronLegacy(
        std::string const& name, nvinfer1::Weights const& beta, nvinfer1::Weights const& gamma);

    SkipLayerNormInterleavedPluginMTronLegacy(std::string const& name, void const* data, size_t length);

    // It doesn't make sense to make SkipLayerNormInterleavedPlugin without
    // arguments, so we delete default constructor.
    SkipLayerNormInterleavedPluginMTronLegacy() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2 Methods
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
};

class SkipLayerNormInterleavedPluginBaseLegacyCreator : public nvinfer1::IPluginCreator
{
public:
    SkipLayerNormInterleavedPluginBaseLegacyCreator();

    char const* getPluginName() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

class SkipLayerNormInterleavedPluginHFaceLegacyCreator : public SkipLayerNormInterleavedPluginBaseLegacyCreator
{
public:
    SkipLayerNormInterleavedPluginHFaceLegacyCreator();

    char const* getPluginVersion() const noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;
};

class SkipLayerNormInterleavedPluginMTronLegacyCreator : public SkipLayerNormInterleavedPluginBaseLegacyCreator
{
public:
    SkipLayerNormInterleavedPluginMTronLegacyCreator();

    char const* getPluginVersion() const noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;
};

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_SKIP_LAYER_NORM_INTERLEAVED_PLUGIN_LEGACY_H
