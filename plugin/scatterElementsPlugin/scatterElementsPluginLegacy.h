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

#ifndef TRT_SCATTER_ELEMENTS_PLUGIN_LEGACY_H
#define TRT_SCATTER_ELEMENTS_PLUGIN_LEGACY_H

#include "common/plugin.h"
#include "scatterElementsCommon.h"

namespace nvinfer1
{
namespace plugin
{

class ScatterElementsPluginV2 final : public nvinfer1::IPluginV2DynamicExt
{
public:
    ScatterElementsPluginV2() = delete;
    ScatterElementsPluginV2(ScatterElementsPluginV2 const&) = delete;
    ScatterElementsPluginV2(std::string const&, int32_t);
    ScatterElementsPluginV2(ReductionType, int32_t);
    ScatterElementsPluginV2(void const* serialData, size_t serialLength);
    ~ScatterElementsPluginV2() override = default;

    // IPluginV2 methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* libNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setClipParam(bool clip) noexcept;
    void setScoreBits(int32_t scoreBits) noexcept;
    void setCaffeSemantics(bool caffeSemantics) noexcept;

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    // IPluginV2DynamicExt methods
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

private:
    ReductionType mReduction;
    int32_t mAxis;
    std::string mNamespace;

    static constexpr int32_t kINDICES_TENSOR_IDX = 1;
    static constexpr int32_t kUPDATES_TENSOR_IDX = 2;
    static constexpr int32_t kDATA_TENSOR_IDX = 0;
    // outputs
    static constexpr int32_t kOUTPUT_TENSOR_IDX = 0;
};

class ScatterElementsPluginV2Creator : public nvinfer1::IPluginCreator
{
public:
    ScatterElementsPluginV2Creator();

    ~ScatterElementsPluginV2Creator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2DynamicExt* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2DynamicExt* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection gFC;
    std::vector<nvinfer1::PluginField> gPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_SCATTER_ELEMENTS_PLUGIN_LEGACY_H
