/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_GROUPNORM_PLUGIN_H
#define TRT_GROUPNORM_PLUGIN_H

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "common/plugin.h"
#include "groupNormPluginCommon.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
class GroupNormPlugin : public IPluginV2DynamicExt
{
public:
    GroupNormPlugin() = delete;
    GroupNormPlugin(std::string const& name, float epsilon, int32_t bSwish);
    GroupNormPlugin(std::string const& name, void const* buffer, size_t length);
    ~GroupNormPlugin() override = default;

    GroupNormPlugin(const GroupNormPlugin& /*other*/) = default;
    GroupNormPlugin& operator=(const GroupNormPlugin& /*other*/) = delete;
    GroupNormPlugin(GroupNormPlugin&& /*other*/) noexcept = delete;
    GroupNormPlugin& operator=(GroupNormPlugin&& /*other*/) noexcept = delete;

    // Methods inherited from IPluginV2
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

    // Method inherited from IPluginV2Ext
    DataType getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    // Methods inherited from IPluginV2DynamicExt
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
    size_t getWorkspaceSizeInBytes() const;

    std::string mName;
    std::string mNameSpace;

    float mEpsilon{};
    int32_t mBSwish{};
    GroupNormNHWCParams mParams;
};

class GroupNormPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    GroupNormPluginCreator();
    ~GroupNormPluginCreator();

    GroupNormPluginCreator(const GroupNormPluginCreator& /*other*/) = delete;
    GroupNormPluginCreator& operator=(const GroupNormPluginCreator& /*other*/) = delete;
    GroupNormPluginCreator(GroupNormPluginCreator&& /*other*/) noexcept = delete;
    GroupNormPluginCreator& operator=(GroupNormPluginCreator&& /*other*/) noexcept = delete;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    PluginFieldCollection const* getFieldNames() noexcept override;
    IPluginV2* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;
    IPluginV2* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNameSpace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_GROUPNORM_PLUGIN_H
