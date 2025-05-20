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

/*
 * V3 version of the plugin using IPluginV3 interfaces.
 * This implementation follows TensorRT's plugin V3 API.
 */

#ifndef TRT_MULTISCALE_DEFORMABLE_ATTN_PLUGIN_H
#define TRT_MULTISCALE_DEFORMABLE_ATTN_PLUGIN_H

// Standard library includes
#include <memory>
#include <string>
#include <vector>

#include "NvInferPlugin.h"

// TensorRT includes
#include "common/plugin.h"

namespace nvinfer1
{
namespace plugin
{

// Forward declarations
class MultiscaleDeformableAttnPlugin;
class MultiscaleDeformableAttnPluginCreator;

// V3 Plugin implementation
class MultiscaleDeformableAttnPlugin : public IPluginV3,
                                       public IPluginV3OneCore,
                                       public IPluginV3OneBuild,
                                       public IPluginV3OneRuntime
{
public:
    // Constructors/destructors
    MultiscaleDeformableAttnPlugin();
    ~MultiscaleDeformableAttnPlugin() = default;

    // IPluginV3 methods
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;

    // IPluginV3OneCore methods
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    int32_t getNbOutputs() const noexcept override;
    IPluginV3* clone() noexcept override;

    // IPluginV3OneBuild methods
    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;
    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    // IPluginV3OneRuntime methods
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;
    int32_t onShapeChange(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override;

private:
    // Serialization helpers
    std::vector<PluginField> mDataToSerialize;
    PluginFieldCollection mFCToSerialize;

    // Plugin namespace
    std::string mNamespace;
};

// Plugin creator class
class MultiscaleDeformableAttnPluginCreator : public IPluginCreatorV3One
{
public:
    // Constructor
    MultiscaleDeformableAttnPluginCreator();

    // IPluginCreatorV3One methods
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    PluginFieldCollection const* getFieldNames() noexcept override;
    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    char const* getPluginNamespace() const noexcept override;

private:
    // Plugin fields and namespace
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_MULTISCALE_DEFORMABLE_ATTN_PLUGIN_H
