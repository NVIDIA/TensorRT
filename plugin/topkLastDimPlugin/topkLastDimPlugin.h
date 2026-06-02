/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_TOPK_LAST_DIM_PLUGIN_H
#define TRT_TOPK_LAST_DIM_PLUGIN_H

#include "NvInfer.h"
#include "common/plugin.h"
#include <cstdint>
#include <string>
#include <vector>

namespace nvinfer1::plugin
{

class TopkLastDimPlugin : public IPluginV3,
                          public IPluginV3OneCore,
                          public IPluginV3OneBuild,
                          public IPluginV3OneRuntime
{
public:
    //! \param axis Axis along which to compute top-k. -1 means last dimension.
    TopkLastDimPlugin(int32_t typeId, int32_t k, int32_t isLargest, int32_t axis = -1);
    TopkLastDimPlugin(TopkLastDimPlugin const&) = default;
    ~TopkLastDimPlugin() override = default;

    // IPluginV3
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;
    IPluginV3* clone() noexcept override;

    // IPluginV3OneCore
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // IPluginV3OneBuild
    int32_t getNbOutputs() const noexcept override;
    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    // IPluginV3OneRuntime
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;
    PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    //! Resolve mAxis to a non-negative value given the input rank.
    int32_t resolveAxis(int32_t nbDims) const;

    //! Compute the element size for the configured data type.
    int32_t elementSize() const;

    //! Compute workspace needed for transpose buffers (0 when axis == last dim).
    size_t transposeWorkspaceSize(Dims const& inputDims) const;

    //! Compute workspace needed by the AIR TopK kernel itself.
    size_t topkKernelWorkspaceSize(int32_t numRows, int32_t rowLength) const;

    template <typename T>
    int32_t enqueueImpl(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    int32_t mTypeId; //!< DataType stored as int32_t for field-based serialization.
    int32_t mK;
    int32_t mIsLargest; //!< Boolean stored as int32_t for field-based serialization.
    int32_t mAxis;      //!< Axis for top-k. -1 means last dimension.

    std::string mNamespace;

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class TopkLastDimPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    TopkLastDimPluginCreator();
    ~TopkLastDimPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    PluginFieldCollection const* getFieldNames() noexcept override;
    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;
    char const* getPluginNamespace() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace nvinfer1::plugin

#endif // TRT_TOPK_LAST_DIM_PLUGIN_H
