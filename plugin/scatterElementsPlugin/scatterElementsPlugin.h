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

#ifndef TRT_SCATTER_ELEMENTS_PLUGIN_H
#define TRT_SCATTER_ELEMENTS_PLUGIN_H
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "common/plugin.h"
#include "scatterElementsCommon.h"

namespace nvinfer1
{
namespace plugin
{

class ScatterElementsPluginV3 : public IPluginV3,
                                public IPluginV3OneCore,
                                public IPluginV3OneBuild,
                                public IPluginV3OneRuntime
{
public:
    // ctor and dtor
    ScatterElementsPluginV3() = delete;

    ScatterElementsPluginV3(ScatterElementsPluginV3 const&) = delete;

    ScatterElementsPluginV3(std::string const&, int32_t);

    ScatterElementsPluginV3(ReductionType, int32_t);

    ~ScatterElementsPluginV3() override = default;

    // IPluginV3 Methods
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;

    ScatterElementsPluginV3* clone() noexcept override;
    // end IPluginV3 Methods

    // IPluginV3Core Methods
    char const* getPluginVersion() const noexcept override;

    char const* getPluginName() const noexcept override;

    char const* getPluginNamespace() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

    // end IPluginV3Core Methods

    // IPluginV3Build Methods
    int32_t getNbOutputs() const noexcept override;

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    // end IPluginV3Build Methods

    // IPluginV3Runtime Methods
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;

    PluginFieldCollection const* getFieldsToSerialize() noexcept override;
    // end IPluginV3Runtime Methods

private:
    ReductionType mReduction;
    int32_t mAxis;
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
    std::string mNamespace;
    // input metadata
    static constexpr int32_t kINDICES_TENSOR_IDX = 1;
    static constexpr int32_t kUPDATES_TENSOR_IDX = 2;
    static constexpr int32_t kDATA_TENSOR_IDX = 0;
    // output metadata
    static constexpr int32_t kOUTPUT_TENSOR_IDX = 0;
};

class ScatterElementsPluginV3Creator : public nvinfer1::IPluginCreatorV3One
{
public:
    // ctor and dtor
    ScatterElementsPluginV3Creator();

    ~ScatterElementsPluginV3Creator() override = default;

    // get plugin metadata
    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    char const* getPluginNamespace() const noexcept override;

    // setter
    void setPluginNamespace(char const* libNamespace) noexcept;

    // create plugin
    IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;

private:
    nvinfer1::PluginFieldCollection gFC;
    std::vector<PluginField> gPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_SCATTER_ELEMENTS_PLUGIN_H
