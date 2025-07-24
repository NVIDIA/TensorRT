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
 **************************************************************************
 * Modified from mmcv (https://github.com/open-mmlab/mmcv/tree/master/mmcv)
 * Copyright (c) OpenMMLab. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 [see LICENSE for details]
 * https://github.com/open-mmlab/mmcv/blob/master/LICENSE
 **************************************************************************
 */

#ifndef TRT_MODULATED_DEFORM_CONV_PLUGIN_H
#define TRT_MODULATED_DEFORM_CONV_PLUGIN_H

#include <cstdint>
#include <cuda.h>
#include <memory>
#include <string>
#include <vector>

#include "common/bertCommon.h"
#include "common/checkMacrosPlugin.h"
#include "common/cublasWrapper.h"
#include "common/plugin.h"
#include "common/serialize.hpp"

#include "modulatedDeformConvCudaHelper.h"

namespace nvinfer1
{
namespace plugin
{

class ModulatedDeformableConvPluginDynamic final : public nvinfer1::IPluginV3,
                                                   public nvinfer1::IPluginV3OneCore,
                                                   public nvinfer1::IPluginV3OneBuild,
                                                   public nvinfer1::IPluginV3OneRuntime
{
public:
    ModulatedDeformableConvPluginDynamic(std::string const& name, nvinfer1::Dims const stride,
        nvinfer1::Dims const padding, nvinfer1::Dims const dilation, int32_t const deformableGroup,
        int32_t const group);

    ModulatedDeformableConvPluginDynamic() = delete;

    ~ModulatedDeformableConvPluginDynamic() override;

    // --- IPluginV3 methods ---
    nvinfer1::IPluginV3* clone() noexcept override;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    // --- IPluginV3OneCore methods ---
    int32_t getNbOutputs() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;

    // --- IPluginV3OneBuild methods ---
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    // --- IPluginV3OneRuntime methods ---
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDescs, nvinfer1::PluginTensorDesc const* outputDescs,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override;

private:
    // Helper method to manage cuBLAS resources
    void setCublasResources(std::shared_ptr<nvinfer1::pluginInternal::CublasWrapper> cublasWrapper);

private:
    std::string const mLayerName;
    std::string mNamespace;

    nvinfer1::Dims mStride;
    nvinfer1::Dims mPadding;
    nvinfer1::Dims mDilation;
    int32_t mDeformableGroup;
    int32_t mGroup;
    int32_t mWithBias;

    nvinfer1::pluginInternal::cublasHandle_t mCublasHandle{nullptr};
    // the wrapper pointer is shared among all plugins attached to the same context.
    std::shared_ptr<nvinfer1::pluginInternal::CublasWrapper> mCublasWrapper;

    nvinfer1::PluginFieldCollection mFCToSerialize;
    std::vector<nvinfer1::PluginField> mDataToSerialize;
};

class ModulatedDeformableConvPluginDynamicCreator final : public nvinfer1::IPluginCreatorV3One
{
public:
    ModulatedDeformableConvPluginDynamicCreator();
    ~ModulatedDeformableConvPluginDynamicCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;
    char const* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_MODULATED_DEFORM_CONV_PLUGIN_H
