/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>
#include <vector>

#include "common/bertCommon.h"
#include "common/checkMacrosPlugin.h"
#include "common/plugin.h"
#include "common/serialize.hpp"
#include "modulatedDeformConvCudaHelper.h"

namespace nvinfer1
{
namespace plugin
{

class ModulatedDeformableConvPluginDynamic : public nvinfer1::IPluginV2DynamicExt
{
public:
    ModulatedDeformableConvPluginDynamic(std::string const& name, const nvinfer1::Dims stride,
        const nvinfer1::Dims padding, const nvinfer1::Dims dilation, int32_t const deformableGroup,
        int32_t const group);

    ModulatedDeformableConvPluginDynamic(const std::string name, void const* data, size_t length);

    ModulatedDeformableConvPluginDynamic() = delete;

    ~ModulatedDeformableConvPluginDynamic() override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext,
        nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;
    void detachFromContext() noexcept override;

    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

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

private:
    const std::string mLayerName;
    std::string mNamespace;

    nvinfer1::Dims mStride;
    nvinfer1::Dims mPadding;
    nvinfer1::Dims mDilation;
    int32_t mDeformableGroup;
    int32_t mGroup;
    bool mWithBias;

    nvinfer1::pluginInternal::cublasHandle_t mCublasHandle{nullptr};
    // the wrapper pointer is shared among all plugins attached to the same context.
    std::shared_ptr<nvinfer1::pluginInternal::CublasWrapper> mCublasWrapper;
};

class ModulatedDeformableConvPluginDynamicCreator : public nvinfer1::IPluginCreator
{
public:
    ModulatedDeformableConvPluginDynamicCreator();

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_MODULATED_DEFORM_CONV_PLUGIN_H
