/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_MULTISCALE_DEFORMABLE_ATTN_PLUGIN_H
#define TRT_MULTISCALE_DEFORMABLE_ATTN_PLUGIN_H

// For loadLibrary
#ifdef _MSC_VER
// Needed so that the max/min definitions in windows.h do not conflict with
// std::max/min.
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#else
#include <dlfcn.h>
#endif

#include <memory>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferVersion.h"

#include "plugin.h"

#if NV_TENSORRT_MAJOR > 7
#define PLUGIN_NOEXCEPT noexcept
#else
#define PLUGIN_NOEXCEPT
#endif

using namespace nvinfer1::plugin;

namespace nvinfer1
{
namespace plugin
{
class MultiscaleDeformableAttnPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    MultiscaleDeformableAttnPlugin();

    MultiscaleDeformableAttnPlugin(void const* data, size_t length);

    // IPluginV2DynamicExt methods
    nvinfer1::IPluginV2DynamicExt* clone() const PLUGIN_NOEXCEPT override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) PLUGIN_NOEXCEPT override;
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) PLUGIN_NOEXCEPT override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) PLUGIN_NOEXCEPT override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const PLUGIN_NOEXCEPT override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) PLUGIN_NOEXCEPT override;
    void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext,
        nvinfer1::IGpuAllocator* gpuAllocator) PLUGIN_NOEXCEPT override;
    void detachFromContext() PLUGIN_NOEXCEPT override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const PLUGIN_NOEXCEPT override;

    // IPluginV2 Methods
    char const* getPluginType() const PLUGIN_NOEXCEPT override;
    char const* getPluginVersion() const PLUGIN_NOEXCEPT override;
    int32_t getNbOutputs() const PLUGIN_NOEXCEPT override;
    int32_t initialize() PLUGIN_NOEXCEPT override;
    void terminate() PLUGIN_NOEXCEPT override;
    size_t getSerializationSize() const PLUGIN_NOEXCEPT override;
    void serialize(void* buffer) const PLUGIN_NOEXCEPT override;
    void destroy() PLUGIN_NOEXCEPT override;
    void setPluginNamespace(char const* pluginNamespace) PLUGIN_NOEXCEPT override;
    char const* getPluginNamespace() const PLUGIN_NOEXCEPT override;

private:
    std::string mNamespace;

#if NV_TENSORRT_MAJOR < 8
    using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::configurePlugin;
    using nvinfer1::IPluginV2DynamicExt::enqueue;
    using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
    using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
    using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::supportsFormat;
#endif
};

class MultiscaleDeformableAttnPluginCreator : public nvinfer1::IPluginCreator
{
public:
    MultiscaleDeformableAttnPluginCreator();
    char const* getPluginName() const PLUGIN_NOEXCEPT override;
    char const* getPluginVersion() const PLUGIN_NOEXCEPT override;
    nvinfer1::PluginFieldCollection const* getFieldNames() PLUGIN_NOEXCEPT override;
    nvinfer1::IPluginV2* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc) PLUGIN_NOEXCEPT override;
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) PLUGIN_NOEXCEPT override;
    void setPluginNamespace(char const* pluginNamespace) PLUGIN_NOEXCEPT override;
    char const* getPluginNamespace() const PLUGIN_NOEXCEPT override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif
