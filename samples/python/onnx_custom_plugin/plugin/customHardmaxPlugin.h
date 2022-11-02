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

#ifndef TRT_HARDMAX_PLUGIN_H
#define TRT_HARDMAX_PLUGIN_H

#include "NvInferPlugin.h"
#include <cublas_v2.h>
#include <string>
#include <vector>

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class HardmaxPlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    HardmaxPlugin() = delete;
    HardmaxPlugin(int32_t axis);
    HardmaxPlugin(void const* serialData, size_t serialLength);
    ~HardmaxPlugin() override;

    template <typename TDataType>
    TDataType const* pointer_const_cast(void const* const p);

    template <typename TDataType>
    TDataType* pointer_cast(void* p);

    int32_t getNbOutputs() const noexcept override;

    // DynamicExt plugins returns DimsExprs class instead of Dims
    nvinfer1::DimsExprs getOutputDimensions(int32_t index, nvinfer1::DimsExprs const* inputs, int32_t nbInputDims,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override; // determine output dims based on input info

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    void attachToContext(
        cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept override;

    void detachFromContext() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

private:
    std::string mNamespace;

    // Number of elements in the axis along which hardmax is performed.
    int32_t mAxisSize{0};

    // Product of dimensions before and after mAxis.
    // For example, if the input dimensions are [3, 4, 5, 6, 7] and mAxis = 2,
    // then mDimProductOuter = 12 and mDimProductInner = 42.
    int32_t mDimProductOuter{1};
    int32_t mDimProductInner{1};

    cublasHandle_t mCublas;

    // Attributes
    // Axis along which to perform hardmax.
    // Can be negative initially, but once configurePlugin() is called it will
    // be converted to a positive axis.
    int32_t mAxis{-1};
};

class HardmaxPluginCreator : public nvinfer1::IPluginCreator
{
public:
    HardmaxPluginCreator();

    ~HardmaxPluginCreator() override = default;

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
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

#endif // TRT_HARDMAX_PLUGIN_H
