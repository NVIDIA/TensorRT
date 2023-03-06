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

#include <cuda.h>
#if CUDA_VERSION >= 10010

#ifndef TRT_GELU_PLUGIN_H
#define TRT_GELU_PLUGIN_H

#include "NvInferPlugin.h"
#include "common/bertCommon.h"
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
namespace bert
{

int computeGelu(cudaStream_t stream, int n, float const* input, float* output);

int computeGelu(cudaStream_t stream, int n, half const* input, half* output);

int computeGeluBias(
    float* output, float const* input, float const* bias, int const ld, int const cols, cudaStream_t stream);

int computeGeluBias(
    half* output, half const* input, half const* bias, int const ld, int const cols, cudaStream_t stream);

class GeluPluginDynamic : public nvinfer1::IPluginV2DynamicExt
{
public:
    GeluPluginDynamic(const std::string name, const nvinfer1::DataType type, nvinfer1::Weights const& bias);

    GeluPluginDynamic(const std::string name, void const* data, size_t length);

    // It doesn't make sense to make GeluPluginDynamic without arguments, so we delete
    // default constructor.
    GeluPluginDynamic() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept override;
    int enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 Methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    const std::string mLayerName;
    std::string mNamespace;

    nvinfer1::DataType mType;
    bool mHasBias;
    bert::cuda_shared_ptr<void> mBiasDev;
    size_t mLd;

    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2::enqueue;
    using IPluginV2Ext::configurePlugin;
};

class GeluPluginDynamicCreator : public nvinfer1::IPluginCreator
{
public:
    GeluPluginDynamicCreator();

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
} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_GELU_PLUGIN_H

#endif // CUDA_VERSION >= 10010
