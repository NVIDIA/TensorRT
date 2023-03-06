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

#ifndef TRT_SPLIT_PLUGIN_H
#define TRT_SPLIT_PLUGIN_H
#include <NvInfer.h>

#include "common/checkMacrosPlugin.h"
#include "common/serialize.hpp"

#include <iostream>
#include <string>
#include <thrust/device_vector.h>

namespace
{
constexpr char const* SPLIT_PLUGIN_VERSION{"1"};
constexpr char const* SPLIT_PLUGIN_NAME{"Split"};
} // namespace

namespace nvinfer1
{
namespace plugin
{
class SplitPlugin final : public nvinfer1::IPluginV2DynamicExt
{
    int _axis;
    std::vector<int> _output_lengths;
    int _nx, _ny, _nz;
    int _x_stride, _y_stride, _z_stride;
    thrust::device_vector<int> _d_segment_offsets;
    thrust::device_vector<float*> _d_output_ptrs;

    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2::enqueue;
    using IPluginV2Ext::configurePlugin;

protected:
    void deserialize(void const* serialData, size_t serialLength) noexcept
    {
        deserialize_value(&serialData, &serialLength, &_axis);
        deserialize_value(&serialData, &serialLength, &_output_lengths);
    }
    size_t getSerializationSize() const noexcept override
    {
        return serialized_size(_axis) + serialized_size(_output_lengths);
    }
    void serialize(void* buffer) const noexcept override
    {
        serialize_value(&buffer, _axis);
        serialize_value(&buffer, _output_lengths);
    }

public:
    SplitPlugin(int axis, int* const& output_lengths, int noutput)
        : _axis(axis)
        , _output_lengths(std::vector<int>(output_lengths, output_lengths + noutput))
    {
        PLUGIN_ASSERT(axis <= nvinfer1::Dims::MAX_DIMS);
    }
    SplitPlugin(int axis, std::vector<int> output_lengths)
        : _axis(axis)
        , _output_lengths(output_lengths)
    {
        PLUGIN_ASSERT(axis <= nvinfer1::Dims::MAX_DIMS);
    }
    SplitPlugin(void const* serialData, size_t serialLength)
    {
        this->deserialize(serialData, serialLength);
    }

    bool supportsFormatCombination(
        int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept override;
    nvinfer1::DataType getOutputDataType(
        int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept override;
    int enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override
    {
        return new SplitPlugin{_axis, _output_lengths};
    }
    void destroy() noexcept override
    {
        delete this;
    }
    char const* getPluginVersion() const noexcept override
    {
        return SPLIT_PLUGIN_VERSION;
    }
    char const* getPluginType() const noexcept override
    {
        return SPLIT_PLUGIN_NAME;
    }
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* /*inputs*/, int /*nbInputs*/,
        nvinfer1::PluginTensorDesc const* /*outputs*/, int /*nbOutputs*/) const noexcept override
    {
        return 0;
    }
    void setPluginNamespace(char const* /*pluginNamespace*/) noexcept override {}
    char const* getPluginNamespace() const noexcept override
    {
        return "";
    }
    int getNbOutputs() const noexcept override
    {
        return _output_lengths.size();
    }
    void attachToContext(
        cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, nvinfer1::IGpuAllocator* /*allocator*/) noexcept override
    {
    }
    void detachFromContext() noexcept override {}
};

class SplitPluginCreator : public nvinfer1::IPluginCreator
{
public:
    SplitPluginCreator() {}

    ~SplitPluginCreator() override {}

    char const* getPluginName() const noexcept override
    {
        return SPLIT_PLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override
    {
        return SPLIT_PLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override
    {
        std::cerr << "Function not implemented" << std::endl;
        return nullptr;
    }

    nvinfer1::IPluginV2DynamicExt* createPlugin(
        char const* /*name*/, nvinfer1::PluginFieldCollection const* /*fc*/) noexcept override
    {
        std::cerr << "Function not implemented" << std::endl;
        return nullptr;
    }

    nvinfer1::IPluginV2DynamicExt* deserializePlugin(
        char const* /*name*/, void const* serialData, size_t serialLength) noexcept override
    {
        return new SplitPlugin{serialData, serialLength};
    }

    void setPluginNamespace(char const* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1
#endif // TRT_SPLIT_PLUGIN_H
