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
constexpr const char* SPLIT_PLUGIN_VERSION{"1"};
constexpr const char* SPLIT_PLUGIN_NAME{"Split"};
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

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
  int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override
    {
        return new SplitPlugin{_axis, _output_lengths};
    }
    void destroy() noexcept override
    {
        delete this;
    }
    const char* getPluginVersion() const noexcept override
    {
        return SPLIT_PLUGIN_VERSION;
    }
    const char* getPluginType() const noexcept override
    {
        return SPLIT_PLUGIN_NAME;
    }
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
        const nvinfer1::PluginTensorDesc* /*outputs*/, int /*nbOutputs*/) const noexcept override
    {
        return 0;
    }
    void setPluginNamespace(const char* /*pluginNamespace*/) noexcept override {}
    const char* getPluginNamespace() const noexcept override
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

    ~SplitPluginCreator() {}

    const char* getPluginName() const noexcept
    {
        return SPLIT_PLUGIN_NAME;
    }

    const char* getPluginVersion() const noexcept
    {
        return SPLIT_PLUGIN_VERSION;
    }

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept
    {
        std::cerr << "Function not implemented" << std::endl;
        return nullptr;
    }

    nvinfer1::IPluginV2DynamicExt* createPlugin(const char* /*name*/, const nvinfer1::PluginFieldCollection* /*fc*/) noexcept
    {
        std::cerr << "Function not implemented" << std::endl;
        return nullptr;
    }

    nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* /*name*/, const void* serialData, size_t serialLength) noexcept
    {
        return new SplitPlugin{serialData, serialLength};
    }

    void setPluginNamespace(const char* libNamespace) noexcept
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const noexcept
    {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1
#endif // TRT_SPLIT_PLUGIN_H
