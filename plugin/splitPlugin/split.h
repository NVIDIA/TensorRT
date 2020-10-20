/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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

#include "checkMacrosPlugin.h"
#include "serialize.hpp"

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

protected:
    // Supress warnings about hiding function names due to overloads and overrides of virtuals.
    using IPluginV2DynamicExt::enqueue;
    using IPluginV2DynamicExt::getOutputDimensions;
    using IPluginV2DynamicExt::getWorkspaceSize;
    using IPluginV2DynamicExt::configurePlugin;
    void deserialize(void const* serialData, size_t serialLength)
    {
        deserialize_value(&serialData, &serialLength, &_axis);
        deserialize_value(&serialData, &serialLength, &_output_lengths);
    }
    size_t getSerializationSize() const override
    {
        return serialized_size(_axis) + serialized_size(_output_lengths);
    }
    void serialize(void* buffer) const override
    {
        serialize_value(&buffer, _axis);
        serialize_value(&buffer, _output_lengths);
    }

public:
    SplitPlugin(int axis, int* const& output_lengths, int noutput)
        : _axis(axis)
        , _output_lengths(std::vector<int>(output_lengths, output_lengths + noutput))
    {
        assert(axis <= nvinfer1::Dims::MAX_DIMS);
    }
    SplitPlugin(int axis, std::vector<int> output_lengths)
        : _axis(axis)
        , _output_lengths(output_lengths)
    {
        assert(axis <= nvinfer1::Dims::MAX_DIMS);
    }
    SplitPlugin(void const* serialData, size_t serialLength)
    {
        this->deserialize(serialData, serialLength);
    }

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;
    int initialize() override;
    void terminate() override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT override;
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;

    nvinfer1::IPluginV2DynamicExt* clone() const override
    {
        return new SplitPlugin{_axis, _output_lengths};
    }
    void destroy() override
    {
        delete this;
    }
    const char* getPluginVersion() const override
    {
        return SPLIT_PLUGIN_VERSION;
    }
    const char* getPluginType() const override
    {
        return SPLIT_PLUGIN_NAME;
    }
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
        const nvinfer1::PluginTensorDesc* /*outputs*/, int /*nbOutputs*/) const TRTNOEXCEPT override
    {
        return 0;
    }
    void setPluginNamespace(const char* /*pluginNamespace*/) override {}
    const char* getPluginNamespace() const override
    {
        return "";
    }
    int getNbOutputs() const override
    {
        return _output_lengths.size();
    }
    void attachToContext(
        cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, nvinfer1::IGpuAllocator* /*allocator*/) override
    {
    }
    void detachFromContext() override {}
};

class SplitPluginCreator : public nvinfer1::IPluginCreator
{
public:
    SplitPluginCreator() {}

    ~SplitPluginCreator() {}

    const char* getPluginName() const
    {
        return SPLIT_PLUGIN_NAME;
    }

    const char* getPluginVersion() const
    {
        return SPLIT_PLUGIN_VERSION;
    }

    const nvinfer1::PluginFieldCollection* getFieldNames()
    {
        std::cerr << "Function not implemented" << std::endl;
        return nullptr;
    }

    nvinfer1::IPluginV2DynamicExt* createPlugin(const char* /*name*/, const nvinfer1::PluginFieldCollection* /*fc*/)
    {
        std::cerr << "Function not implemented" << std::endl;
        return nullptr;
    }

    nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* /*name*/, const void* serialData, size_t serialLength)
    {
        return new SplitPlugin{serialData, serialLength};
    }

    void setPluginNamespace(const char* libNamespace)
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const
    {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1
#endif // TRT_SPLIT_PLUGIN_H
