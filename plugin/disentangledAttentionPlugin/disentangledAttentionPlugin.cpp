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

#include <numeric>
#include <stdexcept>
#include "disentangledAttentionPlugin.h"
#include "NvInferPlugin.h"
#include <cuda_fp16.h>


using namespace nvinfer1;
using nvinfer1::plugin::DisentangledAttentionPlugin;
using nvinfer1::plugin::DisentangledAttentionPluginCreator;


// Static class fields initialization
PluginFieldCollection DisentangledAttentionPluginCreator::mFC{};
std::vector<PluginField> DisentangledAttentionPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(DisentangledAttentionPluginCreator);

#define CHECK_CUDNN(call)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        cudnnStatus_t status = call;                                                                                   \
        if (status != CUDNN_STATUS_SUCCESS)                                                                            \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

namespace
{
constexpr char const* DEBERTA_NAME{"DisentangledAttention_TRT"};
constexpr char const* DEBERTA_VERSION{"1"};
} // namespace

DisentangledAttentionPlugin::DisentangledAttentionPlugin() {}

DisentangledAttentionPlugin::DisentangledAttentionPlugin(int32_t span, float factor)
    : mSpan(span)
    , mFactor(factor)
{
}

DisentangledAttentionPlugin::DisentangledAttentionPlugin(void const* serialData, size_t serialLength)
{
    // Deserialize in the same order as serialization
    deserialize_value(&serialData, &serialLength, &mSpan);
    deserialize_value(&serialData, &serialLength, &mFactor);
}

DisentangledAttentionPlugin::~DisentangledAttentionPlugin()
{
    terminate();
}

int32_t DisentangledAttentionPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t DisentangledAttentionPlugin::initialize() noexcept
{
    // if need large amount of GPU memory, recommend to specify in getWorkspaceSize so TRT allocates it. If not, when a
    // plugin is called many times, the memory manually allocated by this initialize() is repeated many times -- may
    // overflow
    return 0;
}

char const* DisentangledAttentionPlugin::getPluginType() const noexcept
{
    return DEBERTA_NAME;
}

char const* DisentangledAttentionPlugin::getPluginVersion() const noexcept
{
    return DEBERTA_VERSION;
}

// IPluginV2DynamicExt Methods
nvinfer1::DimsExprs DisentangledAttentionPlugin::getOutputDimensions(
    int32_t index, nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    nvinfer1::DimsExprs output;

    PLUGIN_ASSERT(nbInputs == 3); // 3 inputs
    output = inputs[0];           // same as input[0], i.e. data0

    PLUGIN_ASSERT(index < 1); // only one output

    return output;
}

void DisentangledAttentionPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void DisentangledAttentionPlugin::detachFromContext() noexcept {}

template <typename TDataType>
TDataType const* DisentangledAttentionPlugin::pointer_const_cast(void const* const p)
{
    return static_cast<TDataType const*>(p);
}

template <typename TDataType>
TDataType* DisentangledAttentionPlugin::pointer_cast(void* const p)
{
    return static_cast<TDataType*>(p);
}

int32_t DisentangledAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{

#if kDISENTANGLED_VERSION == 1
    nvinfer1::Dims dims0 = inputDesc[0].dims;
    nvinfer1::Dims dims1 = inputDesc[1].dims;
    nvinfer1::Dims dims2 = inputDesc[2].dims;
    dim3 dimData0(dims0.d[0], dims0.d[1], dims0.d[2]);
    dim3 dimData1(dims1.d[0], dims1.d[1], dims1.d[2]);
    dim3 dimData2(dims2.d[0], dims2.d[1], dims2.d[2]);
    dim3 dimResult(dimData0);

    dim3 block_optimized(kDISENTANGLED_TILESIZE_V1, kDISENTANGLED_BLOCKDIMY_V1);
    dim3 grid_optimized((dimResult.z - 1) / kDISENTANGLED_TILESIZE_V1 + 1,
        (dimResult.y - 1) / kDISENTANGLED_TILESIZE_V1 + 1, dimResult.x);

    if (inputDesc[0].type == nvinfer1::DataType::kFLOAT)
    {
        auto const* data0 = pointer_const_cast<float>(inputs[0]);
        auto const* data1 = pointer_const_cast<float>(inputs[1]);
        auto const* data2 = pointer_const_cast<float>(inputs[2]);
        auto* result = pointer_cast<float>(outputs[0]);
        disentangled_kernel_wrapper<float, kDISENTANGLED_TILESIZE_V1, kDISENTANGLED_BLOCKDIMY_V1>(data0, data1, data2,
            result, dimData0, dimData1, dimData2, dimResult, mFactor, mSpan, block_optimized, grid_optimized, stream);
    }
    else if (inputDesc[0].type == nvinfer1::DataType::kHALF)
    {
        auto const* data0 = pointer_const_cast<__half>(inputs[0]);
        auto const* data1 = pointer_const_cast<__half>(inputs[1]);
        auto const* data2 = pointer_const_cast<__half>(inputs[2]);
        auto* result = pointer_cast<__half>(outputs[0]);
        __half factor = __float2half(mFactor);
        disentangled_kernel_wrapper<__half, kDISENTANGLED_TILESIZE_V1, kDISENTANGLED_BLOCKDIMY_V1>(data0, data1, data2,
            result, dimData0, dimData1, dimData2, dimResult, factor, mSpan, block_optimized, grid_optimized, stream);
    }
    else if (inputDesc[0].type == nvinfer1::DataType::kINT8)
    {
        auto const* data0 = pointer_const_cast<int8_t>(inputs[0]);
        auto const* data1 = pointer_const_cast<int8_t>(inputs[1]);
        auto const* data2 = pointer_const_cast<int8_t>(inputs[2]);
        auto* result = pointer_cast<int8_t>(outputs[0]);
        int8_t factor = int8_t(mFactor);
        disentangled_kernel_wrapper<int8_t, kDISENTANGLED_TILESIZE_V1, kDISENTANGLED_BLOCKDIMY_V1>(data0, data1, data2,
            result, dimData0, dimData1, dimData2, dimResult, factor, mSpan, block_optimized, grid_optimized, stream);
    }
#elif kDISENTANGLED_VERSION == 2
    nvinfer1::Dims dims0 = inputDesc[0].dims;
    nvinfer1::Dims dims1 = inputDesc[1].dims;
    nvinfer1::Dims dims2 = inputDesc[2].dims;
    dim3 dimData0(dims0.d[0], dims0.d[1], dims0.d[2]);
    dim3 dimData1(dims1.d[0], dims1.d[1], dims1.d[2]);
    dim3 dimData2(dims2.d[0], dims2.d[1], dims2.d[2]);
    dim3 dimResult(dimData0);

    dim3 block_optimized(kDISENTANGLED_TILESIZE_V2, kDISENTANGLED_BLOCKDIMY_V2);
    dim3 grid_optimized((dimResult.z - 1) / kDISENTANGLED_TILESIZE_V2 + 1,
        (dimResult.y - 1) / kDISENTANGLED_TILESIZE_V2 + 1, dimResult.x);

    if (inputDesc[0].type == nvinfer1::DataType::kFLOAT)
    {
        auto const* data0 = pointer_const_cast<float>(inputs[0]);
        auto const* data1 = pointer_const_cast<float>(inputs[1]);
        auto const* data2 = pointer_const_cast<float>(inputs[2]);
        auto* result = pointer_cast<float>(outputs[0]);
        disentangled_kernel_wrapper<float, kDISENTANGLED_TILESIZE_V2, kDISENTANGLED_BLOCKDIMY_V2>(data0, data1, data2,
            result, dimData0, dimData1, dimData2, dimResult, mFactor, mSpan, block_optimized, grid_optimized, stream);
    }
    else if (inputDesc[0].type == nvinfer1::DataType::kHALF)
    {
        auto const* data0 = pointer_const_cast<__half>(inputs[0]);
        auto const* data1 = pointer_const_cast<__half>(inputs[1]);
        auto const* data2 = pointer_const_cast<__half>(inputs[2]);
        auto* result = pointer_cast<__half>(outputs[0]);
        __half factor = __float2half(mFactor);
        disentangled_kernel_wrapper<__half, kDISENTANGLED_TILESIZE_V2, kDISENTANGLED_BLOCKDIMY_V2>(data0, data1, data2,
            result, dimData0, dimData1, dimData2, dimResult, factor, mSpan, block_optimized, grid_optimized, stream);
    }
    else if (inputDesc[0].type == nvinfer1::DataType::kINT8)
    {
        auto const* data0 = pointer_const_cast<int8_t>(inputs[0]);
        auto const* data1 = pointer_const_cast<int8_t>(inputs[1]);
        auto const* data2 = pointer_const_cast<int8_t>(inputs[2]);
        auto* result = pointer_cast<int8_t>(outputs[0]);
        int8_t factor = int8_t(mFactor);
        disentangled_kernel_wrapper<int8_t, kDISENTANGLED_TILESIZE_V2, kDISENTANGLED_BLOCKDIMY_V2>(data0, data1, data2,
            result, dimData0, dimData1, dimData2, dimResult, factor, mSpan, block_optimized, grid_optimized, stream);
    }
#endif

    return cudaPeekAtLastError();
}

size_t DisentangledAttentionPlugin::getSerializationSize() const noexcept
{
    return sizeof(mSpan) + sizeof(mFactor);
}

void DisentangledAttentionPlugin::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mSpan);
    serialize_value(&buffer, mFactor);
}

bool DisentangledAttentionPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{

    PLUGIN_ASSERT(inOut && pos < (nbInputs + nbOutputs));

    bool const consistentFloatPrecision
        = (inOut[pos].type == inOut[0].type); // all inputs & outputs should have the same precision type

    // 3 inputs, 1 output
    switch (pos)
    {
    case 0: 
        return (inOut[pos].type == nvinfer1::DataType::kINT8 || inOut[pos].type == nvinfer1::DataType::kHALF || inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR && consistentFloatPrecision; // linear means row-major ordering
    case 1:
        return (inOut[pos].type == nvinfer1::DataType::kINT8 || inOut[pos].type == nvinfer1::DataType::kHALF || inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR && consistentFloatPrecision;
    case 2:
        return (inOut[pos].type == nvinfer1::DataType::kINT8 || inOut[pos].type == nvinfer1::DataType::kHALF || inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR && consistentFloatPrecision;
    case 3:
        return (inOut[pos].type == nvinfer1::DataType::kINT8 || inOut[pos].type == nvinfer1::DataType::kHALF || inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR && consistentFloatPrecision;
    }
    // types: kFLOAT, kHALF, kINT32, kINT8
    return false;
}

void DisentangledAttentionPlugin::terminate() noexcept
{
}

void DisentangledAttentionPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* DisentangledAttentionPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new DisentangledAttentionPlugin(mSpan, mFactor);
        plugin->setPluginNamespace(mPluginNamespace);
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void DisentangledAttentionPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{

    // inputs
    PLUGIN_ASSERT(nbInputs == 3); // 3 inputs

    // check for valid input dimensions
    PLUGIN_ASSERT(in[0].desc.dims.nbDims == 3);
    PLUGIN_ASSERT(in[1].desc.dims.nbDims == 3);
    PLUGIN_ASSERT(in[2].desc.dims.nbDims == 3);

    // check BN (batch_size * num_heads) dimension consistency
    PLUGIN_ASSERT(in[0].desc.dims.d[0] == in[1].desc.dims.d[0]);
    PLUGIN_ASSERT(in[0].desc.dims.d[0] == in[2].desc.dims.d[0]);

    // check S (sequence_length) dimension consistency
    PLUGIN_ASSERT(in[0].desc.dims.d[1] == in[1].desc.dims.d[1]);
    PLUGIN_ASSERT(in[0].desc.dims.d[1] == in[2].desc.dims.d[1]);
    PLUGIN_ASSERT(in[0].desc.dims.d[1] == in[0].desc.dims.d[2]);

    // check K (2 * span) dimension consistency for in[1] and in[2]
    PLUGIN_ASSERT(in[1].desc.dims.d[2] == 2 * mSpan);
    PLUGIN_ASSERT(in[2].desc.dims.d[2] == 2 * mSpan);

    // Outputs (same dimension as in[0])
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(out[0].desc.dims.nbDims == 3);
    PLUGIN_ASSERT(in[0].desc.dims.d[0] == out[0].desc.dims.d[0]);
    PLUGIN_ASSERT(in[0].desc.dims.d[1] == out[0].desc.dims.d[1]);
    PLUGIN_ASSERT(in[0].desc.dims.d[2] == out[0].desc.dims.d[2]);
}

nvinfer1::DataType DisentangledAttentionPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    PLUGIN_ASSERT(inputTypes && nbInputs > 0 && index < 1);
    return inputTypes[0]; // version 1, same as data1; version 2, same as data0
}

size_t DisentangledAttentionPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

void DisentangledAttentionPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    mPluginNamespace = libNamespace;
}

char const* DisentangledAttentionPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

DisentangledAttentionPluginCreator::DisentangledAttentionPluginCreator()
{
    mPluginAttributes.clear();

    // consistent with the ONNX model attr fields
    mPluginAttributes.emplace_back(PluginField("span", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("factor", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* DisentangledAttentionPluginCreator::getPluginName() const noexcept
{
    return DEBERTA_NAME;
}

char const* DisentangledAttentionPluginCreator::getPluginVersion() const noexcept
{
    return DEBERTA_VERSION;
}

PluginFieldCollection const* DisentangledAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

char const* DisentangledAttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void DisentangledAttentionPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

IPluginV2DynamicExt* DisentangledAttentionPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        // Set default invalid values (for assert in case when attributes are missing)
        int32_t span = 0;
        float factor = 0.0F;
        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare("span") == 0)
            {
                span = *static_cast<int32_t const*>(fc->fields[i].data);
            }
            if (field_name.compare("factor") == 0)
            {
                factor = *static_cast<float const*>(fc->fields[i].data);
            }
        }

        PLUGIN_ASSERT(span >= 0);
        PLUGIN_ASSERT(factor > 0.0F && factor < 1.0F); // factor is 1/sqrt(3d), therefore must less than 1

        DisentangledAttentionPlugin* plugin = new DisentangledAttentionPlugin(span, factor);
        plugin->setPluginNamespace(mNamespace.c_str());

        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* DisentangledAttentionPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        DisentangledAttentionPlugin* plugin = new DisentangledAttentionPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());

        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
