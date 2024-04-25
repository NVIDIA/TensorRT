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

#include "disentangledAttentionPlugin.h"
#include "NvInferPlugin.h"
#include <cuda_fp16.h>
#include <numeric>
#include <stdexcept>

using namespace nvinfer1;
using nvinfer1::plugin::DisentangledAttentionPlugin;
using nvinfer1::plugin::DisentangledAttentionPluginCreator;

// Static class fields initialization
PluginFieldCollection DisentangledAttentionPluginCreator::mFC{};
std::vector<PluginField> DisentangledAttentionPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(DisentangledAttentionPluginCreator);

namespace
{
constexpr char const* kDEBERTA_PLUGIN_NAME{"DisentangledAttention_TRT"};
constexpr char const* kDEBERTA_PLUGIN_VERSION{"1"};
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

int32_t DisentangledAttentionPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t DisentangledAttentionPlugin::initialize() noexcept
{
    return 0;
}

char const* DisentangledAttentionPlugin::getPluginType() const noexcept
{
    return kDEBERTA_PLUGIN_NAME;
}

char const* DisentangledAttentionPlugin::getPluginVersion() const noexcept
{
    return kDEBERTA_PLUGIN_VERSION;
}

// IPluginV2DynamicExt Methods
nvinfer1::DimsExprs DisentangledAttentionPlugin::getOutputDimensions(
    int32_t index, nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(index == 0); // Only one output
        return inputs[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nvinfer1::DimsExprs{};
}

template <typename TDataType>
void DisentangledAttentionPlugin::enqueueType(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, cudaStream_t stream,
    TDataType factor)
{
    nvinfer1::Dims dims0 = inputDesc[0].dims;
    nvinfer1::Dims dims1 = inputDesc[1].dims;
    nvinfer1::Dims dims2 = inputDesc[2].dims;
    dim3 dimData0(dims0.d[0], dims0.d[1], dims0.d[2]);
    dim3 dimData1(dims1.d[0], dims1.d[1], dims1.d[2]);
    dim3 dimData2(dims2.d[0], dims2.d[1], dims2.d[2]);
    dim3 dimResult(dimData0);

    dim3 blockOptimized(kDISENTANGLED_TILESIZE, kDISENTANGLED_BLOCKDIMY);
    dim3 gridOptimized(
        (dimResult.z - 1) / kDISENTANGLED_TILESIZE + 1, (dimResult.y - 1) / kDISENTANGLED_TILESIZE + 1, dimResult.x);

    auto const* data0 = static_cast<TDataType const*>(inputs[0]);
    auto const* data1 = static_cast<TDataType const*>(inputs[1]);
    auto const* data2 = static_cast<TDataType const*>(inputs[2]);
    auto* result = static_cast<TDataType*>(outputs[0]);
    disentangled_kernel_wrapper<TDataType, kDISENTANGLED_TILESIZE, kDISENTANGLED_BLOCKDIMY>(data0, data1, data2, result,
        dimData0, dimData1, dimData2, dimResult, factor, mSpan, blockOptimized, gridOptimized, stream);
}

int32_t DisentangledAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs,
    void* /* workspace */, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr);

        switch (inputDesc[0].type)
        {
        case nvinfer1::DataType::kFLOAT:
            enqueueType<float>(inputDesc, outputDesc, inputs, outputs, stream, mFactor);
            break;
        case nvinfer1::DataType::kHALF:
            enqueueType<__half>(inputDesc, outputDesc, inputs, outputs, stream, __float2half(mFactor));
            break;
        case nvinfer1::DataType::kINT8:
            enqueueType<int8_t>(inputDesc, outputDesc, inputs, outputs, stream, static_cast<int8_t>(mFactor));
            break;
        default: PLUGIN_VALIDATE(false, "Unsupported Datatype"); break;
        }
        return cudaPeekAtLastError();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return STATUS_FAILURE;
    }
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

    return (inOut[pos].type == nvinfer1::DataType::kINT8 || inOut[pos].type == nvinfer1::DataType::kHALF
               || inOut[pos].type == nvinfer1::DataType::kFLOAT)
        && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR && consistentFloatPrecision;
}

void DisentangledAttentionPlugin::terminate() noexcept {}

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
        plugin->setPluginNamespace(mNamespace.c_str());
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
    try
    {
        // inputs
        PLUGIN_VALIDATE(nbInputs == 3); // 3 inputs

        // check for valid input dimensions
        PLUGIN_VALIDATE(in[0].desc.dims.nbDims == 3);
        PLUGIN_VALIDATE(in[1].desc.dims.nbDims == 3);
        PLUGIN_VALIDATE(in[2].desc.dims.nbDims == 3);

        // check BN (batch_size * num_heads) dimension consistency
        PLUGIN_VALIDATE(in[0].desc.dims.d[0] == in[1].desc.dims.d[0]);
        PLUGIN_VALIDATE(in[0].desc.dims.d[0] == in[2].desc.dims.d[0]);

        // check S (sequence_length) dimension consistency
        PLUGIN_VALIDATE(in[0].desc.dims.d[1] == in[1].desc.dims.d[1]);
        PLUGIN_VALIDATE(in[0].desc.dims.d[1] == in[2].desc.dims.d[1]);
        PLUGIN_VALIDATE(in[0].desc.dims.d[1] == in[0].desc.dims.d[2]);

        // check K (2 * span) dimension consistency for in[1] and in[2]
        PLUGIN_VALIDATE(in[1].desc.dims.d[2] == 2 * mSpan);
        PLUGIN_VALIDATE(in[2].desc.dims.d[2] == 2 * mSpan);

        // Outputs (same dimension as in[0])
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(out[0].desc.dims.nbDims == 3);
        PLUGIN_VALIDATE(in[0].desc.dims.d[0] == out[0].desc.dims.d[0]);
        PLUGIN_VALIDATE(in[0].desc.dims.d[1] == out[0].desc.dims.d[1]);
        PLUGIN_VALIDATE(in[0].desc.dims.d[2] == out[0].desc.dims.d[2]);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

nvinfer1::DataType DisentangledAttentionPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputTypes != nullptr);
        PLUGIN_VALIDATE(nbInputs > 0);
        PLUGIN_VALIDATE(index == 0);
        return inputTypes[0]; // version 1, same as data1; version 2, same as data0
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nvinfer1::DataType{};
}

size_t DisentangledAttentionPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

void DisentangledAttentionPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* DisentangledAttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
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
    return kDEBERTA_PLUGIN_NAME;
}

char const* DisentangledAttentionPluginCreator::getPluginVersion() const noexcept
{
    return kDEBERTA_PLUGIN_VERSION;
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
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

IPluginV2DynamicExt* DisentangledAttentionPluginCreator::createPlugin(
    char const* /*name*/, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);

        // Set default invalid values (for assert in case when attributes are missing)
        int32_t span = 0;
        float factor = 0.F;
        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            std::string fieldName = fc->fields[i].name;
            if (fieldName.compare("span") == 0)
            {
                span = *static_cast<int32_t const*>(fc->fields[i].data);
            }
            if (fieldName.compare("factor") == 0)
            {
                factor = *static_cast<float const*>(fc->fields[i].data);
            }
        }

        PLUGIN_VALIDATE(span >= 0);
        PLUGIN_VALIDATE(factor > 0.F && factor < 1.F); // factor is 1/sqrt(3d), therefore must less than 1

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
    char const* /*name*/, void const* serialData, size_t serialLength) noexcept
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
