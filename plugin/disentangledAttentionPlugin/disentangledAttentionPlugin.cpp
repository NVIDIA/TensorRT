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

#include "disentangledAttentionPlugin.h"
#include "NvInferPlugin.h"
#include <cuda_fp16.h>
#include <numeric>
#include <optional>
#include <stdexcept>

using namespace nvinfer1;
using nvinfer1::plugin::DisentangledAttentionPlugin;
using nvinfer1::plugin::DisentangledAttentionPluginCreator;

REGISTER_TENSORRT_PLUGIN(DisentangledAttentionPluginCreator);

namespace
{
constexpr char const* kDEBERTA_PLUGIN_NAME{"DisentangledAttention_TRT"};
constexpr char const* kDEBERTA_PLUGIN_VERSION{"2"};
} // namespace

DisentangledAttentionPlugin::DisentangledAttentionPlugin()
    : mSpan(0)
    , mFactor(0.0f)
{
}

DisentangledAttentionPlugin::DisentangledAttentionPlugin(int32_t span, float factor)
    : mSpan(span)
    , mFactor(factor)
{
}

// IPluginV3OneCore methods

int32_t DisentangledAttentionPlugin::getNbOutputs() const noexcept
{
    return 1;
}

char const* DisentangledAttentionPlugin::getPluginName() const noexcept
{
    return kDEBERTA_PLUGIN_NAME;
}

char const* DisentangledAttentionPlugin::getPluginVersion() const noexcept
{
    return kDEBERTA_PLUGIN_VERSION;
}

IPluginV3* DisentangledAttentionPlugin::clone() noexcept
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

void DisentangledAttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace;
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

IPluginCapability* DisentangledAttentionPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

PluginFieldCollection const* DisentangledAttentionPlugin::getFieldsToSerialize() noexcept
{
    try
    {
        mDataToSerialize.clear();

        mDataToSerialize.emplace_back("span", &mSpan, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("factor", &mFactor, PluginFieldType::kFLOAT32, 1);

        mFCToSerialize.nbFields = mDataToSerialize.size();
        mFCToSerialize.fields = mDataToSerialize.data();

        return &mFCToSerialize;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

// IPluginV3OneBuild methods

int32_t DisentangledAttentionPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* shapeInputs, int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(nbInputs == 3);
        PLUGIN_VALIDATE(outputs != nullptr);
        PLUGIN_VALIDATE(nbOutputs == 1);

        // Output has the same shape as the first input
        outputs[0] = inputs[0];

        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

int32_t DisentangledAttentionPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(in != nullptr && out != nullptr && nbInputs == 3 && nbOutputs == 1);

        // Validate input and output shapes
        for (int32_t i = 0; i < nbInputs; i++)
        {
            PLUGIN_VALIDATE(in[i].desc.dims.nbDims == in[0].desc.dims.nbDims);
        }

        // Check data types are consistent
        PLUGIN_VALIDATE(in[0].desc.type == in[1].desc.type && in[0].desc.type == in[2].desc.type);
        PLUGIN_VALIDATE(out[0].desc.type == in[0].desc.type);

        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

int32_t DisentangledAttentionPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputTypes != nullptr && outputTypes != nullptr);
        PLUGIN_VALIDATE(nbInputs == 3 && nbOutputs == 1);

        // Output has the same data type as the first input
        outputTypes[0] = inputTypes[0];

        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

bool DisentangledAttentionPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_ASSERT(inOut && pos < (nbInputs + nbOutputs));

        // All inputs and outputs should have the same precision type
        bool const consistentFloatPrecision = (inOut[pos].desc.type == inOut[0].desc.type);

        return (inOut[pos].desc.type == DataType::kINT8 || inOut[pos].desc.type == DataType::kHALF
                   || inOut[pos].desc.type == DataType::kFLOAT)
            && inOut[pos].desc.format == PluginFormat::kLINEAR && consistentFloatPrecision;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

// IPluginV3OneRuntime methods

template <typename TDataType>
void DisentangledAttentionPlugin::enqueueType(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, cudaStream_t stream, TDataType factor)
{
    Dims dims0 = inputDesc[0].dims;
    Dims dims1 = inputDesc[1].dims;
    Dims dims2 = inputDesc[2].dims;
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

int32_t DisentangledAttentionPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr);

        switch (inputDesc[0].type)
        {
        case DataType::kFLOAT: enqueueType<float>(inputDesc, outputDesc, inputs, outputs, stream, mFactor); break;
        case DataType::kHALF:
            enqueueType<__half>(inputDesc, outputDesc, inputs, outputs, stream, __float2half(mFactor));
            break;
        case DataType::kINT8:
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

size_t DisentangledAttentionPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t DisentangledAttentionPlugin::onShapeChange(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr && outputs != nullptr);
        PLUGIN_VALIDATE(nbInputs == 3 && nbOutputs == 1);

        // Check that all inputs have the same data type
        DataType dataType = inputs[0].type;
        PLUGIN_VALIDATE(inputs[1].type == dataType && inputs[2].type == dataType);

        // Check that output has the same data type
        PLUGIN_VALIDATE(outputs[0].type == dataType);

        // Validate dimensions
        PLUGIN_VALIDATE(inputs[0].dims.nbDims == inputs[1].dims.nbDims);
        PLUGIN_VALIDATE(inputs[0].dims.nbDims == inputs[2].dims.nbDims);
        PLUGIN_VALIDATE(outputs[0].dims.nbDims == inputs[0].dims.nbDims);

        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

IPluginV3* DisentangledAttentionPlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    try
    {
        return this->clone();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

// -------------------- Creator class Implementation --------------------

DisentangledAttentionPluginCreator::DisentangledAttentionPluginCreator()
{
    mPluginAttributes.clear();
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

IPluginV3* DisentangledAttentionPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const* fields = fc->fields;
        std::optional<int32_t> span;
        std::optional<float> factor;

        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "span"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                span = *static_cast<int32_t const*>(fields[i].data);
            }
            else if (!strcmp(attrName, "factor"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                factor = *static_cast<float const*>(fields[i].data);
            }
        }

        // Validate that all required fields were found
        PLUGIN_VALIDATE(span.has_value(), "Required attribute 'span' not found");
        PLUGIN_VALIDATE(factor.has_value(), "Required attribute 'factor' not found");
        PLUGIN_VALIDATE(span.value() >= 0);
        PLUGIN_VALIDATE(
            factor.value() > 0.F && factor.value() < 1.F); // factor is 1/sqrt(3d), therefore must less than 1

        auto* plugin = new DisentangledAttentionPlugin(span.value(), factor.value());
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void DisentangledAttentionPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* DisentangledAttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
