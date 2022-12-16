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

#include "splitGeLUPlugin.h"
#include "splitGeLUKernel.h"

using namespace nvinfer1;
using namespace plugin;

using nvinfer1::plugin::SplitGeLUPlugin;
using nvinfer1::plugin::SplitGeLUPluginCreator;

namespace
{
static std::string const kSPLIT_GELU_PLUGIN_NAME{"SplitGeLU"};
static std::string const kSPLIT_GELU_PLUGIN_VERSION{"1"};
size_t constexpr kSERIALIZATION_SIZE{3 * sizeof(float)};
} // namespace

PluginFieldCollection SplitGeLUPluginCreator::mFC{};
std::vector<PluginField> SplitGeLUPluginCreator::mPluginAttributes;

// class SplitGeLUPlugin
SplitGeLUPlugin::SplitGeLUPlugin(std::string const& name)
    : mName(name)
{
    mFDiv = 1.4140625F;
    mFAdd = 1.F;
    mFMul = 0.5F;
}

SplitGeLUPlugin::SplitGeLUPlugin(std::string const& name, void const* buffer, size_t length)
    : mName(name)
{
    PLUGIN_VALIDATE(buffer != nullptr);
    PLUGIN_VALIDATE(length == kSERIALIZATION_SIZE);

    auto const* d = static_cast<char const*>(buffer);
    auto const* a = d;

    mFDiv = read<float>(d);
    mFAdd = read<float>(d);
    mFMul = read<float>(d);

    PLUGIN_VALIDATE(d == a + length);
}

IPluginV2DynamicExt* SplitGeLUPlugin::clone() const noexcept
{
    try
    {
        auto p = new SplitGeLUPlugin(*this);
        p->setPluginNamespace(mNameSpace.c_str());
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

int32_t SplitGeLUPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DataType SplitGeLUPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    DataType ret{};
    try
    {
        PLUGIN_VALIDATE(inputTypes != nullptr);
        PLUGIN_VALIDATE(nbInputs > 0);
        ret = inputTypes[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return ret;
}

DimsExprs SplitGeLUPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    DimsExprs ret{};
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(nbInputs > 0);
        DimsExprs output = inputs[0];
        PLUGIN_VALIDATE(output.nbDims >= 3);
        output.d[2] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[0].d[2], *exprBuilder.constant(2));
        ret = output;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return ret;
}

bool SplitGeLUPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE(nbInputs + nbOutputs > 0);

        PLUGIN_VALIDATE(pos >= 0 && pos <= 1);

        if (pos == 0)
        {
            return (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF)
                && inOut[0].format == TensorFormat::kLINEAR;
        }

        PLUGIN_VALIDATE(pos < nbInputs + nbOutputs);
        return inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

void SplitGeLUPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
}

size_t SplitGeLUPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t SplitGeLUPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr);
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(inputs[0] != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);
        PLUGIN_VALIDATE(outputs[0] != nullptr);
        PLUGIN_VALIDATE(outputDesc != nullptr);

        int32_t const gridSize = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
        int32_t const nHalfHiddenSize = inputDesc[0].dims.d[2] / 2; // HHS

        if (inputDesc[0].type == DataType::kFLOAT)
        {
            auto const input = static_cast<float const*>(inputs[0]);
            auto output = static_cast<float*>(outputs[0]);
            launchSplitGeLUKernel<float>(stream, gridSize, nHalfHiddenSize, input, output, mFDiv, mFAdd, mFMul);
        }
        else
        {
            auto const input = static_cast<half const*>(inputs[0]);
            auto output = static_cast<half*>(outputs[0]);
            launchSplitGeLUKernel<half>(stream, gridSize, nHalfHiddenSize, input, output, mFDiv, mFAdd, mFMul);
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return -1;
}

void SplitGeLUPlugin::destroy() noexcept
{
    delete this;
}

int32_t SplitGeLUPlugin::initialize() noexcept
{
    return 0;
}

void SplitGeLUPlugin::terminate() noexcept {}

size_t SplitGeLUPlugin::getSerializationSize() const noexcept
{
    return kSERIALIZATION_SIZE;
}

void SplitGeLUPlugin::serialize(void* buffer) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(buffer != nullptr);
        auto* d = static_cast<char*>(buffer);
        auto* const a = d;
        write(d, mFDiv); // float
        write(d, mFAdd); // float
        write(d, mFMul); // float
        PLUGIN_VALIDATE(d == a + getSerializationSize());
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

void SplitGeLUPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(pluginNamespace != nullptr);
        mNameSpace = pluginNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* SplitGeLUPlugin::getPluginNamespace() const noexcept
{

    return mNameSpace.c_str();
}

char const* SplitGeLUPlugin::getPluginType() const noexcept
{
    return kSPLIT_GELU_PLUGIN_NAME.c_str();
}

char const* SplitGeLUPlugin::getPluginVersion() const noexcept
{
    return kSPLIT_GELU_PLUGIN_VERSION.c_str();
}

// class SplitGeLUPluginCreator
SplitGeLUPluginCreator::SplitGeLUPluginCreator()
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

SplitGeLUPluginCreator::~SplitGeLUPluginCreator() {}

IPluginV2* SplitGeLUPluginCreator::createPlugin(char const* name, PluginFieldCollection const* /*fc*/) noexcept
{
    try
    {
        PLUGIN_VALIDATE(name != nullptr);
        return new SplitGeLUPlugin(name);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SplitGeLUPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        PLUGIN_VALIDATE(name != nullptr);
        PLUGIN_VALIDATE(serialData != nullptr);
        return new SplitGeLUPlugin(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

char const* SplitGeLUPluginCreator::getPluginName() const noexcept
{
    return kSPLIT_GELU_PLUGIN_NAME.c_str();
}

char const* SplitGeLUPluginCreator::getPluginVersion() const noexcept
{
    return kSPLIT_GELU_PLUGIN_VERSION.c_str();
}

PluginFieldCollection const* SplitGeLUPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}
