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

#include "seqLen2SpatialKernel.h"
#include "seqLen2SpatialPlugin.h"

using namespace nvinfer1;
using namespace plugin;

using nvinfer1::plugin::SeqLen2SpatialPlugin;
using nvinfer1::plugin::SeqLen2SpatialPluginCreator;

namespace
{
static std::string const kSEQLEN2SPATIAL_PLUGIN_NAME{"SeqLen2Spatial"};
static std::string const kSEQLEN2SPATIAL_PLUGIN_VERSION{"1"};
size_t constexpr kSERIALIZATION_SIZE{0};
} // namespace

SeqLen2SpatialPlugin::SeqLen2SpatialPlugin(std::string const& name)
    : mName(name)
{
}

SeqLen2SpatialPlugin::SeqLen2SpatialPlugin(std::string const& name, void const* buffer, size_t length)
    : mName(name)
{
}

IPluginV2DynamicExt* SeqLen2SpatialPlugin::clone() const noexcept
{
    try
    {
        auto plugin = new SeqLen2SpatialPlugin(*this);
        plugin->setPluginNamespace(mNameSpace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

int32_t SeqLen2SpatialPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DataType SeqLen2SpatialPlugin::getOutputDataType(
    int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
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

DimsExprs SeqLen2SpatialPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    return inputs[3];
}

bool SeqLen2SpatialPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE(nbInputs + nbOutputs > 0);
        PLUGIN_VALIDATE(pos < nbInputs + nbOutputs);
        PLUGIN_VALIDATE(pos >= 0 && pos <= 4);

        if (pos == 0)
        {
            return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF)
                && inOut[pos].format == TensorFormat::kLINEAR;
        }

        if (pos == 1 || pos == 2)
        {
            return inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format;
        }

        if (pos == 3)
        {
            return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
        }

        if (pos == 4)
        {
            return inOut[pos].type == inOut[0].type
                && ((inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kHWC)
                    || (inOut[pos].type == DataType::kHALF && inOut[pos].format == TensorFormat::kHWC8));
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

void SeqLen2SpatialPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
}

size_t SeqLen2SpatialPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t SeqLen2SpatialPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr);
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);
        PLUGIN_VALIDATE(outputDesc != nullptr);

        int32_t const BS = inputDesc[0].dims.d[0];
        int32_t const HW = inputDesc[0].dims.d[1];
        int32_t const C = inputDesc[0].dims.d[2];
        int32_t const gridSize = BS * HW;
        nvinfer1::DataType const dtype = inputDesc[0].type;
        launchSeqLen2SpatialKernel(inputs, outputs, dtype, gridSize, C, stream);
        return 0;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return -1;
}

void SeqLen2SpatialPlugin::destroy() noexcept
{
    delete this;
}

int32_t SeqLen2SpatialPlugin::initialize() noexcept
{
    return 0;
}

void SeqLen2SpatialPlugin::terminate() noexcept {}

size_t SeqLen2SpatialPlugin::getSerializationSize() const noexcept
{
    return kSERIALIZATION_SIZE;
}

void SeqLen2SpatialPlugin::serialize(void* buffer) const noexcept
{
}

void SeqLen2SpatialPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNameSpace = pluginNamespace;
}

char const* SeqLen2SpatialPlugin::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

char const* SeqLen2SpatialPlugin::getPluginType() const noexcept
{
    return kSEQLEN2SPATIAL_PLUGIN_NAME.c_str();
}

char const* SeqLen2SpatialPlugin::getPluginVersion() const noexcept
{
    return kSEQLEN2SPATIAL_PLUGIN_VERSION.c_str();
}

PluginFieldCollection SeqLen2SpatialPluginCreator::mFC{};
std::vector<PluginField> SeqLen2SpatialPluginCreator::mPluginAttributes;

SeqLen2SpatialPluginCreator::SeqLen2SpatialPluginCreator()
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

SeqLen2SpatialPluginCreator::~SeqLen2SpatialPluginCreator() {}

IPluginV2* SeqLen2SpatialPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        return new SeqLen2SpatialPlugin(name);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SeqLen2SpatialPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        return new SeqLen2SpatialPlugin(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

char const* SeqLen2SpatialPluginCreator::getPluginName() const noexcept
{
    return kSEQLEN2SPATIAL_PLUGIN_NAME.c_str();
}

char const* SeqLen2SpatialPluginCreator::getPluginVersion() const noexcept
{
    return kSEQLEN2SPATIAL_PLUGIN_VERSION.c_str();
}

PluginFieldCollection const* SeqLen2SpatialPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}
