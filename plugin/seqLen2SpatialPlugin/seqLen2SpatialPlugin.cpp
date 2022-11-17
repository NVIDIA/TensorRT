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

#include "seqLen2SpatialPlugin.h"
#include "seqLen2SpatialKernel.h"

using namespace nvinfer1;
using namespace plugin;

using nvinfer1::plugin::SeqLen2SpatialPlugin;
using nvinfer1::plugin::SeqLen2SpatialPluginCreator;

namespace
{
char const* kSEQLEN2SPATIAL_PLUGIN_NAME{"SeqLen2Spatial"};
char const* kSEQLEN2SPATIAL_PLUGIN_VERSION{"1"};
size_t constexpr kSERIALIZATION_SIZE{2 * sizeof(int32_t)};
} // namespace

SeqLen2SpatialPlugin::SeqLen2SpatialPlugin(std::string const& name, int32_t height, int32_t width)
    : mName(name)
    , mHeight(height)
    , mWidth(width)
{
}

SeqLen2SpatialPlugin::SeqLen2SpatialPlugin(std::string const& name, void const* buffer, size_t length)
    : mName(name)
{
    PLUGIN_VALIDATE(buffer != nullptr);
    PLUGIN_VALIDATE(length == kSERIALIZATION_SIZE);

    char const* d = static_cast<char const*>(buffer);
    char const* a = d;

    mHeight = read<int32_t>(d);
    mWidth = read<int32_t>(d);

    PLUGIN_VALIDATE(d == a + length);
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

DataType SeqLen2SpatialPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

DimsExprs SeqLen2SpatialPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    DimsExprs inputDims = inputs[0];
    DimsExprs outputDims;
    outputDims.nbDims = 4;
    outputDims.d[0] = inputDims.d[0];
    outputDims.d[1] = inputDims.d[2];
    outputDims.d[2] = exprBuilder.constant(mHeight);
    outputDims.d[3] = exprBuilder.constant(mWidth);
    return outputDims;
}

bool SeqLen2SpatialPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    switch (pos)
    {
    case 0:
        return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF) && inOut[pos].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format;
    case 2:
        return inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format;
    case 3:
        return inOut[pos].type == inOut[0].type
            && ((inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kHWC)
            || (inOut[pos].type == DataType::kHALF && inOut[pos].format == TensorFormat::kHWC8));
    default: // should NOT be here!
        return false;
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
    int32_t const BS = inputDesc[0].dims.d[0];
    int32_t const HW = inputDesc[0].dims.d[1];
    int32_t const C = inputDesc[0].dims.d[2];
    int32_t const gridSize = BS * HW;
    nvinfer1::DataType const dtype = inputDesc[0].type;
    try
    {
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
    PLUGIN_ASSERT(buffer != nullptr);
    char* d = static_cast<char*>(buffer);
    char* a = d;
    write(d, mHeight); // int32_t
    write(d, mWidth); // int32_t
    PLUGIN_ASSERT(d == a + getSerializationSize());
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
    return kSEQLEN2SPATIAL_PLUGIN_NAME;
}

char const* SeqLen2SpatialPlugin::getPluginVersion() const noexcept
{
    return kSEQLEN2SPATIAL_PLUGIN_VERSION;
}

PluginFieldCollection SeqLen2SpatialPluginCreator::mFC{};
std::vector<PluginField> SeqLen2SpatialPluginCreator::mPluginAttributes;

SeqLen2SpatialPluginCreator::SeqLen2SpatialPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("height", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("width", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

SeqLen2SpatialPluginCreator::~SeqLen2SpatialPluginCreator() {}

IPluginV2* SeqLen2SpatialPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const* fields = fc->fields;

        // default values
        int32_t mHeight{0};
        int32_t mWidth{0};

        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "height"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                mHeight = static_cast<int32_t>(*(static_cast<int32_t const*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "width"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                mWidth = static_cast<int32_t>(*(static_cast<int32_t const*>(fields[i].data)));
            }
        }
        return new SeqLen2SpatialPlugin(name, mHeight, mWidth);
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
        PLUGIN_VALIDATE(serialData != nullptr);
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
    return kSEQLEN2SPATIAL_PLUGIN_NAME;
}

char const* SeqLen2SpatialPluginCreator::getPluginVersion() const noexcept
{
    return kSEQLEN2SPATIAL_PLUGIN_VERSION;
}

PluginFieldCollection const* SeqLen2SpatialPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}
