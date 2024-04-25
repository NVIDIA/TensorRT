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

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>

#include "common/serialize.hpp"
#include "scatterElementsPlugin.h"
#include "scatterElementsPluginKernel.h"

namespace nvinfer1
{
namespace plugin
{

std::map<std::string, ReductionType> const gReduceToEnum{
    {"add", ReductionType::kSUM},
    {"mean", ReductionType::kMEAN},
    {"mul", ReductionType::kMUL},
    {"min", ReductionType::kMIN},
    {"max", ReductionType::kMAX},
};

// Static class fields initialization
PluginFieldCollection ScatterElementsPluginCreator::gFC{};
std::vector<PluginField> ScatterElementsPluginCreator::gPluginAttributes;

namespace
{
constexpr char const* kSCATTER_ELEMENTS_NAME{"ScatterElements"};
constexpr char const* kSCATTER_ELEMENTS_VERSION{"1"};
} // namespace

ScatterElementsPlugin::ScatterElementsPlugin(ReductionType reduction, int32_t dim)
    : mReduction(reduction)
    , mAxis(dim)
{
}

ScatterElementsPlugin::ScatterElementsPlugin(std::string const& reduction, int32_t dim)
    : mReduction(gReduceToEnum.at(reduction))
    , mAxis(dim)
{
}

ScatterElementsPlugin::ScatterElementsPlugin(void const* serialData, size_t serialLength)
{
    deserialize_value(&serialData, &serialLength, &mReduction);
    deserialize_value(&serialData, &serialLength, &mAxis);
}

int32_t ScatterElementsPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t ScatterElementsPlugin::initialize() noexcept
{
    return 0;
}

char const* ScatterElementsPlugin::getPluginType() const noexcept
{
    return kSCATTER_ELEMENTS_NAME;
}

char const* ScatterElementsPlugin::getPluginVersion() const noexcept
{
    return kSCATTER_ELEMENTS_VERSION;
}

DimsExprs ScatterElementsPlugin::getOutputDimensions(
    int32_t index, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(nbInputs == 3);
        PLUGIN_VALIDATE(inputs);
        PLUGIN_VALIDATE(index <= kOUTPUT_TENSOR_IDX);
        // both outputs are of the same size
        DimsExprs out(inputs[kDATA_TENSOR_IDX]);
        return out;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs();
}

int32_t ScatterElementsPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc[kINDICES_TENSOR_IDX].type == DataType::kINT64);

        runScatterElementsKernel(outputs[kOUTPUT_TENSOR_IDX], inputs[kDATA_TENSOR_IDX], inputs[kUPDATES_TENSOR_IDX],
            inputs[kINDICES_TENSOR_IDX], outputDesc[kOUTPUT_TENSOR_IDX], inputDesc[kDATA_TENSOR_IDX],
            inputDesc[kUPDATES_TENSOR_IDX], inputDesc[kINDICES_TENSOR_IDX], mAxis, mReduction, stream);
        return 0;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return -1;
    }
}

size_t ScatterElementsPlugin::getSerializationSize() const noexcept
{
    auto ret = serialized_size(mReduction) + serialized_size(mAxis);
    return ret;
}

void ScatterElementsPlugin::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mReduction);
    serialize_value(&buffer, mAxis);
}

bool ScatterElementsPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inOut && pos < (nbInputs + nbOutputs));

        if (inOut[pos].format != PluginFormat::kLINEAR)
        {
            return false;
        }

        auto mytype = inOut[pos].type;
        auto firsttype = inOut[kDATA_TENSOR_IDX].type;

        // Only INT64 is supported for indices
        return pos == kINDICES_TENSOR_IDX ? (mytype == DataType::kINT64)
                                          : (mytype == firsttype)
                && (mytype == DataType::kFLOAT || mytype == DataType::kHALF
                    || (hasBfloat16AtomicAdd() && mytype == DataType::kBF16) || mytype == DataType::kINT32
                    || mytype == DataType::kINT64);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return false;
    }
}

void ScatterElementsPlugin::terminate() noexcept {}

void ScatterElementsPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* ScatterElementsPlugin::clone() const noexcept
{
    auto* plugin = new ScatterElementsPlugin(mReduction, mAxis);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void ScatterElementsPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(nbInputs == 3);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

DataType ScatterElementsPlugin::getOutputDataType(
    int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputTypes && nbInputs == 3 && index == kOUTPUT_TENSOR_IDX);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return inputTypes[kDATA_TENSOR_IDX];
}

size_t ScatterElementsPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

void ScatterElementsPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* ScatterElementsPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

//
// ScatterElementsPluginCreator
//

ScatterElementsPluginCreator::ScatterElementsPluginCreator()
{
    gPluginAttributes.clear();
    gPluginAttributes.emplace_back(PluginField("reduction"));
    gPluginAttributes.emplace_back(PluginField("axis"));
    gFC.nbFields = gPluginAttributes.size();
    gFC.fields = gPluginAttributes.data();
}

char const* ScatterElementsPluginCreator::getPluginName() const noexcept
{
    return kSCATTER_ELEMENTS_NAME;
}

char const* ScatterElementsPluginCreator::getPluginVersion() const noexcept
{
    return kSCATTER_ELEMENTS_VERSION;
}

PluginFieldCollection const* ScatterElementsPluginCreator::getFieldNames() noexcept
{
    return &gFC;
}

char const* ScatterElementsPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void ScatterElementsPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

IPluginV2DynamicExt* ScatterElementsPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    std::string reductionArg;
    int32_t axisArg = 0;
    ScatterElementsPlugin* plugin = nullptr;

    try
    {
        PLUGIN_VALIDATE(fc != nullptr);
        auto fields = fc->fields;

        std::set<std::string> requiredFields{"reduction"};
        plugin::validateRequiredAttributesExist(requiredFields, fc);

        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            PLUGIN_VALIDATE(fields[i].name != nullptr);
            PLUGIN_VALIDATE(fields[i].data != nullptr);
            if (strcmp(fields[i].name, "axis") == 0)
            {
                auto data = static_cast<int32_t const*>(fields[i].data);
                axisArg = *data;
            }
            else if (strcmp(fields[i].name, "reduction") == 0)
            {
                auto data = static_cast<char const*>(fields[i].data);
                reductionArg = std::string(data);
            }
        }

        PLUGIN_VALIDATE(gReduceToEnum.find(reductionArg) != gReduceToEnum.end(),
            (reductionArg + ": invalid value for 'reduction' plugin argument").c_str());

        plugin = new ScatterElementsPlugin(reductionArg, axisArg);
        plugin->setPluginNamespace(mNamespace.c_str());
    }
    catch (std::exception& e)
    {
        caughtError(e);
    }
    return plugin;
}

IPluginV2DynamicExt* ScatterElementsPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    ScatterElementsPlugin* plugin = new ScatterElementsPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

} // namespace plugin
} // namespace nvinfer1
