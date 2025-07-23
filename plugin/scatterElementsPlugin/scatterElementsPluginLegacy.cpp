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

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>

#include "common/serialize.hpp"
#include "scatterElementsPluginKernel.h"
#include "scatterElementsPluginLegacy.h"

namespace nvinfer1::plugin
{

std::unordered_map<std::string, ReductionType> const kREDUCE_STR_TO_ENUM{
    {"add", ReductionType::kSUM},
    {"mean", ReductionType::kMEAN},
    {"mul", ReductionType::kMUL},
    {"min", ReductionType::kMIN},
    {"max", ReductionType::kMAX},
};

namespace
{
constexpr char const* kSCATTER_ELEMENTS_NAME{"ScatterElements"};
constexpr char const* kSCATTER_ELEMENTS_VERSION{"1"};
} // namespace

ScatterElementsPluginV2::ScatterElementsPluginV2(ReductionType reduction, int32_t dim)
    : mReduction(reduction)
    , mAxis(dim)
{
}

ScatterElementsPluginV2::ScatterElementsPluginV2(std::string const& reduction, int32_t dim)
    : mReduction(kREDUCE_STR_TO_ENUM.at(reduction))
    , mAxis(dim)
{
}

ScatterElementsPluginV2::ScatterElementsPluginV2(void const* serialData, size_t serialLength)
{
    deserialize_value(&serialData, &serialLength, &mReduction);
    deserialize_value(&serialData, &serialLength, &mAxis);
}

int32_t ScatterElementsPluginV2::getNbOutputs() const noexcept
{
    return 1;
}

int32_t ScatterElementsPluginV2::initialize() noexcept
{
    return 0;
}

char const* ScatterElementsPluginV2::getPluginType() const noexcept
{
    return kSCATTER_ELEMENTS_NAME;
}

char const* ScatterElementsPluginV2::getPluginVersion() const noexcept
{
    return kSCATTER_ELEMENTS_VERSION;
}

DimsExprs ScatterElementsPluginV2::getOutputDimensions(
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

int32_t ScatterElementsPluginV2::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
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

size_t ScatterElementsPluginV2::getSerializationSize() const noexcept
{
    auto ret = serialized_size(mReduction) + serialized_size(mAxis);
    return ret;
}

void ScatterElementsPluginV2::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mReduction);
    serialize_value(&buffer, mAxis);
}

bool ScatterElementsPluginV2::supportsFormatCombination(
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

void ScatterElementsPluginV2::terminate() noexcept {}

void ScatterElementsPluginV2::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* ScatterElementsPluginV2::clone() const noexcept
{
    auto* plugin = new ScatterElementsPluginV2(mReduction, mAxis);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void ScatterElementsPluginV2::configurePlugin(
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

DataType ScatterElementsPluginV2::getOutputDataType(
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

size_t ScatterElementsPluginV2::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

void ScatterElementsPluginV2::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* ScatterElementsPluginV2::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

//
// ScatterElementsPluginV2Creator
//

ScatterElementsPluginV2Creator::ScatterElementsPluginV2Creator()
{
    gPluginAttributes.clear();
    gPluginAttributes.emplace_back(PluginField("reduction"));
    gPluginAttributes.emplace_back(PluginField("axis"));
    gFC.nbFields = gPluginAttributes.size();
    gFC.fields = gPluginAttributes.data();
}

char const* ScatterElementsPluginV2Creator::getPluginName() const noexcept
{
    return kSCATTER_ELEMENTS_NAME;
}

char const* ScatterElementsPluginV2Creator::getPluginVersion() const noexcept
{
    return kSCATTER_ELEMENTS_VERSION;
}

PluginFieldCollection const* ScatterElementsPluginV2Creator::getFieldNames() noexcept
{
    return &gFC;
}

char const* ScatterElementsPluginV2Creator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void ScatterElementsPluginV2Creator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

IPluginV2DynamicExt* ScatterElementsPluginV2Creator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    std::string reductionArg;
    int32_t axisArg = 0;
    ScatterElementsPluginV2* plugin = nullptr;

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

        PLUGIN_VALIDATE(kREDUCE_STR_TO_ENUM.find(reductionArg) != kREDUCE_STR_TO_ENUM.end(),
            (reductionArg + ": invalid value for 'reduction' plugin argument").c_str());

        plugin = new ScatterElementsPluginV2(reductionArg, axisArg);
        plugin->setPluginNamespace(mNamespace.c_str());
    }
    catch (std::exception& e)
    {
        caughtError(e);
    }
    return plugin;
}

IPluginV2DynamicExt* ScatterElementsPluginV2Creator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    ScatterElementsPluginV2* plugin = new ScatterElementsPluginV2(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

} // namespace nvinfer1::plugin
