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
#include "scatterElementsPlugin.h"
#include "scatterElementsPluginKernel.h"

namespace nvinfer1::plugin
{

std::unordered_map<std::string, ReductionType> const kREDUCE_STR_TO_ENUM{
    {"add", ReductionType::kSUM},
    {"mean", ReductionType::kMEAN},
    {"mul", ReductionType::kMUL},
    {"min", ReductionType::kMIN},
    {"max", ReductionType::kMAX},
};
std::unordered_map<ReductionType, std::string> const kREDUCE_ENUM_TO_STR{
    {ReductionType::kSUM, "add"},
    {ReductionType::kMEAN, "mean"},
    {ReductionType::kMUL, "mul"},
    {ReductionType::kMIN, "min"},
    {ReductionType::kMAX, "max"},
};

namespace
{
constexpr char const* kSCATTER_PLUGIN_VERSION{"2"};
constexpr char const* kSCATTER_PLUGIN_NAME{"ScatterElements"};
} // namespace

ScatterElementsPluginV3::ScatterElementsPluginV3(ReductionType reduction, int32_t dim)
    : mReduction(reduction)
    , mAxis(dim)
{
}

ScatterElementsPluginV3::ScatterElementsPluginV3(std::string const& reduction, int32_t dim)
    : mReduction(kREDUCE_STR_TO_ENUM.at(reduction))
    , mAxis(dim)
{
}

int32_t ScatterElementsPluginV3::getNbOutputs() const noexcept
{
    return 1;
}

IPluginCapability* ScatterElementsPluginV3::getCapabilityInterface(PluginCapabilityType type) noexcept
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

char const* ScatterElementsPluginV3::getPluginVersion() const noexcept
{
    return kSCATTER_PLUGIN_VERSION;
}

int32_t ScatterElementsPluginV3::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* shapeInputs, int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_ASSERT(nbInputs == 3);
        PLUGIN_ASSERT(inputs != nullptr);
        PLUGIN_ASSERT(nbOutputs == 1);
        outputs[kOUTPUT_TENSOR_IDX] = inputs[kDATA_TENSOR_IDX];
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

int32_t ScatterElementsPluginV3::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc[kINDICES_TENSOR_IDX].type == DataType::kINT64);

        runScatterElementsKernel(outputs[kOUTPUT_TENSOR_IDX], inputs[kDATA_TENSOR_IDX], inputs[kUPDATES_TENSOR_IDX],
            inputs[kINDICES_TENSOR_IDX], outputDesc[kOUTPUT_TENSOR_IDX], inputDesc[kDATA_TENSOR_IDX],
            inputDesc[kUPDATES_TENSOR_IDX], inputDesc[kINDICES_TENSOR_IDX], mAxis, mReduction, stream);
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return -1;
}

int32_t ScatterElementsPluginV3::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(in != nullptr);
    PLUGIN_ASSERT(out != nullptr);
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(nbInputs == 3);
    auto rank = in[0].dims.nbDims;
    // rank of input should be >=1
    PLUGIN_ASSERT(rank >= 1);
    // rank of indices should be same as rank of data
    PLUGIN_ASSERT(in[1].dims.nbDims == rank);
    // rank and shape of updates should be same as indices
    PLUGIN_ASSERT(in[2].dims.nbDims == rank);
    PLUGIN_VALIDATE(std::equal(in[2].dims.d, in[2].dims.d + rank, in[1].dims.d));
    return pluginStatus_t::STATUS_SUCCESS;
}

PluginFieldCollection const* ScatterElementsPluginV3::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    // "reduction" field is serialized as string
    mDataToSerialize.emplace_back("reduction", kREDUCE_ENUM_TO_STR.at(mReduction).c_str(), PluginFieldType::kCHAR,
        kREDUCE_ENUM_TO_STR.at(mReduction).size());
    mDataToSerialize.emplace_back("axis", &mAxis, PluginFieldType::kINT32, 1);

    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

bool ScatterElementsPluginV3::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inOut && pos < (nbInputs + nbOutputs));

        if (inOut[pos].desc.format != PluginFormat::kLINEAR)
        {
            return false;
        }

        auto currentType = inOut[pos].desc.type;
        auto firstType = inOut[kDATA_TENSOR_IDX].desc.type;

        // Only INT64 is supported for indices
        return pos == kINDICES_TENSOR_IDX ? (currentType == DataType::kINT64)
                                          : (currentType == firstType)
                && (currentType == DataType::kFLOAT || currentType == DataType::kHALF
                    || (hasBfloat16AtomicAdd() && currentType == DataType::kBF16) || currentType == DataType::kINT32
                    || currentType == DataType::kINT64);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return false;
    }
}

ScatterElementsPluginV3* ScatterElementsPluginV3::clone() noexcept
{
    try
    {
        auto* plugin = new ScatterElementsPluginV3(mReduction, mAxis);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV3* ScatterElementsPluginV3::attachToContext(IPluginResourceContext* context) noexcept
{
    ScatterElementsPluginV3* obj = clone();
    return obj;
}

int32_t ScatterElementsPluginV3::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(nbInputs == 3);
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

int32_t ScatterElementsPluginV3::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_ASSERT(inputTypes != nullptr);
        PLUGIN_ASSERT(nbInputs == 3);
        PLUGIN_ASSERT(nbOutputs == 1);
        outputTypes[kOUTPUT_TENSOR_IDX] = inputTypes[kDATA_TENSOR_IDX];
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

size_t ScatterElementsPluginV3::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

void ScatterElementsPluginV3::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_ASSERT(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* ScatterElementsPluginV3::getPluginName() const noexcept
{
    return kSCATTER_PLUGIN_NAME;
}

char const* ScatterElementsPluginV3::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

//
// ScatterElementsPluginV3Creator
//

ScatterElementsPluginV3Creator::ScatterElementsPluginV3Creator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> guard(sMutex);
    gPluginAttributes.clear();
    gPluginAttributes.emplace_back(PluginField("reduction"));
    gPluginAttributes.emplace_back(PluginField("axis"));
    gFC.nbFields = gPluginAttributes.size();
    gFC.fields = gPluginAttributes.data();
}

char const* ScatterElementsPluginV3Creator::getPluginName() const noexcept
{
    return kSCATTER_PLUGIN_NAME;
}

char const* ScatterElementsPluginV3Creator::getPluginVersion() const noexcept
{
    return kSCATTER_PLUGIN_VERSION;
}

PluginFieldCollection const* ScatterElementsPluginV3Creator::getFieldNames() noexcept
{
    return &gFC;
}

char const* ScatterElementsPluginV3Creator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void ScatterElementsPluginV3Creator::setPluginNamespace(char const* libNamespace) noexcept
{
    PLUGIN_VALIDATE(libNamespace != nullptr);
    mNamespace = libNamespace;
}

IPluginV3* ScatterElementsPluginV3Creator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    std::string reductionArg;
    int32_t axisArg = 0;
    ScatterElementsPluginV3* plugin = nullptr;

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
                reductionArg = fields[i].length != -1 ? std::string(data, fields[i].length) : std::string(data);
            }
        }

        PLUGIN_VALIDATE(kREDUCE_STR_TO_ENUM.find(reductionArg) != kREDUCE_STR_TO_ENUM.end(),
            (reductionArg + ": invalid value for 'reduction' plugin argument").c_str());

        plugin = new ScatterElementsPluginV3(reductionArg, axisArg);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

} // namespace nvinfer1::plugin
