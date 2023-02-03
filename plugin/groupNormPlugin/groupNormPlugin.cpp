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

#include "groupNormPlugin.h"
#include "groupNormKernel.h"

#include <cmath>

using namespace nvinfer1;
using namespace plugin;

using nvinfer1::plugin::GroupNormPlugin;
using nvinfer1::plugin::GroupNormPluginCreator;

namespace
{
static std::string const kGROUP_NORM_PLUGIN_NAME{"GroupNorm"};
static std::string const kGROUP_NORM_PLUGIN_VERSION{"1"};
size_t constexpr kSERIALIZATION_SIZE{sizeof(float) + sizeof(int32_t)};
} // namespace

int32_t findMaxDivisor(int32_t n, int32_t maxAllowedDivisor)
{
    int32_t maxDivisor = -1;
    for (int32_t i = 1; i <= std::sqrt(n); i++)
    {
        if (n % i == 0)
        {
            int32_t divisor1 = n / i;
            int32_t divisor2 = i;

            if (divisor1 > maxDivisor && divisor1 < maxAllowedDivisor)
            {
                maxDivisor = divisor1;
            }
            if (divisor2 > maxDivisor && divisor2 < maxAllowedDivisor)
            {
                maxDivisor = divisor2;
            }
        }
    }
    return maxDivisor;
}

// class GroupNormPlugin
GroupNormPlugin::GroupNormPlugin(std::string const& name, float epsilon, int32_t bSwish)
    : mName(name)
    , mEpsilon(epsilon)
    , mBSwish(bSwish)
{
    memset(&mParams, 0, sizeof(mParams));
}

GroupNormPlugin::GroupNormPlugin(std::string const& name, void const* buffer, size_t length)
    : mName(name)
{
    PLUGIN_VALIDATE(buffer != nullptr);
    PLUGIN_VALIDATE(length == kSERIALIZATION_SIZE);

    auto const* d = static_cast<char const*>(buffer);
    auto const* a = d;

    mEpsilon = read<float>(d);
    mBSwish = read<int32_t>(d);

    PLUGIN_VALIDATE(d == a + length);
}

IPluginV2DynamicExt* GroupNormPlugin::clone() const noexcept
{
    try
    {
        auto p = new GroupNormPlugin(*this);
        p->setPluginNamespace(mNameSpace.c_str());
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

int32_t GroupNormPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DataType GroupNormPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
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

DimsExprs GroupNormPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    DimsExprs ret{};
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(nbInputs > 0);
        ret = inputs[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return ret;
}

bool GroupNormPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(pos >= 0 && pos <= 3);
        if (pos == 0)
        {
            return inOut[0].type == DataType::kHALF && inOut[0].format == TensorFormat::kHWC8;
        }
        if (pos == 1 || pos == 2)
        {
            return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
        }
        if (pos == 3)
        {
            return inOut[pos].format == inOut[0].format && inOut[pos].type == inOut[0].type;
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

void GroupNormPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
}

size_t GroupNormPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return getWorkspaceSizeInBytes();
}

size_t GroupNormPlugin::getWorkspaceSizeInBytes() const
{
    return (sizeof(float) * 2) * 32 * 32; // sizeof(float2) * maxBatchSize * maxNumberOfGroup. float2
                                          // contians two buffers for sum and squared sum
}

int32_t GroupNormPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        int32_t cPerBlock = 320;
        int32_t maxBlocksPerHW = 1024;

        switch (inputDesc[0].dims.d[1])
        {
        case 960:
        case 1920: cPerBlock = 480; break;
        case 512:
        case 256: cPerBlock = 256; break;
        case 128: cPerBlock = 128; break;
        default: cPerBlock = 320;
        }

        mParams.withSwish = bool(mBSwish);
        mParams.dst = static_cast<half*>(outputs[0]);
        mParams.src = static_cast<half const*>(inputs[0]);
        mParams.gamma = static_cast<float const*>(inputs[1]);
        mParams.beta = static_cast<float const*>(inputs[2]);
        mParams.redBuffer = static_cast<float*>(workspace);
        mParams.n = inputDesc[0].dims.d[0];
        mParams.h = inputDesc[0].dims.d[2];
        mParams.w = inputDesc[0].dims.d[3];
        mParams.c = inputDesc[0].dims.d[1];
        mParams.groups = 32;
        mParams.hw = mParams.h * mParams.w;
        const int32_t blocksPerHW = findMaxDivisor(mParams.hw, maxBlocksPerHW);
        mParams.hwPerBlock = divUp(mParams.hw, blocksPerHW);
        mParams.cPerBlock = cPerBlock;
        mParams.cPerGroup = mParams.c / mParams.groups;
        mParams.hwc = mParams.hw * mParams.c;
        mParams.invHWC = 1.F / (float) (mParams.hw * mParams.cPerGroup);
        mParams.groupsPerBlock = cPerBlock / mParams.cPerGroup;

        cudaMemsetAsync(mParams.redBuffer, 0, getWorkspaceSizeInBytes(), stream);
        groupNormNHWCSum(mParams, stream);
        groupNormNHWCScale(mParams, stream);

        return 0;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return -1;
}

void GroupNormPlugin::destroy() noexcept
{
    delete this;
}

int32_t GroupNormPlugin::initialize() noexcept
{
    return 0;
}

void GroupNormPlugin::terminate() noexcept {}

size_t GroupNormPlugin::getSerializationSize() const noexcept
{
    return kSERIALIZATION_SIZE;
}

void GroupNormPlugin::serialize(void* buffer) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(buffer != nullptr);
        auto* d = static_cast<char*>(buffer);
        auto* a = d;
        write(d, mEpsilon); // float
        write(d, mBSwish);  // int32_t
        PLUGIN_VALIDATE(d == a + getSerializationSize());
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

void GroupNormPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNameSpace = pluginNamespace;
}

char const* GroupNormPlugin::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

char const* GroupNormPlugin::getPluginType() const noexcept
{
    return kGROUP_NORM_PLUGIN_NAME.c_str();
}

char const* GroupNormPlugin::getPluginVersion() const noexcept
{
    return kGROUP_NORM_PLUGIN_VERSION.c_str();
}

// class GroupNormPluginCreator
PluginFieldCollection GroupNormPluginCreator::mFC{};
std::vector<PluginField> GroupNormPluginCreator::mPluginAttributes;

GroupNormPluginCreator::GroupNormPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("bSwish", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

GroupNormPluginCreator::~GroupNormPluginCreator() {}

IPluginV2* GroupNormPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        float epsilon = 1e-5F;
        int32_t bSwish = 0;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            if (fc->fields[i].name == std::string("epsilon"))
            {
                epsilon = static_cast<float>(*(static_cast<float const*>((fc->fields[i].data))));
                continue;
            }
            if (fc->fields[i].name == std::string("bSwish"))
            {
                bSwish = static_cast<int32_t>(*(static_cast<int32_t const*>((fc->fields[i].data))));
                continue;
            }
        }
        return new GroupNormPlugin(name, epsilon, bSwish);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* GroupNormPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        return new GroupNormPlugin(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

char const* GroupNormPluginCreator::getPluginName() const noexcept
{
    return kGROUP_NORM_PLUGIN_NAME.c_str();
}

char const* GroupNormPluginCreator::getPluginVersion() const noexcept
{
    return kGROUP_NORM_PLUGIN_VERSION.c_str();
}

PluginFieldCollection const* GroupNormPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}
