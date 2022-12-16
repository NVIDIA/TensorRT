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
#include "lReluPlugin.h"
#include "common/checkMacrosPlugin.h"
#include "common/kernel.h"

using namespace nvinfer1;
using nvinfer1::plugin::LReluPluginCreator;
using nvinfer1::plugin::LReLU;

static const char* LRELU_PLUGIN_VERSION{"1"};
static const char* LRELU_PLUGIN_NAME{"LReLU_TRT"};
PluginFieldCollection LReluPluginCreator::mFC{};
std::vector<PluginField> LReluPluginCreator::mPluginAttributes;

// LeakyReLU {{{
LReLU::LReLU(float negSlope)
    : mNegSlope(negSlope)
    , mBatchDim(1)
{
    PLUGIN_VALIDATE(negSlope >= 0.0F);
}

LReLU::LReLU(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char*>(buffer), *a = d;
    mNegSlope = read<float>(d);
    mBatchDim = read<int>(d);
    PLUGIN_VALIDATE(d == a + length);
}

int LReLU::getNbOutputs() const noexcept
{
    return 1;
}

Dims LReLU::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    PLUGIN_ASSERT(nbInputDims == 1);
    PLUGIN_ASSERT(index == 0);
    return inputs[0];
}

int LReLU::enqueue(
    int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    pluginStatus_t status = lReLUInference(stream, mBatchDim * batchSize, mNegSlope, inputData, outputData);
    return status;
}

size_t LReLU::getSerializationSize() const noexcept
{
    // mNegSlope, mBatchDim
    return sizeof(float) + sizeof(int);
}

void LReLU::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mNegSlope);
    write(d, mBatchDim);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void LReLU::configureWithFormat(const Dims* inputDims, int /* nbInputs */, const Dims* /* outputDims */, int nbOutputs,
    DataType type, PluginFormat format, int) noexcept
{
    PLUGIN_ASSERT(type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
    PLUGIN_ASSERT(mBatchDim == 1);
    PLUGIN_ASSERT(nbOutputs == 1);
    for (int i = 0; i < inputDims[0].nbDims; ++i)
    {
        mBatchDim *= inputDims[0].d[i];
    }
}

bool LReLU::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

int LReLU::initialize() noexcept
{
    return 0;
}

void LReLU::terminate() noexcept {}

size_t LReLU::getWorkspaceSize(int /* maxBatchSize */) const noexcept
{
    return 0;
}

const char* LReLU::getPluginType() const noexcept
{
    return LRELU_PLUGIN_NAME;
}

const char* LReLU::getPluginVersion() const noexcept
{
    return LRELU_PLUGIN_VERSION;
}

void LReLU::destroy() noexcept
{
    delete this;
}

IPluginV2* LReLU::clone() const noexcept
{
    try
    {
        IPluginV2* plugin = new LReLU(mNegSlope);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

LReluPluginCreator::LReluPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("negSlope", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* LReluPluginCreator::getPluginName() const noexcept
{
    return LRELU_PLUGIN_NAME;
}

const char* LReluPluginCreator::getPluginVersion() const noexcept
{
    return LRELU_PLUGIN_VERSION;
}

const PluginFieldCollection* LReluPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* LReluPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        PLUGIN_VALIDATE(fc->nbFields == 1);
        PLUGIN_VALIDATE(fields[0].type == PluginFieldType::kFLOAT32);
        PLUGIN_VALIDATE(!strcmp(fields[0].name, "negSlope"));
        float negSlope = *(static_cast<const float*>(fields[0].data));

        return new LReLU(negSlope);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* LReluPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call LReluPlugin::destroy()
        return new LReLU(serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
// LeakReLU }}}
