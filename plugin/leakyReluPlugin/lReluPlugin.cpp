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
#include "lReluPlugin.h"
#include "common/checkMacrosPlugin.h"
#include "common/kernels/kernel.h"

namespace nvinfer1::plugin
{
static char const* const kLRELU_PLUGIN_VERSION{"1"};
static char const* const kLRELU_PLUGIN_NAME{"LReLU_TRT"};

// LeakyReLU {{{
LReLU::LReLU(float negSlope)
    : mNegSlope(negSlope)
    , mBatchDim(1)
{
    PLUGIN_VALIDATE(negSlope >= 0.0F);
}

LReLU::LReLU(void const* buffer, size_t length)
{
    char const *d = reinterpret_cast<char const*>(buffer), *a = d;
    mNegSlope = read<float>(d);
    mBatchDim = read<int32_t>(d);
    PLUGIN_VALIDATE(d == a + length);
}

int32_t LReLU::getNbOutputs() const noexcept
{
    return 1;
}

Dims LReLU::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept
{
    PLUGIN_ASSERT(nbInputDims == 1);
    PLUGIN_ASSERT(index == 0);
    return inputs[0];
}

int32_t LReLU::enqueue(
    int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    void const* inputData = inputs[0];
    void* outputData = outputs[0];
    pluginStatus_t status = lReLUInference(stream, mBatchDim * batchSize, mNegSlope, inputData, outputData);
    return status;
}

size_t LReLU::getSerializationSize() const noexcept
{
    // mNegSlope, mBatchDim
    return sizeof(float) + sizeof(int32_t);
}

void LReLU::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mNegSlope);
    write(d, mBatchDim);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void LReLU::configureWithFormat(Dims const* inputDims, int32_t /* nbInputs */, Dims const* /* outputDims */,
    int32_t nbOutputs, DataType type, PluginFormat format, int32_t) noexcept
{
    PLUGIN_ASSERT(type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
    PLUGIN_ASSERT(mBatchDim == 1);
    PLUGIN_ASSERT(nbOutputs == 1);
    for (int32_t i = 0; i < inputDims[0].nbDims; ++i)
    {
        mBatchDim *= inputDims[0].d[i];
    }
}

bool LReLU::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

int32_t LReLU::initialize() noexcept
{
    return 0;
}

void LReLU::terminate() noexcept {}

size_t LReLU::getWorkspaceSize(int32_t /* maxBatchSize */) const noexcept
{
    return 0;
}

char const* LReLU::getPluginType() const noexcept
{
    return kLRELU_PLUGIN_NAME;
}

char const* LReLU::getPluginVersion() const noexcept
{
    return kLRELU_PLUGIN_VERSION;
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

char const* LReluPluginCreator::getPluginName() const noexcept
{
    return kLRELU_PLUGIN_NAME;
}

char const* LReluPluginCreator::getPluginVersion() const noexcept
{
    return kLRELU_PLUGIN_VERSION;
}

PluginFieldCollection const* LReluPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* LReluPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        gLogWarning << "LReluPlugin is deprecated since TensorRT 9.0. Use INetworkDefinition::addActivation() to add "
                       "an IActivationLayer "
                       "with ActivationType::kLEAKY_RELU."
                    << std::endl;
        PluginField const* fields = fc->fields;
        PLUGIN_VALIDATE(fc->nbFields == 1);
        PLUGIN_VALIDATE(fields[0].type == PluginFieldType::kFLOAT32);
        PLUGIN_VALIDATE(!strcmp(fields[0].name, "negSlope"));
        float negSlope = *(static_cast<float const*>(fields[0].data));

        return new LReLU(negSlope);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* LReluPluginCreator::deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        gLogWarning << "LReluPlugin is deprecated since TensorRT 9.0. Use INetworkDefinition::addActivation() to add "
                       "an IActivationLayer "
                       "with ActivationType::kLEAKY_RELU."
                    << std::endl;
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
} // namespace nvinfer1::plugin

// LeakReLU }}}
