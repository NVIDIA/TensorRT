/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "lReluPlugin.h"
#include "checkMacrosPlugin.h"
#include "kernel.h"

using namespace nvinfer1;
using nvinfer1::PluginType;
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
}

LReLU::LReLU(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char*>(buffer), *a = d;
    mNegSlope = read<float>(d);
    mBatchDim = read<int>(d);
    ASSERT(d == a + length);
}

int LReLU::getNbOutputs() const
{
    return 1;
}

Dims LReLU::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims == 1);
    ASSERT(index == 0);
    return inputs[0];
}

int LReLU::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    pluginStatus_t status = lReLUInference(stream, mBatchDim * batchSize, mNegSlope, inputData, outputData);
    ASSERT(status == STATUS_SUCCESS);
    return status;
}

size_t LReLU::getSerializationSize() const
{
    // mNegSlope, mBatchDim
    return sizeof(float) + sizeof(int);
}

void LReLU::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mNegSlope);
    write(d, mBatchDim);
    ASSERT(d == a + getSerializationSize());
}

void LReLU::configureWithFormat(
    const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int)
{
    ASSERT(type == DataType::kFLOAT && format == PluginFormat::kNCHW);
    ASSERT(mBatchDim == 1);
    ASSERT(nbOutputs == 1);
    for (int i = 0; i < inputDims[0].nbDims; ++i)
    {
        mBatchDim *= inputDims[0].d[i];
    }
}

bool LReLU::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

int LReLU::initialize()
{
    return 0;
}

void LReLU::terminate() {}

size_t LReLU::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

const char* LReLU::getPluginType() const
{
    return LRELU_PLUGIN_NAME;
}

const char* LReLU::getPluginVersion() const
{
    return LRELU_PLUGIN_VERSION;
}

void LReLU::destroy()
{
    delete this;
}

IPluginV2* LReLU::clone() const
{
    IPluginV2* plugin = new LReLU(mNegSlope);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

LReluPluginCreator::LReluPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("negSlope", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* LReluPluginCreator::getPluginName() const
{
    return LRELU_PLUGIN_NAME;
}

const char* LReluPluginCreator::getPluginVersion() const
{
    return LRELU_PLUGIN_VERSION;
}

const PluginFieldCollection* LReluPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* LReluPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    ASSERT(fc->nbFields == 1);
    ASSERT(fields[0].type == PluginFieldType::kFLOAT32);
    negSlope = *(static_cast<const float*>(fields[0].data));

    return new LReLU(negSlope);
}

IPluginV2* LReluPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call LReluPlugin::destroy()
    return new LReLU(serialData, serialLength);
}
// LeakReLU }}}
