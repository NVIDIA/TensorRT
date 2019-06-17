/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "normalizePlugin.h"
#include <cstring>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>

using namespace nvinfer1;
using nvinfer1::plugin::Normalize;
using nvinfer1::plugin::NormalizePluginCreator;

namespace
{
const char* NORMALIZE_PLUGIN_VERSION{"1"};
const char* NORMALIZE_PLUGIN_NAME{"Normalize_TRT"};
} // namespace

PluginFieldCollection NormalizePluginCreator::mFC{};
std::vector<PluginField> NormalizePluginCreator::mPluginAttributes;

Normalize::Normalize(const Weights* weights, int nbWeights, bool acrossSpatial, bool channelShared, float eps)
    : acrossSpatial(acrossSpatial)
    , channelShared(channelShared)
    , eps(eps)
{
    mNbWeights = nbWeights;
    ASSERT(nbWeights == 1);
    ASSERT(weights[0].count >= 1);
    mWeights = copyToDevice(weights[0].values, weights[0].count);
    cublasCreate(&mCublas);
}

Normalize::Normalize(
    const Weights* weights, int nbWeights, bool acrossSpatial, bool channelShared, float eps, int C, int H, int W)
    : acrossSpatial(acrossSpatial)
    , channelShared(channelShared)
    , eps(eps)
    , C(C)
    , H(H)
    , W(W)
{
    mNbWeights = nbWeights;
    ASSERT(nbWeights == 1);
    ASSERT(weights[0].count >= 1);
    mWeights = copyToDevice(weights[0].values, weights[0].count);
    cublasCreate(&mCublas);
}

Normalize::Normalize(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char*>(buffer), *a = d;
    C = read<int>(d);
    H = read<int>(d);
    W = read<int>(d);
    acrossSpatial = read<bool>(d);
    channelShared = read<bool>(d);
    eps = read<float>(d);

    int nbWeights = read<int>(d);
    mWeights = deserializeToDevice(d, nbWeights);
    ASSERT(d == a + length);
}

int Normalize::getNbOutputs() const
{
    // Plugin layer has 1 output
    return 1;
}

Dims Normalize::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims == 1);
    ASSERT(index == 0);
    ASSERT(inputs[0].nbDims == 3);
    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

int Normalize::initialize()
{
    return 0;
}

void Normalize::terminate()
{
    CUBLASASSERT(cublasDestroy(mCublas));
}

size_t Normalize::getWorkspaceSize(int maxBatchSize) const
{
    return normalizePluginWorkspaceSize(acrossSpatial, C, H, W);
}

int Normalize::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    pluginStatus_t status = normalizeInference(stream, mCublas, acrossSpatial, channelShared, batchSize, C, H, W, eps,
        reinterpret_cast<const float*>(mWeights.values), inputData, outputData, workspace);
    ASSERT(status == STATUS_SUCCESS);
    return 0;
}

size_t Normalize::getSerializationSize() const
{
    // C,H,W, acrossSpatial,channelShared, eps, mWeights.count,mWeights.values
    return sizeof(int) * 3 + sizeof(bool) * 2 + sizeof(float) + sizeof(int) + mWeights.count * sizeof(float);
}

void Normalize::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, C);
    write(d, H);
    write(d, W);
    write(d, acrossSpatial);
    write(d, channelShared);
    write(d, eps);
    write(d, (int) mWeights.count);
    serializeFromDevice(d, mWeights);

    ASSERT(d == a + getSerializationSize());
}

bool Normalize::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

Weights Normalize::copyToDevice(const void* hostData, size_t count)
{
    void* deviceData;
    CUASSERT(cudaMalloc(&deviceData, count * sizeof(float)));
    CUASSERT(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
    return Weights{DataType::kFLOAT, deviceData, int64_t(count)};
}

void Normalize::serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const
{
    CUASSERT(cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost));
    hostBuffer += deviceWeights.count * sizeof(float);
}

Weights Normalize::deserializeToDevice(const char*& hostBuffer, size_t count)
{
    Weights w = copyToDevice(hostBuffer, count);
    hostBuffer += count * sizeof(float);
    return w;
}

// Set plugin namespace
void Normalize::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* Normalize::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
DataType Normalize::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(index == 0);
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool Normalize::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool Normalize::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Configure the layer with input and output data types.
void Normalize::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    ASSERT(*inputTypes == DataType::kFLOAT && floatFormat == PluginFormat::kNCHW);
    C = inputDims[0].d[0];
    H = inputDims[0].d[1];
    W = inputDims[0].d[2];
    if (channelShared)
    {
        ASSERT(mWeights.count == 1);
    }
    else
    {
        ASSERT(mWeights.count == C);
    }

    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);
    ASSERT(inputDims[0].nbDims >= 1); // number of dimensions of the input tensor must be >=2
    ASSERT(inputDims[0].d[0] == outputDims[0].d[0] && inputDims[0].d[1] == outputDims[0].d[1]
        && inputDims[0].d[2] == outputDims[0].d[2]);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void Normalize::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void Normalize::detachFromContext() {}

const char* Normalize::getPluginType() const
{
    return NORMALIZE_PLUGIN_NAME;
}

const char* Normalize::getPluginVersion() const
{
    return NORMALIZE_PLUGIN_VERSION;
}

void Normalize::destroy()
{
    delete this;
}

// Clone the plugin
IPluginV2Ext* Normalize::clone() const
{
    // Create a new instance
    IPluginV2Ext* plugin = new Normalize(&mWeights, mNbWeights, acrossSpatial, channelShared, eps);

    // Set the namespace
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

NormalizePluginCreator::NormalizePluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("weights", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("acrossSpatial", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("channelShared", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("nbWeights", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* NormalizePluginCreator::getPluginName() const
{
    return NORMALIZE_PLUGIN_NAME;
}

const char* NormalizePluginCreator::getPluginVersion() const
{
    return NORMALIZE_PLUGIN_VERSION;
}

const PluginFieldCollection* NormalizePluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* NormalizePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    std::vector<float> weightValues;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "nbWeights"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mNbWeights = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "acrossSpatial"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mAcrossSpatial = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "channelShared"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mChannelShared = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "eps"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            mEps = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "weights"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            weightValues.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                weightValues.push_back(*w);
                w++;
            }
        }
    }
    Weights weights{DataType::kFLOAT, weightValues.data(), (int64_t) weightValues.size()};

    Normalize* obj = new Normalize(&weights, mNbWeights, mAcrossSpatial, mChannelShared, mEps);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2Ext* NormalizePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call Normalize::destroy()
    Normalize* obj = new Normalize(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
