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
#include "normalizePlugin.h"
#include "common/half.h"
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

Normalize::Normalize(Weights const* weights, int nbWeights, bool acrossSpatial, bool channelShared, float eps)
    : acrossSpatial(acrossSpatial)
    , channelShared(channelShared)
    , eps(eps)
{
    mNbWeights = nbWeights;
    PLUGIN_VALIDATE(nbWeights == 1);
    PLUGIN_VALIDATE(weights[0].count >= 1);
    mWeights = copyToDevice(weights[0].values, weights[0].count);
    mScalarScale = static_cast<float const*>(weights[0].values)[0];
}

Normalize::Normalize(
    Weights const* weights, int nbWeights, float scalarScale, bool acrossSpatial, bool channelShared, float eps, int C, int H, int W)
    : mScalarScale(scalarScale)
    , acrossSpatial(acrossSpatial)
    , channelShared(channelShared)
    , eps(eps)
    , C(C)
    , H(H)
    , W(W)
{
    mNbWeights = nbWeights;
    PLUGIN_VALIDATE(nbWeights == 1);
    PLUGIN_VALIDATE(weights[0].count >= 1);
    mWeights = copyToDevice(weights[0].values, weights[0].count);
}

Normalize::Normalize(const void* buffer, size_t length)
{
    const char *d = static_cast<const char*>(buffer);
    const char *a = d;
    C = read<int>(d);
    H = read<int>(d);
    W = read<int>(d);
    acrossSpatial = read<bool>(d);
    channelShared = read<bool>(d);
    eps = read<float>(d);

    mNbWeights = read<int>(d);
    int count = read<int>(d);
    std::memcpy(&mScalarScale, d, sizeof(float));
    mWeights = deserializeToDevice(d, count);
    PLUGIN_VALIDATE(d == a + length);
}

int Normalize::getNbOutputs() const noexcept
{
    // Plugin layer has 1 output
    return 1;
}

Dims Normalize::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    PLUGIN_ASSERT(nbInputDims == 1);
    PLUGIN_ASSERT(index == 0);
    PLUGIN_ASSERT(inputs[0].nbDims == 3);
    return Dims3(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

int Normalize::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void Normalize::terminate() noexcept
{
}

size_t Normalize::getWorkspaceSize(int maxBatchSize) const noexcept
{
    return normalizePluginWorkspaceSize(acrossSpatial, C, H, W);
}

int Normalize::enqueue(
    int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];

    pluginStatus_t status;

    if(acrossSpatial && channelShared) // Since cublasPointerMode_t is CUBLAS_POINTER_MODE_HOST, scale should be on the host
    {
        status = normalizeInference(stream, mCublas, acrossSpatial, channelShared, batchSize, C, H, W, eps,
        &mScalarScale, inputData, outputData, workspace);
    }
    else // No risk of device pointers being passed to cublas as alpha or beta
    {
        status = normalizeInference(stream, mCublas, acrossSpatial, channelShared, batchSize, C, H, W, eps,
            static_cast<float const*>(mWeights.values), inputData, outputData, workspace);
    }

    return status;
}

size_t Normalize::getSerializationSize() const noexcept
{
    // C,H,W, acrossSpatial,channelShared, eps, mWeights.count,mWeights.values
    return sizeof(int) * 3 + sizeof(bool) * 2 + sizeof(float) + sizeof(int) * 2 + mWeights.count * sizeof(float);
}

void Normalize::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, C);
    write(d, H);
    write(d, W);
    write(d, acrossSpatial);
    write(d, channelShared);
    write(d, eps);
    write(d, (int) mNbWeights);
    write(d, (int) mWeights.count);
    serializeFromDevice(d, mWeights);

    PLUGIN_ASSERT(d == a + getSerializationSize());
}

bool Normalize::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

Weights Normalize::copyToDevice(const void* hostData, size_t count)
{
    void* deviceData;
    PLUGIN_CUASSERT(cudaMalloc(&deviceData, count * sizeof(float)));
    PLUGIN_CUASSERT(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
    return Weights{DataType::kFLOAT, deviceData, int64_t(count)};
}

void Normalize::serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const
{
    PLUGIN_CUASSERT(
        cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost));
    hostBuffer += deviceWeights.count * sizeof(float);
}

Weights Normalize::deserializeToDevice(const char*& hostBuffer, size_t count)
{
    Weights w = copyToDevice(hostBuffer, count);
    hostBuffer += count * sizeof(float);
    return w;
}

// Set plugin namespace
void Normalize::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const char* Normalize::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index
DataType Normalize::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool Normalize::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool Normalize::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void Normalize::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    PLUGIN_ASSERT(*inputTypes == DataType::kFLOAT && floatFormat == PluginFormat::kLINEAR);
    C = inputDims[0].d[0];
    H = inputDims[0].d[1];
    W = inputDims[0].d[2];
    if (channelShared)
    {
        PLUGIN_ASSERT(mWeights.count == 1);
    }
    else
    {
        PLUGIN_ASSERT(mWeights.count == C);
    }

    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(inputDims[0].nbDims >= 1); // number of dimensions of the input tensor must be >=2
    PLUGIN_ASSERT(inputDims[0].d[0] == outputDims[0].d[0] && inputDims[0].d[1] == outputDims[0].d[1]
        && inputDims[0].d[2] == outputDims[0].d[2]);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void Normalize::attachToContext(cudnnContext* cudnn, cublasContext* cublas, IGpuAllocator* gpuAllocator) noexcept
{
    mCublas = cublas;
}

// Detach the plugin object from its execution context.
void Normalize::detachFromContext() noexcept
{
}

const char* Normalize::getPluginType() const noexcept
{
    return NORMALIZE_PLUGIN_NAME;
}

const char* Normalize::getPluginVersion() const noexcept
{
    return NORMALIZE_PLUGIN_VERSION;
}

void Normalize::destroy() noexcept
{
    PLUGIN_CUASSERT(cudaFree(const_cast<void*>(mWeights.values)));
    delete this;
}

// Clone the plugin
IPluginV2Ext* Normalize::clone() const noexcept
{
    try
    {
        // Create a new instance
        IPluginV2Ext* plugin = new Normalize(&mWeights, mNbWeights, mScalarScale, acrossSpatial, channelShared, eps, C, H, W);

        // Set the namespace
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

NormalizePluginCreator::NormalizePluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("weights", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("acrossSpatial", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("channelShared", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("nbWeights", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* NormalizePluginCreator::getPluginName() const noexcept
{
    return NORMALIZE_PLUGIN_NAME;
}

const char* NormalizePluginCreator::getPluginVersion() const noexcept
{
    return NORMALIZE_PLUGIN_VERSION;
}

const PluginFieldCollection* NormalizePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* NormalizePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        std::vector<float> weightValues;
        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "nbWeights"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                mNbWeights = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attrName, "acrossSpatial"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                mAcrossSpatial = *(static_cast<const bool*>(fields[i].data));
            }
            else if (!strcmp(attrName, "channelShared"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                mChannelShared = *(static_cast<const bool*>(fields[i].data));
            }
            else if (!strcmp(attrName, "eps"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                mEps = *(static_cast<const float*>(fields[i].data));
            }
            else if (!strcmp(attrName, "weights"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
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
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* NormalizePluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call Normalize::destroy()
        Normalize* obj = new Normalize(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
