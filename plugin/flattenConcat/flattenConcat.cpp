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
#include "flattenConcat.h"
#include <algorithm>
#include <cstring>
#include <cudnn.h>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::FlattenConcat;
using nvinfer1::plugin::FlattenConcatPluginCreator;

static const char* FLATTENCONCAT_PLUGIN_VERSION{"1"};
static const char* FLATTENCONCAT_PLUGIN_NAME{"FlattenConcat_TRT"};

PluginFieldCollection FlattenConcatPluginCreator::mFC{};
std::vector<PluginField> FlattenConcatPluginCreator::mPluginAttributes;

FlattenConcat::FlattenConcat(int concatAxis, bool ignoreBatch)
    : mIgnoreBatch(ignoreBatch)
    , mConcatAxisID(concatAxis)
{
    ASSERT(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
}

FlattenConcat::FlattenConcat(
    int concatAxis, bool ignoreBatch, int numInputs, int outputConcatAxis, const int* inputConcatAxis, size_t* copySize)
    : mIgnoreBatch(ignoreBatch)
    , mConcatAxisID(concatAxis)
    , mOutputConcatAxis(outputConcatAxis)
    , mNumInputs(numInputs)
{
    ASSERT(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);

    // Allocate memory for mInputConcatAxis, mCopySize members
    LOG_ERROR(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
    LOG_ERROR(cudaMallocHost((void**) &mCopySize, mNumInputs * sizeof(size_t)));

    // Perform deep copy
    if (copySize != nullptr)
    {
        for (int i = 0; i < mNumInputs; i++)
        {
            mCopySize[i] = static_cast<size_t>(copySize[i]);
        }
    }

    for (int i = 0; i < mNumInputs; ++i)
    {
        mInputConcatAxis[i] = inputConcatAxis[i];
    }

    // Create cublas context
    LOG_ERROR(cublasCreate(&mCublas));
}

FlattenConcat::FlattenConcat(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mIgnoreBatch = read<bool>(d);
    mConcatAxisID = read<int>(d);
    ASSERT(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
    mOutputConcatAxis = read<int>(d);
    mNumInputs = read<int>(d);
    LOG_ERROR(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
    LOG_ERROR(cudaMallocHost((void**) &mCopySize, mNumInputs * sizeof(int)));

    std::for_each(mInputConcatAxis, mInputConcatAxis + mNumInputs, [&](int& inp) { inp = read<int>(d); });

    mCHW = read<nvinfer1::DimsCHW>(d);

    std::for_each(mCopySize, mCopySize + mNumInputs, [&](size_t& inp) { inp = read<size_t>(d); });

    ASSERT(d == a + length);
}

FlattenConcat::~FlattenConcat()
{
    if (mInputConcatAxis)
    {
        LOG_ERROR(cudaFreeHost(mInputConcatAxis));
    }
    if (mCopySize)
    {
        LOG_ERROR(cudaFreeHost(mCopySize));
    }
}

int FlattenConcat::getNbOutputs() const
{
    return 1;
}

Dims FlattenConcat::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims >= 1);
    ASSERT(index == 0);

    mNumInputs = nbInputDims;
    LOG_ERROR(cudaMallocHost((void**) &mInputConcatAxis, nbInputDims * sizeof(int)));
    int outputConcatAxis = 0;

    for (int i = 0; i < nbInputDims; ++i)
    {
        int flattenInput = 0;
        ASSERT(inputs[i].nbDims == 3);
        if (mConcatAxisID != 1)
        {
            ASSERT(inputs[i].d[0] == inputs[0].d[0]);
        }
        if (mConcatAxisID != 2)
        {
            ASSERT(inputs[i].d[1] == inputs[0].d[1]);
        }
        if (mConcatAxisID != 3)
        {
            ASSERT(inputs[i].d[2] == inputs[0].d[2]);
        }
        flattenInput = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
        outputConcatAxis += flattenInput;
    }

    return DimsCHW(mConcatAxisID == 1 ? outputConcatAxis : 1, mConcatAxisID == 2 ? outputConcatAxis : 1,
        mConcatAxisID == 3 ? outputConcatAxis : 1);
}

int FlattenConcat::initialize()
{
    return STATUS_SUCCESS;
}

void FlattenConcat::terminate()
{
    LOG_ERROR(cublasDestroy(mCublas));
}

size_t FlattenConcat::getWorkspaceSize(int) const
{
    return 0;
}

int FlattenConcat::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    int numConcats = 1;
    ASSERT(mConcatAxisID != 0);
    // mCHW is the first input tensor
    numConcats = std::accumulate(mCHW.d, mCHW.d + mConcatAxisID - 1, 1, std::multiplies<int>());

    LOG_ERROR(cublasSetStream(mCublas, stream));

    // Num concats will be proportional to number of samples in a batch
    if (!mIgnoreBatch)
    {
        numConcats *= batchSize;
    }

    auto* output = reinterpret_cast<float*>(outputs[0]);
    int offset = 0;
    for (int i = 0; i < mNumInputs; ++i)
    {
        const auto* input = reinterpret_cast<const float*>(inputs[i]);
        float* inputTemp;
        LOG_ERROR(cudaMalloc(&inputTemp, mCopySize[i] * batchSize));
        LOG_ERROR(cudaMemcpyAsync(inputTemp, input, mCopySize[i] * batchSize, cudaMemcpyDeviceToDevice, stream));

        for (int n = 0; n < numConcats; ++n)
        {
            LOG_ERROR(cublasScopy(mCublas, mInputConcatAxis[i], inputTemp + n * mInputConcatAxis[i], 1,
                output + (n * mOutputConcatAxis + offset), 1));
        }
        LOG_ERROR(cudaFree(inputTemp));
        offset += mInputConcatAxis[i];
    }

    return 0;
}

size_t FlattenConcat::getSerializationSize() const
{
    return sizeof(bool) + sizeof(int) * (3 + mNumInputs) + sizeof(nvinfer1::Dims) + (sizeof(mCopySize) * mNumInputs);
}

void FlattenConcat::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mIgnoreBatch);
    write(d, mConcatAxisID);
    write(d, mOutputConcatAxis);
    write(d, mNumInputs);
    for (int i = 0; i < mNumInputs; ++i)
    {
        write(d, mInputConcatAxis[i]);
    }
    write(d, mCHW);
    for (int i = 0; i < mNumInputs; ++i)
    {
        write(d, mCopySize[i]);
    }
    ASSERT(d == a + getSerializationSize());
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void FlattenConcat::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void FlattenConcat::detachFromContext() {}

// Return true if output tensor is broadcast across a batch.
bool FlattenConcat::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool FlattenConcat::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Set plugin namespace
void FlattenConcat::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* FlattenConcat::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
DataType FlattenConcat::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(index < 3);
    return DataType::kFLOAT;
}

void FlattenConcat::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    ASSERT(nbOutputs == 1);
    mCHW = inputDims[0];
    mNumInputs = nbInputs;
    ASSERT(inputDims[0].nbDims == 3);

    if (mInputConcatAxis == nullptr)
    {
        LOG_ERROR(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
    }

    for (int i = 0; i < nbInputs; ++i)
    {
        int flattenInput = 0;
        ASSERT(inputDims[i].nbDims == 3);
        if (mConcatAxisID != 1)
        {
            ASSERT(inputDims[i].d[0] == inputDims[0].d[0]);
        }
        if (mConcatAxisID != 2)
        {
            ASSERT(inputDims[i].d[1] == inputDims[0].d[1]);
        }
        if (mConcatAxisID != 3)
        {
            ASSERT(inputDims[i].d[2] == inputDims[0].d[2]);
        }
        flattenInput = inputDims[i].d[0] * inputDims[i].d[1] * inputDims[i].d[2];
        mInputConcatAxis[i] = flattenInput;
        mOutputConcatAxis += mInputConcatAxis[i];
    }

    for (int i = 0; i < nbInputs; ++i)
    {
        mCopySize[i] = inputDims[i].d[0] * inputDims[i].d[1] * inputDims[i].d[2] * sizeof(float);
    }
}

bool FlattenConcat::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}
const char* FlattenConcat::getPluginType() const
{
    return "FlattenConcat_TRT";
}

const char* FlattenConcat::getPluginVersion() const
{
    return "1";
}

void FlattenConcat::destroy()
{
    delete this;
}

IPluginV2Ext* FlattenConcat::clone() const
{
    auto* plugin
        = new FlattenConcat(mConcatAxisID, mIgnoreBatch, mNumInputs, mOutputConcatAxis, mInputConcatAxis, mCopySize);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

FlattenConcatPluginCreator::FlattenConcatPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("ignoreBatch", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* FlattenConcatPluginCreator::getPluginName() const
{
    return FLATTENCONCAT_PLUGIN_NAME;
}

const char* FlattenConcatPluginCreator::getPluginVersion() const
{
    return FLATTENCONCAT_PLUGIN_VERSION;
}

const PluginFieldCollection* FlattenConcatPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* FlattenConcatPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "axis"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mConcatAxisID = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "ignoreBatch"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mIgnoreBatch = *(static_cast<const bool*>(fields[i].data));
        }
    }

    auto* plugin = new FlattenConcat(mConcatAxisID, mIgnoreBatch);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* FlattenConcatPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call Concat::destroy()
    IPluginV2Ext* plugin = new FlattenConcat(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
