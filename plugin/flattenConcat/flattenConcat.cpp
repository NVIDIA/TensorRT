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
#include "flattenConcat.h"
#include "common/dimsHelpers.h"

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
    PLUGIN_VALIDATE(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
}

FlattenConcat::FlattenConcat(int concatAxis, bool ignoreBatch, int numInputs, int outputConcatAxis,
    const int* inputConcatAxis, const size_t* copySize, nvinfer1::Dims const& chwDims)
    : mCopySize(numInputs)
    , mInputConcatAxis(numInputs)
    , mIgnoreBatch(ignoreBatch)
    , mConcatAxisID(concatAxis)
    , mOutputConcatAxis(outputConcatAxis)
    , mNumInputs(numInputs)
    , mCHW(chwDims)
{
    PLUGIN_VALIDATE(mConcatAxisID >= 1 && mConcatAxisID <= 3);

    std::copy(copySize, copySize + mNumInputs, mCopySize.begin());
    std::copy(inputConcatAxis, inputConcatAxis + mNumInputs, mInputConcatAxis.begin());
}

FlattenConcat::FlattenConcat(const void* data, size_t length)
{
    const char* d = static_cast<const char*>(data);
    const char* const a = d;
    mIgnoreBatch = read<bool>(d);
    mConcatAxisID = read<int>(d);
    PLUGIN_VALIDATE(mConcatAxisID >= 1 && mConcatAxisID <= 3);
    mOutputConcatAxis = read<int>(d);
    mNumInputs = read<int>(d);

    mInputConcatAxis.resize(mNumInputs);
    std::for_each(mInputConcatAxis.begin(), mInputConcatAxis.end(), [&](int& inp) { inp = read<int>(d); });

    mCHW = read<nvinfer1::Dims3>(d);

    mCopySize.resize(mNumInputs);
    std::for_each(mCopySize.begin(), mCopySize.end(), [&](size_t& inp) { inp = read<size_t>(d); });

    PLUGIN_VALIDATE(d == a + length);
}

FlattenConcat::~FlattenConcat() {}

int FlattenConcat::getNbOutputs() const noexcept
{
    return 1;
}

Dims FlattenConcat::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    try
    {
        PLUGIN_ASSERT(nbInputDims >= 1);
        PLUGIN_ASSERT(index == 0);

        mNumInputs = nbInputDims;
        mCopySize.resize(mNumInputs);
        mInputConcatAxis.resize(mNumInputs);
        int outputConcatAxis = 0;

        for (int i = 0; i < nbInputDims; ++i)
        {
            int flattenInput = 0;
            PLUGIN_ASSERT(inputs[i].nbDims == 3);
            if (mConcatAxisID != 1)
            {
                PLUGIN_ASSERT(inputs[i].d[0] == inputs[0].d[0]);
            }
            if (mConcatAxisID != 2)
            {
                PLUGIN_ASSERT(inputs[i].d[1] == inputs[0].d[1]);
            }
            if (mConcatAxisID != 3)
            {
                PLUGIN_ASSERT(inputs[i].d[2] == inputs[0].d[2]);
            }
            flattenInput = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
            outputConcatAxis += flattenInput;
        }

        return Dims3(mConcatAxisID == 1 ? outputConcatAxis : 1, mConcatAxisID == 2 ? outputConcatAxis : 1,
            mConcatAxisID == 3 ? outputConcatAxis : 1);
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return Dims{};
}

int FlattenConcat::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void FlattenConcat::terminate() noexcept {}

size_t FlattenConcat::getWorkspaceSize(int) const noexcept
{
    return 0;
}

int FlattenConcat::enqueue(
    int batchSize, const void* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_ASSERT(mConcatAxisID != 0);
        // mCHW is the first input tensor
        auto numConcats = static_cast<int32_t>(pluginInternal::volume(mCHW, /*start*/ 0, /*stop*/ mConcatAxisID - 1));

        // Num concats will be proportional to number of samples in a batch
        if (!mIgnoreBatch)
        {
            numConcats *= batchSize;
        }

        auto* output = static_cast<float*>(outputs[0]);
        int offset = 0;
        for (int i = 0; i < mNumInputs; ++i)
        {
            const auto* input = static_cast<const float*>(inputs[i]);
            for (int n = 0; n < numConcats; ++n)
            {
                auto status = cublasScopy(mCublas, mInputConcatAxis[i], input + n * mInputConcatAxis[i], 1,
                    output + (n * mOutputConcatAxis + offset), 1);

                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    return STATUS_FAILURE;
                }
            }
            offset += mInputConcatAxis[i];
        }

        return STATUS_SUCCESS;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return -1;
}

size_t FlattenConcat::getSerializationSize() const noexcept
{
    return sizeof(bool) + sizeof(int) * (3 + mNumInputs) + sizeof(nvinfer1::Dims) + (sizeof(decltype(mCopySize)::value_type) * mNumInputs);
}

void FlattenConcat::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    const char* const a = d;
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
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void FlattenConcat::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
    mCublas = cublasContext;
}

// Detach the plugin object from its execution context.
void FlattenConcat::detachFromContext() noexcept {}

// Return true if output tensor is broadcast across a batch.
bool FlattenConcat::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool FlattenConcat::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

// Set plugin namespace
void FlattenConcat::setPluginNamespace(const char* pluginNamespace) noexcept
{
    try
    {
        mPluginNamespace = pluginNamespace;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

const char* FlattenConcat::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index
DataType FlattenConcat::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(index < 3);
    return DataType::kFLOAT;
}

void FlattenConcat::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    try
    {
        PLUGIN_ASSERT(nbOutputs == 1);
        mCHW = inputDims[0];
        mNumInputs = nbInputs;
        PLUGIN_ASSERT(inputDims[0].nbDims == 3);

        mInputConcatAxis.resize(mNumInputs);
        for (int i = 0; i < nbInputs; ++i)
        {
            int flattenInput = 0;
            PLUGIN_ASSERT(inputDims[i].nbDims == 3);
            if (mConcatAxisID != 1)
            {
                PLUGIN_ASSERT(inputDims[i].d[0] == inputDims[0].d[0]);
            }
            if (mConcatAxisID != 2)
            {
                PLUGIN_ASSERT(inputDims[i].d[1] == inputDims[0].d[1]);
            }
            if (mConcatAxisID != 3)
            {
                PLUGIN_ASSERT(inputDims[i].d[2] == inputDims[0].d[2]);
            }
            flattenInput = inputDims[i].d[0] * inputDims[i].d[1] * inputDims[i].d[2];
            mInputConcatAxis[i] = flattenInput;
            mOutputConcatAxis += mInputConcatAxis[i];
        }

        mCopySize.resize(mNumInputs);
        for (int i = 0; i < nbInputs; ++i)
        {
            mCopySize[i] = inputDims[i].d[0] * inputDims[i].d[1] * inputDims[i].d[2] * sizeof(float);
        }
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

bool FlattenConcat::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}
const char* FlattenConcat::getPluginType() const noexcept
{
    return "FlattenConcat_TRT";
}

const char* FlattenConcat::getPluginVersion() const noexcept
{
    return "1";
}

void FlattenConcat::destroy() noexcept
{
    delete this;
}

IPluginV2Ext* FlattenConcat::clone() const noexcept
{
    try
    {
        auto* plugin = new FlattenConcat(mConcatAxisID, mIgnoreBatch, mNumInputs, mOutputConcatAxis,
            mInputConcatAxis.data(), mCopySize.data(), mCHW);
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

FlattenConcatPluginCreator::FlattenConcatPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("ignoreBatch", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* FlattenConcatPluginCreator::getPluginName() const noexcept
{
    return FLATTENCONCAT_PLUGIN_NAME;
}

const char* FlattenConcatPluginCreator::getPluginVersion() const noexcept
{
    return FLATTENCONCAT_PLUGIN_VERSION;
}

const PluginFieldCollection* FlattenConcatPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* FlattenConcatPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        plugin::validateRequiredAttributesExist({"axis", "ignoreBatch"}, fc);
        const PluginField* fields = fc->fields;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "axis"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                mConcatAxisID = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "ignoreBatch"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                auto ignoreBatch = *(static_cast<const int32_t*>(fields[i].data));
                PLUGIN_VALIDATE(ignoreBatch == 0 || ignoreBatch == 1);
                mIgnoreBatch = static_cast<bool>(ignoreBatch);
            }
        }

        auto* plugin = new FlattenConcat(mConcatAxisID, mIgnoreBatch);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* FlattenConcatPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call Concat::destroy()
        IPluginV2Ext* plugin = new FlattenConcat(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
