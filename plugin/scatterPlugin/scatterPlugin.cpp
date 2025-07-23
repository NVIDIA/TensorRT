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
#include "scatterPlugin.h"
#include "common/half.h"
#include <cstring>
#include <iostream>
#include <sstream>

namespace nvinfer1::plugin
{

namespace
{

char const* const kSCATTERND_PLUGIN_VERSION{"1"};
char const* const kSCATTERND_PLUGIN_NAME{"ScatterND"};
} // namespace

ScatterND::ScatterND() {}

int32_t ScatterND::getNbOutputs() const noexcept
{
    // Plugin layer has 1 output
    return 1;
}

DimsExprs ScatterND::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    // output should have same dimensions as data tensor
    DimsExprs ret = inputs[dataTensorIdx];
    return ret;
}

int32_t ScatterND::initialize() noexcept
{
    return 0;
}

void ScatterND::terminate() noexcept {}

bool ScatterND::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(pos < 4);
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(nbOutputs == 1);
    PluginTensorDesc const& desc = inOut[pos];
    bool ret = false;
    switch (pos)
    {
    case dataTensorIdx:
    case updateTensorIdx:
        ret = ((desc.type == DataType::kFLOAT || desc.type == DataType::kINT32)
            && desc.format == TensorFormat::kLINEAR);
        break;
    case indexTensorIdx: ret = (desc.type == DataType::kINT32 && desc.format == TensorFormat::kLINEAR); break;
    case 3:
        ret = ((desc.type == DataType::kFLOAT || desc.type == DataType::kINT32)
            && desc.format == TensorFormat::kLINEAR);
        break;
    }
    return ret;
}

void ScatterND::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
}

int32_t ScatterND::calculateNumSlices(Dims indexTensorDims) const noexcept
{
    int32_t nSlices = 1;
    for (int32_t i = 0; i < indexTensorDims.nbDims - 1; i++)
    {
        nSlices *= indexTensorDims.d[i];
    }
    return nSlices;
}

size_t ScatterND::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    int32_t nSlices = calculateNumSlices(inputs[indexTensorIdx].dims);
    // transformCoeffs + transformed indices
    return outputs[0].dims.MAX_DIMS * sizeof(int32_t) + nSlices * sizeof(int32_t);
}

void ScatterND::calculateTransformCoeff(
    Dims const& dataTensorDims, int32_t indexRank, int32_t* transformCoeff) const noexcept
{
    std::vector<int32_t> pitches;
    for (int32_t i = indexRank - 1, nIndx = 1; i >= 0; i--)
    {
        pitches.push_back(nIndx);
        nIndx *= dataTensorDims.d[i];
    }

    std::reverse(pitches.begin(), pitches.end()); // last dimension pitch is always one (assuming linear mem)

    std::copy(pitches.begin(), pitches.end(), transformCoeff);
}

int32_t ScatterND::calculateCopySize(Dims const& dataDims) const noexcept
{
    int32_t copySize = 1;
    for (int32_t i = 0; i < dataDims.nbDims; i++)
    {
        copySize *= dataDims.d[i];
    }
    copySize *= sizeof(float);
    return copySize;
}

int32_t ScatterND::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    PLUGIN_VALIDATE(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr
        && workspace != nullptr);

    int32_t transformCoeff[nvinfer1::Dims::MAX_DIMS];
    std::memset(transformCoeff, 0, sizeof(int32_t) * outputDesc[0].dims.MAX_DIMS);
    Dims IndexDims = inputDesc[indexTensorIdx].dims;

    Dims dataDims = inputDesc[dataTensorIdx].dims;

    int32_t indexRank = IndexDims.d[IndexDims.nbDims - 1];
    PLUGIN_ASSERT(indexRank <= dataDims.nbDims);

    int32_t nSlices = calculateNumSlices(IndexDims);
    int32_t rowSize = 1;
    int32_t copySize = calculateCopySize(dataDims);
    int32_t elementSizeInBytes = 1;
    switch (inputDesc->type)
    {
    case DataType::kFLOAT:
    case DataType::kINT32: elementSizeInBytes = 4; break;
    case DataType::kHALF: elementSizeInBytes = 2; break;
    case DataType::kINT8:
    case DataType::kUINT8:
    case DataType::kBOOL: elementSizeInBytes = 1; break;
    case DataType::kFP8:
    case DataType::kBF16:
    case DataType::kINT64:
    case DataType::kINT4:
    case DataType::kFP4:
    case DataType::kE8M0: PLUGIN_FAIL("Unsupported data type");
    }

    for (int32_t i = indexRank; i < dataDims.nbDims; i++)
    {
        rowSize *= dataDims.d[i];
    }

    calculateTransformCoeff(dataDims, indexRank, transformCoeff);

    scatterNDInference(stream, transformCoeff, dataDims.nbDims, indexRank, nSlices, rowSize, copySize,
        elementSizeInBytes, inputs[indexTensorIdx], inputs[updateTensorIdx], inputs[dataTensorIdx], outputs[0],
        workspace);

    return 0;
}

size_t ScatterND::getSerializationSize() const noexcept
{
    return 0;
}

void ScatterND::serialize(void* buffer) const noexcept
{
    return;
}

// Set plugin namespace
void ScatterND::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

char const* ScatterND::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index
DataType ScatterND::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    return inputTypes[dataTensorIdx];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void ScatterND::attachToContext(cudnnContext* cudnn, cublasContext* cublas, IGpuAllocator* gpuAllocator) noexcept
{
    return;
}

// Detach the plugin object from its execution context.
void ScatterND::detachFromContext() noexcept {}

char const* ScatterND::getPluginType() const noexcept
{
    return kSCATTERND_PLUGIN_NAME;
}

char const* ScatterND::getPluginVersion() const noexcept
{
    return kSCATTERND_PLUGIN_VERSION;
}

void ScatterND::destroy() noexcept
{
    delete this;
}

// Clone the plugin
IPluginV2DynamicExt* ScatterND::clone() const noexcept
{
    try
    {
        // Create a new instance
        ScatterND* plugin = new ScatterND();
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

ScatterNDPluginCreator::ScatterNDPluginCreator()
{
    mFC.nbFields = 0;
}

char const* ScatterNDPluginCreator::getPluginName() const noexcept
{
    return kSCATTERND_PLUGIN_NAME;
}

char const* ScatterNDPluginCreator::getPluginVersion() const noexcept
{
    return kSCATTERND_PLUGIN_VERSION;
}

PluginFieldCollection const* ScatterNDPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* ScatterNDPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        ScatterND* obj = new ScatterND();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* ScatterNDPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call Normalize::destroy()
        ScatterND* obj = new ScatterND();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
} // namespace nvinfer1::plugin
