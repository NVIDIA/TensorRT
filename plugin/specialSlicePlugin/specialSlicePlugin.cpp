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
#include "specialSlicePlugin.h"
#include "common/kernels/maskRCNNKernels.h"
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::SpecialSlice;
using nvinfer1::plugin::SpecialSlicePluginCreator;

namespace
{
char const* const kSPECIALSLICE_PLUGIN_VERSION{"1"};
char const* const kSPECIALSLICE_PLUGIN_NAME{"SpecialSlice_TRT"};
} // namespace

SpecialSlicePluginCreator::SpecialSlicePluginCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* SpecialSlicePluginCreator::getPluginName() const noexcept
{
    return kSPECIALSLICE_PLUGIN_NAME;
}

char const* SpecialSlicePluginCreator::getPluginVersion() const noexcept
{
    return kSPECIALSLICE_PLUGIN_VERSION;
}

PluginFieldCollection const* SpecialSlicePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* SpecialSlicePluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        gLogWarning
            << "SpecialSlicePlugin is deprecated since TensorRT 9.0. Use INetworkDefinition::addSlice() to add an "
               "ISliceLayer."
            << std::endl;
        return new SpecialSlice();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* SpecialSlicePluginCreator::deserializePlugin(char const* name, void const* data, size_t length) noexcept
{
    try
    {
        gLogWarning
            << "SpecialSlicePlugin is deprecated since TensorRT 9.0. Use INetworkDefinition::addSlice() to add an "
               "ISliceLayer."
            << std::endl;
        return new SpecialSlice(data, length);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

size_t SpecialSlice::getWorkspaceSize(int32_t) const noexcept
{
    return 0;
}

bool SpecialSlice::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

char const* SpecialSlice::getPluginType() const noexcept
{
    return "SpecialSlice_TRT";
}

char const* SpecialSlice::getPluginVersion() const noexcept
{
    return "1";
}

IPluginV2Ext* SpecialSlice::clone() const noexcept
{
    try
    {
        auto plugin = new SpecialSlice(*this);
        plugin->setPluginNamespace(mNameSpace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void SpecialSlice::setPluginNamespace(char const* libNamespace) noexcept
{
    mNameSpace = libNamespace;
}

char const* SpecialSlice::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

size_t SpecialSlice::getSerializationSize() const noexcept
{
    return sizeof(int32_t);
}

void SpecialSlice::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mBboxesCnt);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

SpecialSlice::SpecialSlice(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    mBboxesCnt = read<int32_t>(d);
    PLUGIN_VALIDATE(d == a + length);
}

SpecialSlice::SpecialSlice() {}

int32_t SpecialSlice::initialize() noexcept
{
    return 0;
}

int32_t SpecialSlice::getNbOutputs() const noexcept
{
    return 1;
}

void SpecialSlice::check_valid_inputs(nvinfer1::Dims const* inputs, int32_t nbInputDims)
{

    PLUGIN_ASSERT(nbInputDims == 1);
    // detections: [N, anchors, (y1, x1, y2, x2, class_id, score)]
    PLUGIN_ASSERT(inputs[0].nbDims == 2 && inputs[0].d[1] == 6);
}

Dims SpecialSlice::getOutputDimensions(int32_t index, Dims const* inputDims, int32_t nbInputs) noexcept
{

    PLUGIN_ASSERT(index == 0);
    PLUGIN_ASSERT(nbInputs == 1);
    check_valid_inputs(inputDims, nbInputs);

    nvinfer1::Dims output{};
    output.nbDims = inputDims[0].nbDims;
    // number of anchors
    output.d[0] = inputDims[0].d[0];
    //(y1, x1, y2, x2)
    output.d[1] = 4;

    return output;
}

int32_t SpecialSlice::enqueue(
    int32_t batch_size, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{

    specialSlice(stream, batch_size, mBboxesCnt, inputs[0], outputs[0]);

    return cudaGetLastError() != cudaSuccess;
}

// Return the DataType of the plugin output at the requested index
DataType SpecialSlice::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    // Only 1 input and 1 output from the plugin layer
    PLUGIN_ASSERT(index == 0);

    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool SpecialSlice::isOutputBroadcastAcrossBatch(
    int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool SpecialSlice::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void SpecialSlice::configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
    DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
    bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept
{
    PLUGIN_ASSERT(nbInputs == 1);

    PLUGIN_ASSERT(nbOutputs == 1);

    mBboxesCnt = inputDims[0].d[0];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void SpecialSlice::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void SpecialSlice::detachFromContext() noexcept {}
