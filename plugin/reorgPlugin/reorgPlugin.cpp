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
#include "reorgPlugin.h"

namespace nvinfer1::plugin
{
static char const* const kREORG_PLUGIN_STATIC_VERSION{"1"};
static char const* const kREORG_PLUGIN_DYNAMIC_VERSION{"2"};
static char const* const kREORG_PLUGIN_NAME{"Reorg_TRT"};

template <class TBaseClass>
Reorg<TBaseClass>::Reorg(int32_t strideValue)
    : stride(strideValue)
{
}

template <class TBaseClass>
int32_t Reorg<TBaseClass>::getNbOutputs() const noexcept
{
    return 1;
}

template <class TBaseClass>
int32_t Reorg<TBaseClass>::initialize() noexcept
{
    return STATUS_SUCCESS;
}

template <class TBaseClass>
void Reorg<TBaseClass>::terminate() noexcept
{
}

template <class TBaseClass>
char const* Reorg<TBaseClass>::getPluginType() const noexcept
{
    return kREORG_PLUGIN_NAME;
}

template <class TBaseClass>
void Reorg<TBaseClass>::destroy() noexcept
{
    delete this;
}

// Set plugin namespace
template <class TBaseClass>
void Reorg<TBaseClass>::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

template <class TBaseClass>
char const* Reorg<TBaseClass>::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index
template <class TBaseClass>
DataType Reorg<TBaseClass>::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    // Only 1 input and 1 output from the plugin layer
    PLUGIN_ASSERT(index == 0);

    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
template <class TBaseClass>
void Reorg<TBaseClass>::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
template <class TBaseClass>
void Reorg<TBaseClass>::detachFromContext() noexcept
{
}

ReorgDynamic::ReorgDynamic(int32_t stride)
    : Reorg<IPluginV2DynamicExt>(stride)
{
}

ReorgDynamic::ReorgDynamic(void const* buffer, size_t length)
{
    char const* d = reinterpret_cast<char const*>(buffer);
    char const* a = d;
    stride = read<int32_t>(d);
    PLUGIN_VALIDATE(d == a + length);
}

char const* ReorgDynamic::getPluginVersion() const noexcept
{
    return kREORG_PLUGIN_DYNAMIC_VERSION;
}

size_t ReorgDynamic::getSerializationSize() const noexcept
{
    // stride
    return sizeof(int32_t);
}

size_t ReorgDynamic::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

void ReorgDynamic::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, stride);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

DimsExprs ReorgDynamic::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(outputIndex == 0);
    DimsExprs output{3, {}};
    auto const* strideExpr = exprBuilder.constant(stride);
    auto const* strideSquareExpr = exprBuilder.constant(stride * stride);
    output.d[0] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[0], *strideSquareExpr);
    output.d[1] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[0].d[1], *strideExpr);
    output.d[2] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[0].d[2], *strideExpr);
    return output;
}

bool ReorgDynamic::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(pos >= 0 && pos <= 1);
    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(nbOutputs == 1);
    return (inOut[pos].type == DataType::kFLOAT && inOut[pos].format == PluginFormat::kLINEAR);
}

void ReorgDynamic::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(in->desc.type == DataType::kFLOAT);
    PLUGIN_ASSERT(out->desc.type == DataType::kFLOAT);
    PLUGIN_ASSERT(in->desc.format == PluginFormat::kLINEAR);
    PLUGIN_ASSERT(out->desc.format == PluginFormat::kLINEAR);
    PLUGIN_ASSERT(stride > 0);

    int32_t H = in->desc.dims.d[2];
    int32_t W = in->desc.dims.d[3];
    PLUGIN_ASSERT(H % stride == 0);
    PLUGIN_ASSERT(W % stride == 0);
}

int32_t ReorgDynamic::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    void const* inputData = inputs[0];
    void* outputData = outputs[0];
    int32_t const N = inputDesc[0].dims.d[0];
    int32_t const C = inputDesc[0].dims.d[1];
    int32_t const H = inputDesc[0].dims.d[2];
    int32_t const W = inputDesc[0].dims.d[3];
    pluginStatus_t status = reorgInference(stream, N, C, H, W, stride, inputData, outputData);
    return status;
}

IPluginV2DynamicExt* ReorgDynamic::clone() const noexcept
{
    try
    {
        ReorgDynamic* plugin = new ReorgDynamic(stride);
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

ReorgStatic::ReorgStatic(int32_t stride)
    : Reorg<IPluginV2Ext>(stride)
{
}

ReorgStatic::ReorgStatic(int32_t C, int32_t H, int32_t W, int32_t stride)
    : Reorg<IPluginV2Ext>(stride)
    , C(C)
    , H(H)
    , W(W)
{
}

ReorgStatic::ReorgStatic(void const* buffer, size_t length)
{
    char const* d = reinterpret_cast<char const*>(buffer);
    char const* a = d;
    C = read<int32_t>(d);
    H = read<int32_t>(d);
    W = read<int32_t>(d);
    stride = read<int32_t>(d);
    PLUGIN_VALIDATE(d == a + length);
}

char const* ReorgStatic::getPluginVersion() const noexcept
{
    return kREORG_PLUGIN_STATIC_VERSION;
}

size_t ReorgStatic::getWorkspaceSize(int32_t maxBatchSize) const noexcept
{
    return 0;
}

size_t ReorgStatic::getSerializationSize() const noexcept
{
    // C, H, W, stride
    return sizeof(int32_t) * 4;
}

void ReorgStatic::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, C);
    write(d, H);
    write(d, W);
    write(d, stride);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

Dims ReorgStatic::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept
{
    PLUGIN_ASSERT(nbInputDims == 1);
    PLUGIN_ASSERT(index == 0);
    return Dims3(inputs[0].d[0] * stride * stride, inputs[0].d[1] / stride, inputs[0].d[2] / stride);
}

int32_t ReorgStatic::enqueue(
    int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    void const* inputData = inputs[0];
    void* outputData = outputs[0];
    pluginStatus_t status = reorgInference(stream, batchSize, C, H, W, stride, inputData, outputData);
    return status;
}

bool ReorgStatic::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

IPluginV2Ext* ReorgStatic::clone() const noexcept
{
    try
    {
        ReorgStatic* plugin = new ReorgStatic(C, H, W, stride);
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

// Return true if output tensor is broadcast across a batch.
bool ReorgStatic::isOutputBroadcastAcrossBatch(
    int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool ReorgStatic::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void ReorgStatic::configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
    DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
    bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept
{
    PLUGIN_ASSERT(*inputTypes == DataType::kFLOAT && floatFormat == PluginFormat::kLINEAR);
    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(stride > 0);
    C = inputDims[0].d[0];
    H = inputDims[0].d[1];
    W = inputDims[0].d[2];
    PLUGIN_ASSERT(H % stride == 0);
    PLUGIN_ASSERT(W % stride == 0);
}

template <class TPluginClass>
ReorgPluginCreator<TPluginClass>::ReorgPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

template <class TPluginClass>
char const* ReorgPluginCreator<TPluginClass>::getPluginName() const noexcept
{
    return kREORG_PLUGIN_NAME;
}

template <class TPluginClass>
char const* ReorgPluginCreator<TPluginClass>::getPluginVersion() const noexcept
{
    if (std::is_same_v<TPluginClass, ReorgStatic>)
    {
        return kREORG_PLUGIN_STATIC_VERSION;
    }
    else if (std::is_same_v<TPluginClass, ReorgDynamic>)
    {
        return kREORG_PLUGIN_DYNAMIC_VERSION;
    }
    return "";
}

template <class TPluginClass>
PluginFieldCollection const* ReorgPluginCreator<TPluginClass>::getFieldNames() noexcept
{
    return &mFC;
}

template <class TPluginClass>
IPluginV2Ext* ReorgPluginCreator<TPluginClass>::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PluginField const* fields = fc->fields;
        PLUGIN_VALIDATE(fc->nbFields == 1);
        PLUGIN_VALIDATE(fields[0].type == PluginFieldType::kINT32);
        PLUGIN_VALIDATE(!strcmp(fields[0].name, "stride"));
        stride = static_cast<int32_t>(*(static_cast<int32_t const*>(fields[0].data)));

        PLUGIN_VALIDATE(stride > 0);

        TPluginClass* obj = new TPluginClass(stride);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

template <class TPluginClass>
IPluginV2Ext* ReorgPluginCreator<TPluginClass>::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call ReorgPlugin::destroy()
        TPluginClass* obj = new TPluginClass(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

template class ReorgPluginCreator<ReorgStatic>;
template class ReorgPluginCreator<ReorgDynamic>;

} // namespace nvinfer1::plugin
