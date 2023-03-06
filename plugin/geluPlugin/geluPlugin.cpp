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

#include <cuda.h>
#if CUDA_VERSION >= 10010

#include <cstring>
#include <vector>

#include "NvInfer.h"
#include "common/bertCommon.h"
#include "common/serialize.hpp"
#include "geluPlugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace nvinfer1::plugin::bert;

namespace
{
char const* const kGELU_PLUGIN_VERSION{"1"};
char const* const kGELU_PLUGIN_NAME{"CustomGeluPluginDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection GeluPluginDynamicCreator::mFC{};
std::vector<PluginField> GeluPluginDynamicCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GeluPluginDynamicCreator);

GeluPluginDynamic::GeluPluginDynamic(const std::string name, const DataType type, Weights const& bias)
    : mLayerName(name)
    , mType(type)
    , mLd(bias.count)
{
    mHasBias = (bias.values != nullptr);
    if (mHasBias)
    {
        void* cudaMem{nullptr};
        PLUGIN_CHECK(cudaMalloc(&cudaMem, getWeightsSize(bias, mType)));
        PLUGIN_CHECK(cudaMemcpy(cudaMem, bias.values, getWeightsSize(bias, mType), cudaMemcpyHostToDevice));
        make_cuda_shared(mBiasDev, cudaMem);
    }
}

GeluPluginDynamic::GeluPluginDynamic(const std::string name, void const* data, size_t length)
    : mLayerName(name)
{
    gLogVerbose << "GeluPluginDynamic deserialize\n";
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mLd);
    deserialize_value(&data, &length, &mHasBias);

    if (mHasBias)
    {
        PLUGIN_VALIDATE(mLd > 0);
        char const* d = static_cast<char const*>(data);
        make_cuda_shared(mBiasDev, deserToDev<char>(d, mLd * getElementSize(mType)));
    }
}
// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* GeluPluginDynamic::clone() const noexcept
{
    try
    {
        gLogVerbose << "GeluPluginDynamic clone\n";
        auto* plugin = new GeluPluginDynamic(*this);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::DimsExprs GeluPluginDynamic::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return inputs[0];
}

bool GeluPluginDynamic::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{

    PluginTensorDesc const& input = inOut[0];
    if (pos == 0)
    {
        return (input.type == mType) && (input.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)
    {
        PluginTensorDesc const& output = inOut[1];
        return (input.type == output.type) && (output.format == TensorFormat::kLINEAR);
    }
    return false;
}

void GeluPluginDynamic::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    gLogVerbose << "GeluPluginDynamic configurePlugin\n";
    PLUGIN_ASSERT(mType == in[0].desc.type);
}

size_t GeluPluginDynamic::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}
int GeluPluginDynamic::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int const inputVolume = volume(inputDesc[0].dims);

    int status = -1;

    // Our plugin outputs only one tensor
    // Launch CUDA kernel wrapper and save its return value
    if (mType == DataType::kFLOAT)
    {
        float const* input = static_cast<float const*>(inputs[0]);
        float* output = static_cast<float*>(outputs[0]);
        if (mHasBias)
        {
            float const* bias = static_cast<float*>(mBiasDev.get());
            int const cols = inputVolume / mLd;
            int const rows = mLd;
            status = computeGeluBias(output, input, bias, rows, cols, stream);
        }
        else
        {
            status = computeGelu(stream, inputVolume, input, output);
        }
    }
    else if (mType == DataType::kHALF)
    {
        half const* input = static_cast<half const*>(inputs[0]);

        half* output = static_cast<half*>(outputs[0]);

        if (mHasBias)
        {
            half const* bias = static_cast<half*>(mBiasDev.get());
            int const cols = inputVolume / mLd;
            int const rows = mLd;
            status = computeGeluBias(output, input, bias, rows, cols, stream);
        }
        else
        {
            status = computeGelu(stream, inputVolume, input, output);
        }
    }
    else
    {
        return STATUS_FAILURE;
    }

    return status;
}

// IPluginV2Ext Methods
nvinfer1::DataType GeluPluginDynamic::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    PLUGIN_ASSERT(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
    return inputTypes[0];
}

// IPluginV2 Methods

char const* GeluPluginDynamic::getPluginType() const noexcept
{
    return kGELU_PLUGIN_NAME;
}

char const* GeluPluginDynamic::getPluginVersion() const noexcept
{
    return kGELU_PLUGIN_VERSION;
}

int GeluPluginDynamic::getNbOutputs() const noexcept
{
    return 1;
}

int GeluPluginDynamic::initialize() noexcept
{
    gLogVerbose << "GeluPluginDynamic initalize\n";
    return 0;
}

void GeluPluginDynamic::terminate() noexcept
{
    gLogVerbose << "GeluPluginDynamic terminate\n";
}

size_t GeluPluginDynamic::getSerializationSize() const noexcept
{
    const size_t wordSize = getElementSize(mType);
    const size_t biasSize = mHasBias ? mLd * wordSize : 0;
    return sizeof(mType) + sizeof(mHasBias) + sizeof(mLd) + biasSize;
}

void GeluPluginDynamic::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mLd);
    serialize_value(&buffer, mHasBias);
    if (mHasBias)
    {
        PLUGIN_ASSERT(mLd > 0);
        char* d = static_cast<char*>(buffer);
        serFromDev(d, static_cast<char*>(mBiasDev.get()), mLd * getElementSize(mType));
    }
}

void GeluPluginDynamic::destroy() noexcept
{
    gLogVerbose << "GeluPluginDynamic destroy\n";
    // This gets called when the network containing plugin is destroyed
    mBiasDev.reset();
    delete this;
}

void GeluPluginDynamic::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* GeluPluginDynamic::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////

GeluPluginDynamicCreator::GeluPluginDynamicCreator()
{
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* GeluPluginDynamicCreator::getPluginName() const noexcept
{
    return kGELU_PLUGIN_NAME;
}

char const* GeluPluginDynamicCreator::getPluginVersion() const noexcept
{
    return kGELU_PLUGIN_VERSION;
}

PluginFieldCollection const* GeluPluginDynamicCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* GeluPluginDynamicCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        gLogVerbose << "GeluPluginDynamicCreator createPlugin\n";

        Weights bias{DataType::kFLOAT, nullptr, 0};
        int32_t typeId = -1;
        plugin::validateRequiredAttributesExist({"type_id"}, fc);

        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            std::string field_name(fc->fields[i].name);

            if (field_name.compare("type_id") == 0)
            {
                typeId = *static_cast<int32_t const*>(fc->fields[i].data);
            }
            if (field_name.compare("bias") == 0)
            {
                bias.values = fc->fields[i].data;
                bias.count = fc->fields[i].length;
                bias.type = fieldTypeToDataType(fc->fields[i].type);
            }
        }

        if (typeId < 0 || typeId > 3)
        {
            gLogError << "GeluPluginDynamicCreator: invalid typeId " << typeId << std::endl;
            return nullptr;
        }

        return new GeluPluginDynamic(name, static_cast<DataType>(typeId), bias);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* GeluPluginDynamicCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call GeluPluginDynamic::destroy()
    try
    {
        return new GeluPluginDynamic(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void GeluPluginDynamicCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* GeluPluginDynamicCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

#endif // CUDA_VERSION >= 10010
