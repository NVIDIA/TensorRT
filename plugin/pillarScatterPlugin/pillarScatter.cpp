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

#include "pillarScatter.h"
#include <cstring>

using namespace nvinfer1;
using nvinfer1::plugin::PillarScatterPlugin;
using nvinfer1::plugin::PillarScatterPluginCreator;

static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"PillarScatterPlugin"};

// Static class fields initialization
PluginFieldCollection PillarScatterPluginCreator::mFC{};
std::vector<PluginField> PillarScatterPluginCreator::mPluginAttributes;

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

PillarScatterPlugin::PillarScatterPlugin(size_t h, size_t w)
    : feature_y_size_(h)
    , feature_x_size_(w)
{
}

PillarScatterPlugin::PillarScatterPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    feature_y_size_ = readFromBuffer<size_t>(d);
    feature_x_size_ = readFromBuffer<size_t>(d);
}

nvinfer1::IPluginV2DynamicExt* PillarScatterPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new PillarScatterPlugin(feature_y_size_, feature_x_size_);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::DimsExprs PillarScatterPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(outputIndex == 0);
    nvinfer1::DimsExprs output;
    auto batch_size = inputs[0].d[0];
    output.nbDims = 4;
    output.d[0] = batch_size;
    output.d[1] = inputs[0].d[2];
    output.d[2] = exprBuilder.constant(feature_y_size_);
    output.d[3] = exprBuilder.constant(feature_x_size_);
    return output;
}

bool PillarScatterPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(nbOutputs == 1);
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)
    {
        return (in.type == nvinfer1::DataType::kFLOAT || in.type == nvinfer1::DataType::kHALF) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2)
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 3)
    {
        return (in.type == inOut[0].type) && (in.format == TensorFormat::kLINEAR || in.format == TensorFormat::kHWC8);
    }
    return false;
}

void PillarScatterPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    return;
}

size_t PillarScatterPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}
int PillarScatterPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    try
    {
        int batchSize = inputDesc[0].dims.d[0];
        int maxPillarNum = inputDesc[0].dims.d[1];
        int numFeatures = inputDesc[0].dims.d[2];

        nvinfer1::DataType inputType = inputDesc[0].type;

        auto coords_data = static_cast<const unsigned int *>(inputs[1]);
        auto params_data = static_cast<const unsigned int *>(inputs[2]);

        unsigned int featureY = feature_y_size_;
        unsigned int featureX = feature_x_size_;

        int status = -1;

        if(inputType == nvinfer1::DataType::kHALF){
            auto pillar_features_data = static_cast<const half *>(inputs[0]);
            auto spatial_feature_data = static_cast<half *>(outputs[0]);
            cudaMemsetAsync(spatial_feature_data, 0, batchSize*numFeatures*featureY*featureX * sizeof(half), stream);
            status = pillarScatterKernelLaunch<half>(
                batchSize,
                maxPillarNum,
                numFeatures,
                pillar_features_data,
                coords_data,
                params_data,
                featureX,
                featureY,
                spatial_feature_data,
                stream
                );
            PLUGIN_ASSERT(status == STATUS_SUCCESS);
            return status;
        }
        else if(inputType == nvinfer1::DataType::kFLOAT){
            auto pillar_features_data = static_cast<const float *>(inputs[0]);
            auto spatial_feature_data = static_cast<float *>(outputs[0]);
            cudaMemsetAsync(spatial_feature_data, 0, batchSize*numFeatures*featureY*featureX * sizeof(float), stream);
            status = pillarScatterKernelLaunch<float>(
                batchSize,
                maxPillarNum,
                numFeatures,
                pillar_features_data,
                coords_data,
                params_data,
                featureX,
                featureY,
                spatial_feature_data,
                stream
                );
            PLUGIN_ASSERT(status == STATUS_SUCCESS);
            return status;
        }
        else{
            PLUGIN_ASSERT(status == STATUS_SUCCESS);
            return status;
        }
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return -1;
}

nvinfer1::DataType PillarScatterPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

const char* PillarScatterPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* PillarScatterPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int PillarScatterPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int PillarScatterPlugin::initialize() noexcept
{
    return 0;
}

void PillarScatterPlugin::terminate() noexcept
{
}

size_t PillarScatterPlugin::getSerializationSize() const noexcept
{
    return 3 * sizeof(size_t);
}

void PillarScatterPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<size_t>(d, feature_y_size_);
    writeToBuffer<size_t>(d, feature_x_size_);
}

void PillarScatterPlugin::destroy() noexcept
{
    delete this;
}

void PillarScatterPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* PillarScatterPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

PillarScatterPluginCreator::PillarScatterPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("dense_shape", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* PillarScatterPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* PillarScatterPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* PillarScatterPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* PillarScatterPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        int32_t nbFields = fc->nbFields;
        int32_t targetH = 0;
        int32_t targetW = 0;
        plugin::validateRequiredAttributesExist({"dense_shape"}, fc);
        for (int32_t i = 0; i < nbFields; ++i)
        {
            if (!strcmp(fields[i].name, "dense_shape"))
            {
                int32_t const* ts = static_cast<int32_t const*>(fields[i].data);
                targetH = ts[0];
                targetW = ts[1];
                PLUGIN_VALIDATE(targetH > 0 && targetW > 0);
            }
        }
        IPluginV2* plugin = new PillarScatterPlugin(targetH, targetW);
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* PillarScatterPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed,
        IPluginV2* plugin = new PillarScatterPlugin(serialData, serialLength);
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void PillarScatterPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* PillarScatterPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
