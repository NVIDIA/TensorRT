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

#include "pillarScatter.h"
#include "common/templates.h"
#include <cstring>

namespace nvinfer1::plugin
{

static char const* const kPLUGIN_VERSION{"1"};
static char const* const kPLUGIN_NAME{"PillarScatterPlugin"};

PillarScatterPlugin::PillarScatterPlugin(size_t h, size_t w)
    : feature_y_size_(h)
    , feature_x_size_(w)
{
}

PillarScatterPlugin::PillarScatterPlugin(void const* data, size_t length)
{
    auto const* d = toPointer<char const>(data);
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

nvinfer1::DimsExprs PillarScatterPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
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
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(nbOutputs == 1);
    PluginTensorDesc const& in = inOut[pos];
    if (pos == 0)
    {
        return (in.type == nvinfer1::DataType::kFLOAT || in.type == nvinfer1::DataType::kHALF)
            && (in.format == TensorFormat::kLINEAR);
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
        return (in.type == inOut[0].type) && (in.format == TensorFormat::kLINEAR);
    }
    return false;
}

void PillarScatterPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return;
}

size_t PillarScatterPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}
int32_t PillarScatterPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* /* outputDesc */, void const* const* inputs, void* const* outputs,
    void* /* workspace */, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr && inputs != nullptr && outputs != nullptr);

        int32_t batchSize = inputDesc[0].dims.d[0];
        int32_t maxPillarNum = inputDesc[0].dims.d[1];
        int32_t numFeatures = inputDesc[0].dims.d[2];

        nvinfer1::DataType inputType = inputDesc[0].type;

        auto coords_data = static_cast<uint32_t const*>(inputs[1]);
        auto params_data = static_cast<uint32_t const*>(inputs[2]);

        uint32_t featureY = feature_y_size_;
        uint32_t featureX = feature_x_size_;

        int32_t status = -1;

        if (inputType == nvinfer1::DataType::kHALF)
        {
            auto pillar_features_data = static_cast<half const*>(inputs[0]);
            auto spatial_feature_data = static_cast<half*>(outputs[0]);
            cudaMemsetAsync(
                spatial_feature_data, 0, batchSize * numFeatures * featureY * featureX * sizeof(half), stream);
            status = pillarScatterKernelLaunch<half>(batchSize, maxPillarNum, numFeatures, pillar_features_data,
                coords_data, params_data, featureX, featureY, spatial_feature_data, stream);
        }
        else if (inputType == nvinfer1::DataType::kFLOAT)
        {
            auto const* pillar_features_data = static_cast<float const*>(inputs[0]);
            auto* spatial_feature_data = static_cast<float*>(outputs[0]);
            cudaMemsetAsync(
                spatial_feature_data, 0, batchSize * numFeatures * featureY * featureX * sizeof(float), stream);
            status = pillarScatterKernelLaunch<float>(batchSize, maxPillarNum, numFeatures, pillar_features_data,
                coords_data, params_data, featureX, featureY, spatial_feature_data, stream);
        }
        PLUGIN_ASSERT(status == STATUS_SUCCESS);
        return status;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return -1;
}

nvinfer1::DataType PillarScatterPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

char const* PillarScatterPlugin::getPluginType() const noexcept
{
    return kPLUGIN_NAME;
}

char const* PillarScatterPlugin::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

int32_t PillarScatterPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t PillarScatterPlugin::initialize() noexcept
{
    return 0;
}

void PillarScatterPlugin::terminate() noexcept {}

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

void PillarScatterPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* PillarScatterPlugin::getPluginNamespace() const noexcept
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

char const* PillarScatterPluginCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* PillarScatterPluginCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

PluginFieldCollection const* PillarScatterPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* PillarScatterPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PluginField const* fields = fc->fields;
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
    char const* name, void const* serialData, size_t serialLength) noexcept
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

void PillarScatterPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* PillarScatterPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
} // namespace nvinfer1::plugin
