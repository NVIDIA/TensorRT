/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <cassert>
#include <cstring>
#include "pillarScatter.h"

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
  : feature_y_size_(h), feature_x_size_(w)
{
}

PillarScatterPlugin::PillarScatterPlugin(size_t h, size_t w, size_t channels)
  : feature_y_size_(h), feature_x_size_(w), featureNum_(channels)
{
}

PillarScatterPlugin::PillarScatterPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    feature_y_size_ = readFromBuffer<size_t>(d);
    feature_x_size_ = readFromBuffer<size_t>(d);
    featureNum_ = readFromBuffer<size_t>(d);
}

nvinfer1::IPluginV2DynamicExt* PillarScatterPlugin::clone() const noexcept
{
    auto* plugin = new PillarScatterPlugin(
        feature_y_size_, feature_x_size_, featureNum_
    );
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs PillarScatterPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    assert(outputIndex == 0);
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
    assert(nbInputs == 3);
    assert(nbOutputs == 1);
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
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
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    return false;
}

void PillarScatterPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    featureNum_ = in[0].desc.dims.d[2];
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
  int batchSize = inputDesc[0].dims.d[0];
  int maxPillarNum = inputDesc[0].dims.d[1];
  int numFeatures = inputDesc[0].dims.d[2];
  const float *pillar_features_data = (const float *)(inputs[0]);
  const unsigned int *coords_data = (const unsigned int *)(inputs[1]);
  const unsigned int *params_data = (const unsigned int *)(inputs[2]);
  unsigned int featureNum = featureNum_;
  unsigned int featureY = feature_y_size_;
  unsigned int featureX = feature_x_size_;
  float *spatial_feature_data = (float *)(outputs[0]);
  cudaMemsetAsync(spatial_feature_data, 0, batchSize*featureNum*featureY*featureX * sizeof(float), stream);
  pillarScatterKernelLaunch(
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
  return 0;
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
    writeToBuffer<size_t>(d, featureNum_);
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
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int target_h = 0;
    int target_w = 0;
    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        if (!strcmp(attr_name, "dense_shape"))
        {
            const int* ts = static_cast<const int*>(fields[i].data);
            target_h = ts[0];
            target_w = ts[1];
        }
    }
    IPluginV2* plugin = new PillarScatterPlugin(
        target_h,
        target_w
    );
    return plugin;
}

IPluginV2* PillarScatterPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed,
    IPluginV2* plugin = new PillarScatterPlugin(serialData, serialLength);
    return plugin;
}

void PillarScatterPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* PillarScatterPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
