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

#include "NvInfer.h"

#include "cropAndResizePlugin.h"
#include <cassert>
#include <cstring>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::CropAndResizePlugin;
using nvinfer1::plugin::CropAndResizePluginCreator;

// plugin specific constants
namespace
{
static const char* CROP_AND_RESIZE_PLUGIN_VERSION{"1"};
static const char* CROP_AND_RESIZE_PLUGIN_NAME{"CropAndResize"};
}

// Static class fields initialization
PluginFieldCollection CropAndResizePluginCreator::mFC{};
std::vector<PluginField> CropAndResizePluginCreator::mPluginAttributes;

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

CropAndResizePlugin::CropAndResizePlugin(const std::string name)
    : mLayerName(name)
    , mCropHeight(7)
    , mCropWidth(7)
{
}

CropAndResizePlugin::CropAndResizePlugin(const std::string name, int crop_width, int crop_height)
    : mLayerName(name)
    , mCropHeight(crop_height)
    , mCropWidth(crop_width)
{
}

CropAndResizePlugin::CropAndResizePlugin(const std::string name, const void* serial_buf, size_t serial_size)
    : mLayerName(name)
{
    const char* d = reinterpret_cast<const char*>(serial_buf);
    const char* a = d;
    mCropWidth = readFromBuffer<size_t>(d);
    mCropHeight = readFromBuffer<size_t>(d);
    mInputWidth = readFromBuffer<size_t>(d);
    mInputHeight = readFromBuffer<size_t>(d);
    mDepth = readFromBuffer<size_t>(d);
    mNumboxes = readFromBuffer<size_t>(d);
    ASSERT(d == a + sizeof(size_t) * 6);
}

CropAndResizePlugin::CropAndResizePlugin(const std::string name, int crop_width, int crop_height, int depth,
    int input_width, int input_height, int max_box_num)
    : mLayerName(name)
    , mCropHeight(crop_height)
    , mCropWidth(crop_width)
    , mDepth(depth)
    , mInputHeight(input_height)
    , mInputWidth(input_width)
    , mNumboxes(max_box_num)
{
}

CropAndResizePlugin::~CropAndResizePlugin()
{
}

const char* CropAndResizePlugin::getPluginType() const
{
    return CROP_AND_RESIZE_PLUGIN_NAME;
}

const char* CropAndResizePlugin::getPluginVersion() const
{
    return CROP_AND_RESIZE_PLUGIN_VERSION;
}

int CropAndResizePlugin::getNbOutputs() const
{
    return 1;
}

Dims CropAndResizePlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Validate input arguments
    ASSERT(index == 0);
    ASSERT(nbInputDims == 2);
    ASSERT(inputs->nbDims == 3);
    int channels = inputs->d[0];
    int height = mCropHeight;
    int width = mCropWidth;
    int roi_batch = inputs[1].d[0];
    return DimsNCHW(roi_batch, channels, height, width);
}

int CropAndResizePlugin::initialize()
{
    return 0;
}

int CropAndResizePlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    int status = -1;
    // Our plugin outputs only one tensor
    void* output = outputs[0];
    // Launch CUDA kernel wrapper and save its return value
    status = cropAndResizeInference(stream, mDepth * mInputHeight * mInputWidth * batchSize, inputs[0], inputs[1],
        batchSize, mInputHeight, mInputWidth, mNumboxes, mCropHeight, mCropWidth, mDepth, output);
    return status;
}

size_t CropAndResizePlugin::getSerializationSize() const
{
    return 6 * sizeof(size_t);
}

void CropAndResizePlugin::serialize(void* buffer) const
{
    char* d = reinterpret_cast<char*>(buffer);
    char* a = d;
    writeToBuffer<size_t>(d, mCropWidth);
    writeToBuffer<size_t>(d, mCropHeight);
    writeToBuffer<size_t>(d, mInputWidth);
    writeToBuffer<size_t>(d, mInputHeight);
    writeToBuffer<size_t>(d, mDepth);
    writeToBuffer<size_t>(d, mNumboxes);
    ASSERT(d == a + getSerializationSize());
}

bool CropAndResizePlugin::supportsFormat(DataType type, PluginFormat format) const
{
    // This plugin only supports ordinary floats, and NCHW input format
    if (type == DataType::kFLOAT && format == PluginFormat::kNCHW)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void CropAndResizePlugin::terminate()
{
}

void CropAndResizePlugin::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2Ext* CropAndResizePlugin::clone() const
{
    IPluginV2Ext* plugin
        = new CropAndResizePlugin(mLayerName, mCropWidth, mCropHeight, mDepth, mInputWidth, mInputHeight, mNumboxes);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void CropAndResizePlugin::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* CropAndResizePlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index.
DataType CropAndResizePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // one outputs
    ASSERT(index == 0);
    return DataType::kFLOAT;
}
// Return true if output tensor is broadcast across a batch.
bool CropAndResizePlugin::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool CropAndResizePlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

void CropAndResizePlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    ASSERT(
        inputTypes[0] == DataType::kFLOAT && inputTypes[1] == DataType::kFLOAT && floatFormat == PluginFormat::kNCHW);

    ASSERT(nbInputs == 2);
    ASSERT(nbOutputs == 1);
    mDepth = inputDims[0].d[0];
    mInputHeight = inputDims[0].d[1];
    mInputWidth = inputDims[0].d[2];
    mNumboxes = inputDims[1].d[0];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void CropAndResizePlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void CropAndResizePlugin::detachFromContext()
{
}

CropAndResizePluginCreator::CropAndResizePluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("crop_width", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("crop_height", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

CropAndResizePluginCreator::~CropAndResizePluginCreator()
{
}

const char* CropAndResizePluginCreator::getPluginName() const
{
    return CROP_AND_RESIZE_PLUGIN_NAME;
}

const char* CropAndResizePluginCreator::getPluginVersion() const
{
    return CROP_AND_RESIZE_PLUGIN_VERSION;
}

const PluginFieldCollection* CropAndResizePluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* CropAndResizePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int crop_width, crop_height;

    for (int i = 0; i < nbFields; ++i)
    {
        ASSERT(fields[i].type == PluginFieldType::kINT32);

        if (!strcmp(fields[i].name, "crop_width"))
        {
            crop_width = *(reinterpret_cast<const int*>(fields[i].data));
        }

        if (!strcmp(fields[i].name, "crop_height"))
        {
            crop_height = *(reinterpret_cast<const int*>(fields[i].data));
        }
    }

    ASSERT(crop_width > 0 && crop_height > 0);
    IPluginV2Ext* plugin = new CropAndResizePlugin(name, crop_width, crop_height);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* CropAndResizePluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed,
    IPluginV2Ext* plugin = new CropAndResizePlugin(name, serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
