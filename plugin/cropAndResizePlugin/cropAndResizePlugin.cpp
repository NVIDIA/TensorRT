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

#include "NvInfer.h"

#include "cropAndResizePlugin.h"
#include <cassert>
#include <cstring>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::CropAndResizePlugin;
using nvinfer1::plugin::CropAndResizeDynamicPlugin;
using nvinfer1::plugin::CropAndResizeBasePluginCreator;
using nvinfer1::plugin::CropAndResizePluginCreator;
using nvinfer1::plugin::CropAndResizeDynamicPluginCreator;

// plugin specific constants
namespace
{
static const char* CROP_AND_RESIZE_PLUGIN_VERSION{"1"};
static const char* CROP_AND_RESIZE_PLUGIN_NAMES[] = {"CropAndResize", "CropAndResizeDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection CropAndResizeBasePluginCreator::mFC{};
std::vector<PluginField> CropAndResizeBasePluginCreator::mPluginAttributes;

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

CropAndResizePlugin::CropAndResizePlugin(int crop_width, int crop_height) noexcept
    : mCropWidth(crop_width)
    , mCropHeight(crop_height)
{
}

CropAndResizeDynamicPlugin::CropAndResizeDynamicPlugin(int crop_width, int crop_height) noexcept
    : mCropWidth(crop_width)
    , mCropHeight(crop_height)
{
}

CropAndResizePlugin::CropAndResizePlugin(const void* serial_buf, size_t serial_size) noexcept
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

CropAndResizeDynamicPlugin::CropAndResizeDynamicPlugin(const void* serial_buf, size_t serial_size) noexcept
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

CropAndResizePlugin::CropAndResizePlugin(int crop_width, int crop_height, int depth,
    int input_width, int input_height, int max_box_num) noexcept
    : mCropWidth(crop_width)
    , mCropHeight(crop_height)
    , mDepth(depth)
    , mInputWidth(input_width)
    , mInputHeight(input_height)
    , mNumboxes(max_box_num)
{
}

CropAndResizeDynamicPlugin::CropAndResizeDynamicPlugin(int crop_width, int crop_height, int depth,
    int input_width, int input_height, int max_box_num) noexcept
    : mCropWidth(crop_width)
    , mCropHeight(crop_height)
    , mDepth(depth)
    , mInputWidth(input_width)
    , mInputHeight(input_height)
    , mNumboxes(max_box_num)
{
}

CropAndResizePlugin::~CropAndResizePlugin() noexcept
{

}

CropAndResizeDynamicPlugin::~CropAndResizeDynamicPlugin() noexcept
{

}

const char* CropAndResizePlugin::getPluginType() const noexcept
{
    return CROP_AND_RESIZE_PLUGIN_NAMES[0];
}

const char* CropAndResizeDynamicPlugin::getPluginType() const noexcept
{
    return CROP_AND_RESIZE_PLUGIN_NAMES[1];
}

const char* CropAndResizePlugin::getPluginVersion() const noexcept
{
    return CROP_AND_RESIZE_PLUGIN_VERSION;
}

const char* CropAndResizeDynamicPlugin::getPluginVersion() const noexcept
{
    return CROP_AND_RESIZE_PLUGIN_VERSION;
}

int CropAndResizePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int CropAndResizeDynamicPlugin::getNbOutputs() const noexcept
{
    return 1;
}

Dims CropAndResizePlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
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

DimsExprs CropAndResizeDynamicPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    ASSERT(outputIndex == 0);
    ASSERT(nbInputs == 2);
    ASSERT(inputs[0].nbDims == 4);
    DimsExprs out_dim;
    // 5D output (N, R, C, H, W)
    out_dim.nbDims = 5;
    out_dim.d[0] = inputs[0].d[0]; // N(batch)
    out_dim.d[1] = inputs[1].d[1]; // R(MaxBoxNum)
    out_dim.d[2] = inputs[0].d[1]; // C(channel)
    out_dim.d[3] = exprBuilder.constant(mCropHeight);
    out_dim.d[4] = exprBuilder.constant(mCropWidth);
    return out_dim;
}

int CropAndResizePlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

int CropAndResizeDynamicPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

int CropAndResizePlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream) noexcept
{
    int status = -1;
    // Our plugin outputs only one tensor
    void* output = outputs[0];
    // Launch CUDA kernel wrapper and save its return value
    status = cropAndResizeInference(stream, mDepth * mInputHeight * mInputWidth * batchSize, inputs[0], inputs[1],
        batchSize, mInputHeight, mInputWidth, mNumboxes, mCropHeight, mCropWidth, mDepth, output);
    ASSERT(status == STATUS_SUCCESS);
    return status;
}

int CropAndResizeDynamicPlugin::enqueue(
    const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int status = -1;
    // Our plugin outputs only one tensor
    void* output = outputs[0];
    // Launch CUDA kernel wrapper and save its return value
    int batchSize = inputDesc[0].dims.d[0];
    status = cropAndResizeInference(stream, mDepth * mInputHeight * mInputWidth * batchSize, inputs[0], inputs[1],
        batchSize, mInputHeight, mInputWidth, mNumboxes, mCropHeight, mCropWidth, mDepth, output);
    ASSERT(status == STATUS_SUCCESS);
    return status;
}

size_t CropAndResizePlugin::getSerializationSize() const noexcept
{
    return 6 * sizeof(size_t);
}

size_t CropAndResizeDynamicPlugin::getSerializationSize() const noexcept
{
    return 6 * sizeof(size_t);
}

void CropAndResizePlugin::serialize(void* buffer) const noexcept
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

void CropAndResizeDynamicPlugin::serialize(void* buffer) const noexcept
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

bool CropAndResizePlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
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

bool CropAndResizeDynamicPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // 2 inputs, 1 outputs, so 3 input/output in total
    ASSERT(0 <= pos && pos < 3);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    const bool consistentFloatPrecision = (in[0].type == in[pos].type);
    switch (pos)
    {
    case 0: return in[0].type == DataType::kFLOAT && in[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 1: return in[1].type == DataType::kFLOAT && in[1].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 2: return out[0].type == DataType::kFLOAT && out[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    }
    return false;
}

void CropAndResizePlugin::terminate() noexcept {}

void CropAndResizeDynamicPlugin::terminate() noexcept {}

size_t CropAndResizePlugin::getWorkspaceSize(int) const noexcept
{
return 0;
}

size_t CropAndResizeDynamicPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
return 0;
}

void CropAndResizePlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void CropAndResizeDynamicPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2Ext* CropAndResizePlugin::clone() const noexcept
{
    IPluginV2Ext* plugin
        = new CropAndResizePlugin(mCropWidth, mCropHeight, mDepth, mInputWidth, mInputHeight, mNumboxes);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* CropAndResizeDynamicPlugin::clone() const noexcept
{
    IPluginV2DynamicExt* plugin
        = new CropAndResizeDynamicPlugin(mCropWidth, mCropHeight, mDepth, mInputWidth, mInputHeight, mNumboxes);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void CropAndResizePlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

void CropAndResizeDynamicPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* CropAndResizePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

const char* CropAndResizeDynamicPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index.
DataType CropAndResizePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // one outputs
    ASSERT(index == 0);
    return DataType::kFLOAT;
}

DataType CropAndResizeDynamicPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // one outputs
    ASSERT(index == 0);
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool CropAndResizePlugin::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool CropAndResizePlugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

void CropAndResizePlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    ASSERT(nbInputs == 2);
    ASSERT(nbOutputs == 1);
    mDepth = inputDims[0].d[0];
    mInputHeight = inputDims[0].d[1];
    mInputWidth = inputDims[0].d[2];
    mNumboxes = inputDims[1].d[0];
}

void CropAndResizeDynamicPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    ASSERT(nbInputs == 2);
    ASSERT(nbOutputs == 1);
    mDepth = in[0].desc.dims.d[1];
    mInputHeight = in[0].desc.dims.d[2];
    mInputWidth = in[0].desc.dims.d[3];
    mNumboxes = in[1].desc.dims.d[1];
}
// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void CropAndResizePlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void CropAndResizePlugin::detachFromContext() noexcept {}


CropAndResizeBasePluginCreator::CropAndResizeBasePluginCreator() noexcept
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("crop_width", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("crop_height", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

CropAndResizePluginCreator::CropAndResizePluginCreator() noexcept
{
    mPluginName = CROP_AND_RESIZE_PLUGIN_NAMES[0];
}

CropAndResizeDynamicPluginCreator::CropAndResizeDynamicPluginCreator() noexcept
{
    mPluginName = CROP_AND_RESIZE_PLUGIN_NAMES[1];
}

const char* CropAndResizeBasePluginCreator::getPluginName() const noexcept
{
    return mPluginName.c_str();
}

const char* CropAndResizeBasePluginCreator::getPluginVersion() const noexcept
{
    return CROP_AND_RESIZE_PLUGIN_VERSION;
}

const PluginFieldCollection* CropAndResizeBasePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* CropAndResizePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int crop_width = 0, crop_height = 0;

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
    IPluginV2Ext* plugin = new CropAndResizePlugin(crop_width, crop_height);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* CropAndResizeDynamicPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int crop_width = 0, crop_height = 0;

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
    IPluginV2DynamicExt* plugin = new CropAndResizeDynamicPlugin(crop_width, crop_height);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* CropAndResizePluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed,
    IPluginV2Ext* plugin = new CropAndResizePlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* CropAndResizeDynamicPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed,
    IPluginV2DynamicExt* plugin = new CropAndResizeDynamicPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
