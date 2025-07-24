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

#include "NvInfer.h"

#include "common/templates.h"
#include "cropAndResizePluginLegacy.h"
#include <cstring>
#include <vector>

namespace nvinfer1::plugin
{
// Plugin-specific constants
namespace
{
char const* const kCROP_AND_RESIZE_PLUGIN_VERSION{"1"};
char const* const kCROP_AND_RESIZE_DYNAMIC_PLUGIN_VERSION{"1"};
char const* const kCROP_AND_RESIZE_PLUGIN_NAME{"CropAndResize"};
char const* const kCROP_AND_RESIZE_DYNAMIC_PLUGIN_NAME{"CropAndResizeDynamic"};
} // namespace

// Legacy Plugin Implementation
CropAndResizePlugin::CropAndResizePlugin(int32_t cropWidth, int32_t cropHeight)
    : mCropWidth(cropWidth)
    , mCropHeight(cropHeight)
{
}

CropAndResizeDynamicPluginLegacy::CropAndResizeDynamicPluginLegacy(int32_t cropWidth, int32_t cropHeight)
    : mCropWidth(cropWidth)
    , mCropHeight(cropHeight)
{
}

CropAndResizePlugin::CropAndResizePlugin(void const* serialBuf, size_t serialSize)
{
    auto const* d = toPointer<uint8_t const>(serialBuf);
    auto const* a = d;
    mCropWidth = readFromBuffer<int32_t>(d);
    mCropHeight = readFromBuffer<int32_t>(d);
    mInputWidth = readFromBuffer<int32_t>(d);
    mInputHeight = readFromBuffer<int32_t>(d);
    mDepth = readFromBuffer<int32_t>(d);
    mNumBoxes = readFromBuffer<int32_t>(d);
    PLUGIN_ASSERT(d == a + sizeof(int32_t) * 6);
}

CropAndResizeDynamicPluginLegacy::CropAndResizeDynamicPluginLegacy(void const* serialBuf, size_t serialSize)
{
    auto const* d = reinterpret_cast<uint8_t const*>(serialBuf);
    auto const* a = d;
    mCropWidth = readFromBuffer<int32_t>(d);
    mCropHeight = readFromBuffer<int32_t>(d);
    mInputWidth = readFromBuffer<int32_t>(d);
    mInputHeight = readFromBuffer<int32_t>(d);
    mDepth = readFromBuffer<int32_t>(d);
    mNumBoxes = readFromBuffer<int32_t>(d);
    PLUGIN_ASSERT(d == a + sizeof(int32_t) * 6);
}

CropAndResizePlugin::CropAndResizePlugin(
    int32_t cropWidth, int32_t cropHeight, int32_t depth, int32_t inputWidth, int32_t inputHeight, int32_t maxBoxNum)
    : mCropWidth(cropWidth)
    , mCropHeight(cropHeight)
    , mDepth(depth)
    , mInputWidth(inputWidth)
    , mInputHeight(inputHeight)
    , mNumBoxes(maxBoxNum)
{
}

CropAndResizeDynamicPluginLegacy::CropAndResizeDynamicPluginLegacy(
    int32_t cropWidth, int32_t cropHeight, int32_t depth, int32_t inputWidth, int32_t inputHeight, int32_t maxBoxNum)
    : mCropWidth(cropWidth)
    , mCropHeight(cropHeight)
    , mDepth(depth)
    , mInputWidth(inputWidth)
    , mInputHeight(inputHeight)
    , mNumBoxes(maxBoxNum)
{
}

CropAndResizePlugin::~CropAndResizePlugin() {}

CropAndResizeDynamicPluginLegacy::~CropAndResizeDynamicPluginLegacy() noexcept {}

char const* CropAndResizePlugin::getPluginType() const noexcept
{
    return kCROP_AND_RESIZE_PLUGIN_NAME;
}

char const* CropAndResizeDynamicPluginLegacy::getPluginType() const noexcept
{
    return kCROP_AND_RESIZE_DYNAMIC_PLUGIN_NAME;
}

char const* CropAndResizePlugin::getPluginVersion() const noexcept
{
    return kCROP_AND_RESIZE_PLUGIN_VERSION;
}

char const* CropAndResizeDynamicPluginLegacy::getPluginVersion() const noexcept
{
    return kCROP_AND_RESIZE_DYNAMIC_PLUGIN_VERSION;
}

int32_t CropAndResizePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t CropAndResizeDynamicPluginLegacy::getNbOutputs() const noexcept
{
    return 1;
}

Dims CropAndResizePlugin::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept
{
    try
    {
        PLUGIN_VALIDATE(index == 0);
        PLUGIN_VALIDATE(nbInputDims == 2);
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(inputs->nbDims == 3);
        int32_t channels = inputs->d[0];
        int32_t height = mCropHeight;
        int32_t width = mCropWidth;
        int32_t roiBatch = inputs[1].d[0];
        return Dims4(roiBatch, channels, height, width);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return Dims{};
}

DimsExprs CropAndResizeDynamicPluginLegacy::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(outputIndex == 0);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(inputs[0].nbDims == 4);
        DimsExprs outDim;
        // 5D output (N, R, C, H, W)
        outDim.nbDims = 5;
        outDim.d[0] = inputs[0].d[0]; // N(batch)
        outDim.d[1] = inputs[1].d[1]; // R(MaxBoxNum)
        outDim.d[2] = inputs[0].d[1]; // C(channel)
        outDim.d[3] = exprBuilder.constant(mCropHeight);
        outDim.d[4] = exprBuilder.constant(mCropWidth);
        return outDim;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

int32_t CropAndResizePlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

int32_t CropAndResizeDynamicPluginLegacy::initialize() noexcept
{
    return STATUS_SUCCESS;
}

int32_t CropAndResizePlugin::enqueue(
    int32_t batchSize, void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);

        // Our plugin outputs only one tensor
        void* output = outputs[0];

        // Launch CUDA kernel wrapper and save its return value
        int32_t status = cropAndResizeInference(stream, mDepth * mInputHeight * mInputWidth * batchSize, inputs[0],
            inputs[1], batchSize, mInputHeight, mInputWidth, mNumBoxes, mCropHeight, mCropWidth, mDepth, output);
        return status;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

int32_t CropAndResizeDynamicPluginLegacy::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /* outputDesc */, void const* const* inputs, void* const* outputs, void* /* workspace */,
    cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr && inputs != nullptr && outputs != nullptr);

        // Our plugin outputs only one tensor
        void* output = outputs[0];

        // Launch CUDA kernel wrapper and save its return value
        int32_t batchSize = inputDesc[0].dims.d[0];
        int32_t status = cropAndResizeInference(stream, mDepth * mInputHeight * mInputWidth * batchSize, inputs[0],
            inputs[1], batchSize, mInputHeight, mInputWidth, mNumBoxes, mCropHeight, mCropWidth, mDepth, output);
        return status;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

size_t CropAndResizePlugin::getSerializationSize() const noexcept
{
    return 6 * sizeof(int32_t);
}

size_t CropAndResizeDynamicPluginLegacy::getSerializationSize() const noexcept
{
    return 6 * sizeof(int32_t);
}

void CropAndResizePlugin::serialize(void* buffer) const noexcept
{
    auto* d = reinterpret_cast<uint8_t*>(buffer);
    auto* const a = d;
    writeToBuffer<int32_t>(d, mCropWidth);
    writeToBuffer<int32_t>(d, mCropHeight);
    writeToBuffer<int32_t>(d, mInputWidth);
    writeToBuffer<int32_t>(d, mInputHeight);
    writeToBuffer<int32_t>(d, mDepth);
    writeToBuffer<int32_t>(d, mNumBoxes);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void CropAndResizeDynamicPluginLegacy::serialize(void* buffer) const noexcept
{
    auto* d = reinterpret_cast<uint8_t*>(buffer);
    auto* const a = d;
    writeToBuffer<int32_t>(d, mCropWidth);
    writeToBuffer<int32_t>(d, mCropHeight);
    writeToBuffer<int32_t>(d, mInputWidth);
    writeToBuffer<int32_t>(d, mInputHeight);
    writeToBuffer<int32_t>(d, mDepth);
    writeToBuffer<int32_t>(d, mNumBoxes);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

bool CropAndResizePlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    // This plugin only supports ordinary floats, and NCHW input format
    if (type == DataType::kFLOAT && format == PluginFormat::kLINEAR)
    {
        return true;
    }

    return false;
}

bool CropAndResizeDynamicPluginLegacy::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        // 2 inputs, 1 outputs, so 3 input/output in total
        PLUGIN_VALIDATE(0 <= pos && pos < 3);
        PLUGIN_VALIDATE(inOut != nullptr);
        auto const* in = inOut;
        auto const* out = inOut + nbInputs;
        bool const consistentFloatPrecision = (in[0].type == in[pos].type);
        switch (pos)
        {
        case 0:
            return in[0].type == DataType::kFLOAT && in[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
        case 1:
            return in[1].type == DataType::kFLOAT && in[1].format == PluginFormat::kLINEAR && consistentFloatPrecision;
        case 2:
            return out[0].type == DataType::kFLOAT && out[0].format == PluginFormat::kLINEAR
                && consistentFloatPrecision;
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

void CropAndResizePlugin::terminate() noexcept {}

void CropAndResizeDynamicPluginLegacy::terminate() noexcept {}

size_t CropAndResizePlugin::getWorkspaceSize(int32_t /*maxBatchSize*/) const noexcept
{
    return 0;
}

size_t CropAndResizeDynamicPluginLegacy::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

void CropAndResizePlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void CropAndResizeDynamicPluginLegacy::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2Ext* CropAndResizePlugin::clone() const noexcept
{
    try
    {
        IPluginV2Ext* plugin
            = new CropAndResizePlugin(mCropWidth, mCropHeight, mDepth, mInputWidth, mInputHeight, mNumBoxes);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* CropAndResizeDynamicPluginLegacy::clone() const noexcept
{
    try
    {
        IPluginV2DynamicExt* plugin = new CropAndResizeDynamicPluginLegacy(
            mCropWidth, mCropHeight, mDepth, mInputWidth, mInputHeight, mNumBoxes);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void CropAndResizePlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

void CropAndResizeDynamicPluginLegacy::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* CropAndResizePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* CropAndResizeDynamicPluginLegacy::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index.
DataType CropAndResizePlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        // One output.
        PLUGIN_VALIDATE(index == 0);
        return DataType::kFLOAT;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DataType{};
}

DataType CropAndResizeDynamicPluginLegacy::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        // One output.
        PLUGIN_VALIDATE(index == 0);
        return DataType::kFLOAT;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DataType{};
}

// Return true if output tensor is broadcast across a batch.
bool CropAndResizePlugin::isOutputBroadcastAcrossBatch(
    int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool CropAndResizePlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{
    return false;
}

void CropAndResizePlugin::configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims,
    int32_t nbOutputs, DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
    bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputTypes[0] == DataType::kFLOAT && inputTypes[1] == DataType::kFLOAT
            && floatFormat == PluginFormat::kLINEAR);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);
        mDepth = inputDims[0].d[0];
        mInputHeight = inputDims[0].d[1];
        mInputWidth = inputDims[0].d[2];
        mNumBoxes = inputDims[1].d[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

void CropAndResizeDynamicPluginLegacy::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);
        mDepth = in[0].desc.dims.d[1];
        mInputHeight = in[0].desc.dims.d[2];
        mInputWidth = in[0].desc.dims.d[3];
        mNumBoxes = in[1].desc.dims.d[1];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}
// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void CropAndResizePlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void CropAndResizePlugin::detachFromContext() noexcept {}

// Base Creator Implementation
CropAndResizeBasePluginCreator::CropAndResizeBasePluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("crop_width", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("crop_height", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* CropAndResizeBasePluginCreator::getPluginName() const noexcept
{
    return mPluginName.c_str();
}

char const* CropAndResizeBasePluginCreator::getPluginVersion() const noexcept
{
    return mPluginVersion.c_str();
}

PluginFieldCollection const* CropAndResizeBasePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

// Legacy Creator Implementation
CropAndResizePluginCreator::CropAndResizePluginCreator()
{
    mPluginName = kCROP_AND_RESIZE_PLUGIN_NAME;
    mPluginVersion = kCROP_AND_RESIZE_PLUGIN_VERSION;
}

CropAndResizeDynamicPluginLegacyCreator::CropAndResizeDynamicPluginLegacyCreator()
{
    mPluginName = kCROP_AND_RESIZE_DYNAMIC_PLUGIN_NAME;
    mPluginVersion = kCROP_AND_RESIZE_DYNAMIC_PLUGIN_VERSION;
}

IPluginV2Ext* CropAndResizePluginCreator::createPlugin(char const* /* name */, PluginFieldCollection const* fc) noexcept
{
    try
    {
        gLogWarning << "CropAndResizePlugin (implementing IPluginV2Ext) is deprecated since TensorRT 9.0. Use "
                       "CropAndResizeDynamic plugin."
                    << std::endl;
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const* fields = fc->fields;
        int32_t nbFields = fc->nbFields;
        int32_t cropWidth = 0;
        int32_t cropHeight = 0;

        validateRequiredAttributesExist({"crop_width", "crop_height"}, fc);

        for (int32_t i = 0; i < nbFields; ++i)
        {
            PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);

            if (!strcmp(fields[i].name, "crop_width"))
            {
                cropWidth = *(reinterpret_cast<int32_t const*>(fields[i].data));
            }

            if (!strcmp(fields[i].name, "crop_height"))
            {
                cropHeight = *(reinterpret_cast<int32_t const*>(fields[i].data));
            }
        }

        PLUGIN_VALIDATE(cropWidth > 0 && cropHeight > 0);
        IPluginV2Ext* plugin = new CropAndResizePlugin(cropWidth, cropHeight);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* CropAndResizeDynamicPluginLegacyCreator::createPlugin(
    char const* /*name */, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const* fields = fc->fields;
        int32_t nbFields = fc->nbFields;
        int32_t cropWidth = 0;
        int32_t cropHeight = 0;

        validateRequiredAttributesExist({"crop_width", "crop_height"}, fc);

        for (int32_t i = 0; i < nbFields; ++i)
        {
            PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);

            if (!strcmp(fields[i].name, "crop_width"))
            {
                cropWidth = *(reinterpret_cast<int32_t const*>(fields[i].data));
            }

            if (!strcmp(fields[i].name, "crop_height"))
            {
                cropHeight = *(reinterpret_cast<int32_t const*>(fields[i].data));
            }
        }

        PLUGIN_VALIDATE(cropWidth > 0 && cropHeight > 0);
        IPluginV2DynamicExt* plugin = new CropAndResizeDynamicPluginLegacy(cropWidth, cropHeight);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* CropAndResizePluginCreator::deserializePlugin(
    char const* /* name */, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        gLogWarning << "CropAndResizePlugin (implementing IPluginV2Ext) is deprecated since TensorRT 9.0. Use "
                       "CropAndResizeDynamic plugin."
                    << std::endl;
        // This object will be deleted when the network is destroyed,
        IPluginV2Ext* plugin = new CropAndResizePlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* CropAndResizeDynamicPluginLegacyCreator::deserializePlugin(
    char const* /* name */, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed,
        IPluginV2DynamicExt* plugin = new CropAndResizeDynamicPluginLegacy(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

} // namespace nvinfer1::plugin
