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

#include "multilevelCropAndResizePlugin.h"
#include "common/plugin.h"
#include <algorithm>
#include <cuda_runtime_api.h>

#include <fstream>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::MultilevelCropAndResize;
using nvinfer1::plugin::MultilevelCropAndResizePluginCreator;

namespace
{
const char* MULTILEVELCROPANDRESIZE_PLUGIN_VERSION{"1"};
const char* MULTILEVELCROPANDRESIZE_PLUGIN_NAME{"MultilevelCropAndResize_TRT"};
} // namespace

PluginFieldCollection MultilevelCropAndResizePluginCreator::mFC{};
std::vector<PluginField> MultilevelCropAndResizePluginCreator::mPluginAttributes;

MultilevelCropAndResizePluginCreator::MultilevelCropAndResizePluginCreator() noexcept
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("pooled_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("image_size", nullptr, PluginFieldType::kINT32, 3));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* MultilevelCropAndResizePluginCreator::getPluginName() const noexcept
{
    return MULTILEVELCROPANDRESIZE_PLUGIN_NAME;
}

const char* MultilevelCropAndResizePluginCreator::getPluginVersion() const noexcept
{
    return MULTILEVELCROPANDRESIZE_PLUGIN_VERSION;
}

const PluginFieldCollection* MultilevelCropAndResizePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* MultilevelCropAndResizePluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        plugin::validateRequiredAttributesExist({"pooled_size"}, fc);

        auto imageSize = TLTMaskRCNNConfig::IMAGE_SHAPE;
        const PluginField* fields = fc->fields;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "pooled_size"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                mPooledSize = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "image_size"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                auto const dims = static_cast<int32_t const*>(fields[i].data);
                std::copy_n(dims, 3, imageSize.d);
            }
        }
        return new MultilevelCropAndResize(mPooledSize, imageSize);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* MultilevelCropAndResizePluginCreator::deserializePlugin(
    const char* name, const void* data, size_t length) noexcept
{
    try
    {
        return new MultilevelCropAndResize(data, length);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

MultilevelCropAndResize::MultilevelCropAndResize(int pooled_size, const nvinfer1::Dims& imageSize)
    : mPooledSize({pooled_size, pooled_size})
{

    PLUGIN_VALIDATE(pooled_size > 0);
    PLUGIN_VALIDATE(imageSize.nbDims == 3);
    PLUGIN_VALIDATE(imageSize.d[0] > 0 && imageSize.d[1] > 0 && imageSize.d[2] > 0);
    // shape
    mInputHeight = imageSize.d[1];
    mInputWidth = imageSize.d[2];
    // Threshold to P3: Smaller -> P2
    mThresh = (224 * 224) / (4.0f);
}

int MultilevelCropAndResize::getNbOutputs() const noexcept
{
    return 1;
}

int MultilevelCropAndResize::initialize() noexcept
{
    return 0;
}

void MultilevelCropAndResize::terminate() noexcept
{
}

void MultilevelCropAndResize::destroy() noexcept
{
    delete this;
}

size_t MultilevelCropAndResize::getWorkspaceSize(int) const noexcept
{
    return 0;
}

bool MultilevelCropAndResize::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return ((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kLINEAR);
}

const char* MultilevelCropAndResize::getPluginType() const noexcept
{
    return "MultilevelCropAndResize_TRT";
}

const char* MultilevelCropAndResize::getPluginVersion() const noexcept
{
    return "1";
}

IPluginV2Ext* MultilevelCropAndResize::clone() const noexcept
{
    try
    {
        return new MultilevelCropAndResize(*this);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void MultilevelCropAndResize::setPluginNamespace(const char* libNamespace) noexcept
{
    mNameSpace = libNamespace;
}

const char* MultilevelCropAndResize::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

void MultilevelCropAndResize::check_valid_inputs(const nvinfer1::Dims* inputs, int nbInputDims) noexcept
{
    // to be compatible with tensorflow node's input:
    // roi: [N, anchors, 4],
    // feature_map list(5 maps): p2, p3, p4, p5, p6
    PLUGIN_ASSERT(nbInputDims == 1 + mFeatureMapCount);

    nvinfer1::Dims rois = inputs[0];
    PLUGIN_ASSERT(rois.nbDims == 2);
    PLUGIN_ASSERT(rois.d[1] == 4);

    for (int i = 1; i < nbInputDims; ++i)
    {
        nvinfer1::Dims dims = inputs[i];

        // CHW with the same #C
        PLUGIN_ASSERT(dims.nbDims == 3 && dims.d[0] == inputs[1].d[0]);
    }
}

Dims MultilevelCropAndResize::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{

    check_valid_inputs(inputs, nbInputDims);
    PLUGIN_ASSERT(index == 0);

    nvinfer1::Dims result;
    result.nbDims = 4;

    // mROICount
    result.d[0] = inputs[0].d[0];
    // mFeatureLength
    result.d[1] = inputs[1].d[0];
    // height
    result.d[2] = mPooledSize.y;
    // width
    result.d[3] = mPooledSize.x;

    return result;
}

int32_t MultilevelCropAndResize::enqueue(
    int32_t batch_size, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{

    void* pooled = outputs[0];

    cudaError_t status = roiAlignHalfCenter(stream, batch_size, mFeatureLength, mROICount, mThresh,

        mInputHeight, mInputWidth, inputs[0], &inputs[1], mFeatureSpatialSize,

        pooled, mPooledSize, mPrecision);

    PLUGIN_ASSERT(status == cudaSuccess);
    return 0;
}

size_t MultilevelCropAndResize::getSerializationSize() const noexcept
{
    return sizeof(int) * 2 + sizeof(int) * 4 + sizeof(float) + sizeof(int) * 2 * mFeatureMapCount + sizeof(DataType);
}

void MultilevelCropAndResize::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mPooledSize.y);
    write(d, mPooledSize.x);
    write(d, mFeatureLength);
    write(d, mROICount);
    write(d, mInputHeight);
    write(d, mInputWidth);
    write(d, mThresh);
    for (int i = 0; i < mFeatureMapCount; i++)
    {
        write(d, mFeatureSpatialSize[i].y);
        write(d, mFeatureSpatialSize[i].x);
    }
    write(d, mPrecision);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

MultilevelCropAndResize::MultilevelCropAndResize(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mPooledSize = {read<int>(d), read<int>(d)};
    mFeatureLength = read<int>(d);
    mROICount = read<int>(d);
    mInputHeight = read<int>(d);
    mInputWidth = read<int>(d);
    mThresh = read<float>(d);
    for (int i = 0; i < mFeatureMapCount; i++)
    {
        mFeatureSpatialSize[i].y = read<int>(d);
        mFeatureSpatialSize[i].x = read<int>(d);
    }
    mPrecision = read<DataType>(d);

    PLUGIN_VALIDATE(d == a + length);
}

// Return the DataType of the plugin output at the requested index
DataType MultilevelCropAndResize::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // Only DataType::kFLOAT is acceptable by the plugin layer
    // return DataType::kFLOAT;
    // Align output types with the input feature map data types
    if ((inputTypes[1] == DataType::kFLOAT) || (inputTypes[1] == DataType::kHALF))
        return inputTypes[1];

    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool MultilevelCropAndResize::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool MultilevelCropAndResize::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void MultilevelCropAndResize::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims,
    int nbOutputs, const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    PLUGIN_ASSERT(supportsFormat(inputTypes[0], floatFormat));
    check_valid_inputs(inputDims, nbInputs);

    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(nbInputs == 1 + mFeatureMapCount);

    mROICount = inputDims[0].d[0];
    mFeatureLength = inputDims[1].d[0];

    for (size_t layer = 0; layer < mFeatureMapCount; ++layer)
    {
        mFeatureSpatialSize[layer] = {inputDims[layer + 1].d[1], inputDims[layer + 1].d[2]};
    }

    mPrecision = inputTypes[1];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void MultilevelCropAndResize::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void MultilevelCropAndResize::detachFromContext() noexcept
{
}
