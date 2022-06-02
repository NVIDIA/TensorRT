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
#include "roiAlignPlugin.h"

#include <assert.h>

#include "serialize.hpp"
#include <chrono>
#include <math.h>

using namespace nvinfer1::plugin;
using nvinfer1::plugin::RoIAlignPluginDynamic;

extern void TRTRoIAlignForwardCUDAKernelLauncher_float(const float* input, const float* rois, int roi_cols,
    const int* batch_indices, float* output, int output_size, int channels, int height, int width, int aligned_height,
    int aligned_width, float spatial_scale, int sampling_ratio, int pool_mode, int coord_transform_mode,
    cudaStream_t stream);

namespace
{
const char* ROIALIGN_PLUGIN_VERSION{"1"};
const char* ROIALIGN_PLUGIN_NAME{"RoiAlign"};
} // namespace

nvinfer1::PluginFieldCollection RoIAlignPluginDynamicCreator::mFC{};
std::vector<nvinfer1::PluginField> RoIAlignPluginDynamicCreator::mPluginAttributes;
REGISTER_TENSORRT_PLUGIN(RoIAlignPluginDynamicCreator);

RoIAlignPluginDynamic::RoIAlignPluginDynamic(
    int coordTransformMode, int poolingMode, int outputHeight, int outputWidth, int samplingRatio, float spatialScale)
    : mCoordTransformMode(coordTransformMode)
    , mPoolingMode(poolingMode)
    , mOutputHeight(outputHeight)
    , mOutputWidth(outputWidth)
    , mSamplingRatio(samplingRatio)
    , mSpatialScale(spatialScale)
{
}

RoIAlignPluginDynamic::RoIAlignPluginDynamic(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    deserialize_value(&data, &length, &mCoordTransformMode);
    deserialize_value(&data, &length, &mPoolingMode);
    deserialize_value(&data, &length, &mOutputHeight);
    deserialize_value(&data, &length, &mOutputWidth);
    deserialize_value(&data, &length, &mSamplingRatio);
    deserialize_value(&data, &length, &mSpatialScale);

    ASSERT(d == a + length);
}

nvinfer1::IPluginV2DynamicExt* RoIAlignPluginDynamic::clone() const noexcept
{
    RoIAlignPluginDynamic* plugin = new RoIAlignPluginDynamic(
        mCoordTransformMode, mPoolingMode, mOutputHeight, mOutputWidth, mSamplingRatio, mSpatialScale);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs RoIAlignPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    nvinfer1::DimsExprs ret;
    ret.nbDims = 4;
    ret.d[0] = inputs[1].d[0];
    ret.d[1] = inputs[0].d[1];
    ret.d[2] = exprBuilder.constant(mOutputHeight);
    ret.d[3] = exprBuilder.constant(mOutputWidth);

    return ret;
}

bool RoIAlignPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    ASSERT(0 <= pos && pos < 4);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    switch (pos)
    {
    case 0:
        // X
        return in[0].type == nvinfer1::DataType::kFLOAT && in[0].format == nvinfer1::TensorFormat::kLINEAR;
        break;
    case 1:
        // rois
        return in[1].type == nvinfer1::DataType::kFLOAT && in[1].format == nvinfer1::TensorFormat::kLINEAR;
    case 2:
        // batch_indices
        return in[2].type == nvinfer1::DataType::kINT32 && in[2].format == nvinfer1::TensorFormat::kLINEAR;
    case 3:
        // Y
        return out[0].type == nvinfer1::DataType::kFLOAT && out[0].format == nvinfer1::TensorFormat::kLINEAR;
    }
}

void RoIAlignPluginDynamic::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs, int nbOutputs) noexcept
{
}

size_t RoIAlignPluginDynamic::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int RoIAlignPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workSpace,
    cudaStream_t stream) noexcept
{
    int channels = inputDesc[0].dims.d[1];
    int height = inputDesc[0].dims.d[2];
    int width = inputDesc[0].dims.d[3];

    int batch = inputDesc[0].dims.d[0];

    int output_size
        = outputDesc[0].dims.d[0] * outputDesc[0].dims.d[1] * outputDesc[0].dims.d[2] * outputDesc[0].dims.d[3];

    const void* feat = inputs[0];
    const void* rois = inputs[1];
    const void* batch_indices = inputs[2];
    int num_rois = inputDesc[1].dims.d[0];
    int roi_cols = inputDesc[1].dims.d[1];
    void* output = outputs[0];

    switch (outputDesc[0].type)
    {
    case nvinfer1::DataType::kFLOAT:
        TRTRoIAlignForwardCUDAKernelLauncher_float((const float*) feat, (const float*) rois, roi_cols,
            (const int*) batch_indices, (float*) output, output_size, channels, height, width, mOutputHeight,
            mOutputWidth, mSpatialScale, mSamplingRatio, mPoolingMode, mCoordTransformMode, stream);
        break;

    default: break;
    }

    return 0;
}

nvinfer1::DataType RoIAlignPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

// IPluginV2 Methods
const char* RoIAlignPluginDynamic::getPluginType() const noexcept
{
    return ROIALIGN_PLUGIN_NAME;
}

const char* RoIAlignPluginDynamic::getPluginVersion() const noexcept
{
    return ROIALIGN_PLUGIN_VERSION;
}

int RoIAlignPluginDynamic::getNbOutputs() const noexcept
{
    return 1;
}

int RoIAlignPluginDynamic::initialize() noexcept
{
    return 0;
}

void RoIAlignPluginDynamic::terminate() noexcept {}

size_t RoIAlignPluginDynamic::getSerializationSize() const noexcept
{
    return sizeof(mCoordTransformMode) + sizeof(mPoolingMode) + sizeof(mOutputHeight) + sizeof(mOutputWidth)
        + sizeof(mSamplingRatio) + sizeof(mSpatialScale);
}

void RoIAlignPluginDynamic::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mCoordTransformMode);
    serialize_value(&buffer, mPoolingMode);
    serialize_value(&buffer, mOutputHeight);
    serialize_value(&buffer, mOutputWidth);
    serialize_value(&buffer, mSamplingRatio);
    serialize_value(&buffer, mSpatialScale);
}

void RoIAlignPluginDynamic::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void RoIAlignPluginDynamic::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* RoIAlignPluginDynamic::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

RoIAlignPluginDynamicCreator::RoIAlignPluginDynamicCreator()
{
    mPluginAttributes.emplace_back(nvinfer1::PluginField("coordinate_transformation_mode"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("mode"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("output_height"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("output_width"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("sampling_ratio"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("spatial_scale"));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* RoIAlignPluginDynamicCreator::getPluginName() const noexcept
{
    return ROIALIGN_PLUGIN_NAME;
}

const char* RoIAlignPluginDynamicCreator::getPluginVersion() const noexcept
{
    return ROIALIGN_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* RoIAlignPluginDynamicCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV2* RoIAlignPluginDynamicCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept
{
    int coordTransformMode = static_cast<int>(RoIAlignPluginDynamic::CoordinateTransformationMode::HALF_PIXEL);
    int poolingMode = static_cast<int>(RoIAlignPluginDynamic::PoolingMode::AVG);
    int outputHeight = 1;
    int outputWidth = 1;
    int samplingRatio = 0;
    float spatialScale = 1.;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; i++)
    {
        if (fc->fields[i].data == nullptr)
        {
            continue;
        }
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("coordinate_transformation_mode") == 0)
        {
            int data_size = fields[i].length;
            const char* data_start = static_cast<const char*>(fields[i].data);
            ASSERT(fields[i].type == PluginFieldType::kCHAR);
            std::string modeStr(data_start, data_size);
            if (modeStr == "output_half_pixel")
            {
                coordTransformMode
                    = static_cast<int>(RoIAlignPluginDynamic::CoordinateTransformationMode::OUTPUT_HALF_PIXEL);
            }
            else if (modeStr == "half_pixel")
            {
                coordTransformMode = static_cast<int>(RoIAlignPluginDynamic::CoordinateTransformationMode::HALF_PIXEL);
            }
        }

        if (field_name.compare("mode") == 0)
        {
            int data_size = fc->fields[i].length;
            const char* data_start = static_cast<const char*>(fields[i].data);
            ASSERT(fields[i].type == PluginFieldType::kCHAR);
            std::string modeStr(data_start, data_size);
            if (modeStr.compare("avg") == 0)
            {
                poolingMode = static_cast<int>(RoIAlignPluginDynamic::PoolingMode::AVG);
            }
            else if (modeStr.compare("max") == 0)
            {
                poolingMode = static_cast<int>(RoIAlignPluginDynamic::PoolingMode::MAX);
            }
        }

        if (field_name.compare("output_height") == 0)
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32)
            outputHeight = static_cast<const int*>(fc->fields[i].data)[0];
        }

        if (field_name.compare("output_width") == 0)
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32)
            outputWidth = static_cast<const int*>(fc->fields[i].data)[0];
        }

        if (field_name.compare("spatial_scale") == 0)
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32)
            spatialScale = static_cast<const float*>(fc->fields[i].data)[0];
        }

        if (field_name.compare("sampling_ratio") == 0)
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32)
            samplingRatio = static_cast<const int*>(fc->fields[i].data)[0];
        }
    }

    ASSERT(outputHeight > 0);
    ASSERT(outputWidth > 0);
    ASSERT(spatialScale > 0.);
    ASSERT(samplingRatio >= 0);

    RoIAlignPluginDynamic* plugin = new RoIAlignPluginDynamic(
        coordTransformMode, poolingMode, outputHeight, outputWidth, samplingRatio, spatialScale);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

nvinfer1::IPluginV2* RoIAlignPluginDynamicCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    auto plugin = new RoIAlignPluginDynamic(serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void RoIAlignPluginDynamicCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* RoIAlignPluginDynamicCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
