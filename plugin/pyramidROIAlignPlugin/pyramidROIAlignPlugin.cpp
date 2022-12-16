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
#include "pyramidROIAlignPlugin.h"
#include "common/plugin.h"
#include <cuda_runtime_api.h>
#include <math.h>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::PyramidROIAlign;
using nvinfer1::plugin::PyramidROIAlignPluginCreator;

namespace
{
char const* PYRAMIDROIALGIN_PLUGIN_VERSION{"1"};
char const* PYRAMIDROIALGIN_PLUGIN_NAME{"PyramidROIAlign_TRT"};
} // namespace

PluginFieldCollection PyramidROIAlignPluginCreator::mFC{};
std::vector<PluginField> PyramidROIAlignPluginCreator::mPluginAttributes;

PyramidROIAlignPluginCreator::PyramidROIAlignPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("fpn_scale", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("pooled_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("image_size", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(PluginField("roi_coords_absolute", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("roi_coords_swap", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("roi_coords_plusone", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("roi_coords_transform", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("sampling_ratio", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("legacy", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* PyramidROIAlignPluginCreator::getPluginName() const noexcept
{
    return PYRAMIDROIALGIN_PLUGIN_NAME;
}

char const* PyramidROIAlignPluginCreator::getPluginVersion() const noexcept
{
    return PYRAMIDROIALGIN_PLUGIN_VERSION;
}

PluginFieldCollection const* PyramidROIAlignPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* PyramidROIAlignPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        // Default values for the plugin creator, these will be used when the corresponding
        // plugin field is not passed, allowing to have defaults for "optional" ONNX attributes.
        int32_t pooledSize = 7;
        int32_t transformCoords = 2;
        bool absCoords = true;
        bool swapCoords = false;
        bool plusOneCoords = false;
        bool legacy = false;
        int32_t samplingRatio = 0;
        xy_t imageSize = {MaskRCNNConfig::IMAGE_SHAPE.d[1], MaskRCNNConfig::IMAGE_SHAPE.d[2]};
        int32_t fpnScale = 224;

        PluginField const* fields = fc->fields;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "fpn_scale"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                fpnScale = *(static_cast<int32_t const*>(fields[i].data));
                PLUGIN_VALIDATE(fpnScale >= 1);
            }
            if (!strcmp(attrName, "pooled_size"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                pooledSize = *(static_cast<int32_t const*>(fields[i].data));
                PLUGIN_VALIDATE(pooledSize >= 1);
            }
            if (!strcmp(attrName, "image_size"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                PLUGIN_VALIDATE(fields[i].length == 2);
                auto const dims = static_cast<int32_t const*>(fields[i].data);
                imageSize.y = dims[0];
                imageSize.x = dims[1];
                PLUGIN_VALIDATE(imageSize.y >= 1);
                PLUGIN_VALIDATE(imageSize.x >= 1);
            }
            if (!strcmp(attrName, "roi_coords_absolute"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                absCoords = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "roi_coords_swap"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                swapCoords = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "roi_coords_plusone"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                plusOneCoords = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "roi_coords_transform"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                transformCoords = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "sampling_ratio"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                samplingRatio = *(static_cast<int32_t const*>(fields[i].data));
                PLUGIN_VALIDATE(samplingRatio >= 0);
            }            
            if (!strcmp(attrName, "legacy"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                legacy = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "legacy"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                legacy = *(static_cast<int32_t const*>(fields[i].data));
            }
        }
        return new PyramidROIAlign(
            pooledSize, transformCoords, absCoords, swapCoords, plusOneCoords, samplingRatio, legacy, imageSize, fpnScale);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* PyramidROIAlignPluginCreator::deserializePlugin(
    char const* name, void const* data, size_t length) noexcept
{
    try
    {
        return new PyramidROIAlign(data, length);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

PyramidROIAlign::PyramidROIAlign(int32_t pooledSize, int32_t transformCoords, bool absCoords, bool swapCoords,
    bool plusOneCoords, int32_t samplingRatio, bool legacy, xy_t imageSize, int32_t fpnScale)
    : mPooledSize({pooledSize, pooledSize})
    , mImageSize(imageSize)
    , mFPNScale(fpnScale)
    , mTransformCoords(transformCoords)
    , mAbsCoords(absCoords)
    , mSwapCoords(swapCoords)
    , mPlusOneCoords(plusOneCoords)
    , mSamplingRatio(samplingRatio)
    , mIsLegacy(legacy)
{
    PLUGIN_VALIDATE(pooledSize >= 1);
    PLUGIN_VALIDATE(samplingRatio >= 0);
    PLUGIN_VALIDATE(fpnScale >= 1);
}

int32_t PyramidROIAlign::getNbOutputs() const noexcept
{
    return 1;
}

int32_t PyramidROIAlign::initialize() noexcept
{
    return 0;
}

void PyramidROIAlign::terminate() noexcept {}

void PyramidROIAlign::destroy() noexcept
{
    delete this;
}

size_t PyramidROIAlign::getWorkspaceSize(int) const noexcept
{
    return 0;
}

bool PyramidROIAlign::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

char const* PyramidROIAlign::getPluginType() const noexcept
{
    return PYRAMIDROIALGIN_PLUGIN_NAME;
}

char const* PyramidROIAlign::getPluginVersion() const noexcept
{
    return PYRAMIDROIALGIN_PLUGIN_VERSION;
}

IPluginV2Ext* PyramidROIAlign::clone() const noexcept
{
    try
    {
        auto plugin = new PyramidROIAlign(*this);
        plugin->setPluginNamespace(mNameSpace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void PyramidROIAlign::setPluginNamespace(char const* libNamespace) noexcept
{
    mNameSpace = libNamespace;
}

char const* PyramidROIAlign::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

void PyramidROIAlign::check_valid_inputs(nvinfer1::Dims const* inputs, int32_t nbInputDims)
{
    // to be compatible with tensorflow node's input:
    // roi: [N, anchors, 4],
    // feature_map list(4 maps): p2, p3, p4, p5
    PLUGIN_ASSERT(nbInputDims == 1 + mFeatureMapCount);

    nvinfer1::Dims rois = inputs[0];
    PLUGIN_ASSERT(rois.nbDims == 2);
    PLUGIN_ASSERT(rois.d[1] == 4);

    for (int32_t i = 1; i < nbInputDims; ++i)
    {
        nvinfer1::Dims dims = inputs[i];

        // CHW with the same #C
        PLUGIN_ASSERT(dims.nbDims == 3 && dims.d[0] == inputs[i].d[0]);
    }
}

Dims PyramidROIAlign::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept
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

int32_t PyramidROIAlign::enqueue(
    int32_t batch_size, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept

{
    void* const pooled = outputs[0];
    cudaError_t status;

    // Support legacy UFF mode
    if (mIsLegacy)
    {
        // Legacy values
        mTransformCoords = -1;
        mPlusOneCoords = 0;
        mSwapCoords = true;
        mAbsCoords = false;
        mSamplingRatio = 1;
        float const firstThreshold = (224 * 224 * 2.0f / (MaskRCNNConfig::IMAGE_SHAPE.d[1] * MaskRCNNConfig::IMAGE_SHAPE.d[2])) / (4.0 * 4.0f);
        status = roiAlign(stream, batch_size, mImageSize, mFeatureLength, mROICount, firstThreshold,
            mTransformCoords, mAbsCoords, mSwapCoords, mPlusOneCoords, mSamplingRatio, inputs[0], &inputs[1],
            mFeatureSpatialSize, pooled, mPooledSize);
    }
    else
    {
        // As per FPN paper equation 1 (https://arxiv.org/pdf/1612.03144.pdf)
        // the default 224 FPN scale corresponds to the canonical ImageNet size
        // used to define the ROI scale threshold that samples from P4. Because the
        // plugin works with normalized ROI coordinates, the FPN scale must be normalized
        // by the input image size.
        float const scale = static_cast<float>(mFPNScale);
        float const normScale = sqrtf(scale * scale / (mImageSize.y * mImageSize.x));
        // Furthermore, the roiAlign kernel expects a first threshold instead. This is
        // the *area* of an ROI but for one level down, i.e. at the P2->P3 transition.
        float const firstThreshold = normScale * normScale / 4.F;
        status = roiAlign(stream, batch_size, mImageSize, mFeatureLength, mROICount, firstThreshold,
            mTransformCoords, mAbsCoords, mSwapCoords, mPlusOneCoords, mSamplingRatio, inputs[0], &inputs[1],
            mFeatureSpatialSize, pooled, mPooledSize);
    }
    return status;
}

size_t PyramidROIAlign::getSerializationSize() const noexcept
{
    return sizeof(int) * 2 // mPooledSize
        + sizeof(int) * 2  // mImageSize
        + sizeof(int)      // mFeatureLength
        + sizeof(int)      // mROICount
        + sizeof(int)      // mFPNScale
        + sizeof(int)      // mTransformCoords
        + sizeof(bool)     // mAbsCoords
        + sizeof(bool)     // mSwapCoords
        + sizeof(bool)     // mPlusOneCoords
        + sizeof(int)      // mSamplingRatio
        + sizeof(bool)     // mIsLegacy
        + sizeof(int) * 8; // mFeatureSpatialSize
}

void PyramidROIAlign::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mPooledSize.y);
    write(d, mPooledSize.x);
    write(d, mImageSize.y);
    write(d, mImageSize.x);
    write(d, mFeatureLength);
    write(d, mROICount);
    write(d, mFPNScale);
    write(d, mTransformCoords);
    write(d, mAbsCoords);
    write(d, mSwapCoords);
    write(d, mPlusOneCoords);
    write(d, mSamplingRatio);
    write(d, mIsLegacy);
    write(d, mFeatureSpatialSize[0].y);
    write(d, mFeatureSpatialSize[0].x);
    write(d, mFeatureSpatialSize[1].y);
    write(d, mFeatureSpatialSize[1].x);
    write(d, mFeatureSpatialSize[2].y);
    write(d, mFeatureSpatialSize[2].x);
    write(d, mFeatureSpatialSize[3].y);
    write(d, mFeatureSpatialSize[3].x);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

PyramidROIAlign::PyramidROIAlign(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    mPooledSize = {read<int>(d), read<int>(d)};
    mImageSize = {read<int>(d), read<int>(d)};
    mFeatureLength = read<int>(d);
    mROICount = read<int>(d);
    mFPNScale = read<int>(d);
    mTransformCoords = read<int>(d);
    mAbsCoords = read<bool>(d);
    mSwapCoords = read<bool>(d);
    mPlusOneCoords = read<bool>(d);
    mSamplingRatio = read<int>(d);
    mIsLegacy = read<bool>(d);
    mFeatureSpatialSize[0].y = read<int>(d);
    mFeatureSpatialSize[0].x = read<int>(d);
    mFeatureSpatialSize[1].y = read<int>(d);
    mFeatureSpatialSize[1].x = read<int>(d);
    mFeatureSpatialSize[2].y = read<int>(d);
    mFeatureSpatialSize[2].x = read<int>(d);
    mFeatureSpatialSize[3].y = read<int>(d);
    mFeatureSpatialSize[3].x = read<int>(d);

    PLUGIN_VALIDATE(d == a + length);
}

// Return the DataType of the plugin output at the requested index
DataType PyramidROIAlign::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool PyramidROIAlign::isOutputBroadcastAcrossBatch(
    int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool PyramidROIAlign::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void PyramidROIAlign::configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims,
    int32_t nbOutputs, DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
    bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept
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
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void PyramidROIAlign::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void PyramidROIAlign::detachFromContext() noexcept {}
