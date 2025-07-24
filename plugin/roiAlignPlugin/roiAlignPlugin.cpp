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
#include "roiAlignPlugin.h"
#include "roiAlignKernel.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::ROIAlignV3;
using nvinfer1::plugin::ROIAlignV3PluginCreator;

namespace
{
char const* gRoialignPluginVersion{"2"};
char const* gRoialignPluginName{"ROIAlign_TRT"};
} // namespace

ROIAlignV3PluginCreator::ROIAlignV3PluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> guard(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("coordinate_transformation_mode", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("mode", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("output_height", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("output_width", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("sampling_ratio", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("spatial_scale", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* ROIAlignV3PluginCreator::getPluginName() const noexcept
{
    return gRoialignPluginName;
}

char const* ROIAlignV3PluginCreator::getPluginVersion() const noexcept
{
    return gRoialignPluginVersion;
}

PluginFieldCollection const* ROIAlignV3PluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV3* ROIAlignV3PluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const* fields = fc->fields;

        // default values
        int32_t outputHeight = 1;
        int32_t outputWidth = 1;
        int32_t samplingRatio = 0;
        int32_t mode = 1;
        int32_t aligned = 1;
        float spatialScale = 1.0F;

        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "output_height"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                outputHeight = static_cast<int32_t>(*(static_cast<int32_t const*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "output_width"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                outputWidth = static_cast<int32_t>(*(static_cast<int32_t const*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "sampling_ratio"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                samplingRatio = static_cast<int32_t>(*(static_cast<int32_t const*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "mode"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                mode = static_cast<int32_t>(*(static_cast<int32_t const*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "spatial_scale"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                spatialScale = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "coordinate_transformation_mode"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                aligned = static_cast<int32_t>(*(static_cast<int32_t const*>(fields[i].data)));
            }
        }
        return new ROIAlignV3(outputHeight, outputWidth, samplingRatio, mode, spatialScale, aligned);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void ROIAlignV3PluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* ROIAlignV3PluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

ROIAlignV3::ROIAlignV3(
    int32_t outputHeight, int32_t outputWidth, int32_t samplingRatio, int32_t mode, float spatialScale, int32_t aligned)
    : mOutputHeight(outputHeight)
    , mOutputWidth(outputWidth)
    , mSamplingRatio(samplingRatio)
    , mSpatialScale(spatialScale)
    , mMode(mode)
    , mAligned(aligned)
{
    PLUGIN_VALIDATE(outputHeight > 0);
    PLUGIN_VALIDATE(outputWidth > 0);
    PLUGIN_VALIDATE(samplingRatio >= 0);
    PLUGIN_VALIDATE(mode == 0 || mode == 1);
    PLUGIN_VALIDATE(spatialScale > 0.0F);
    PLUGIN_VALIDATE(aligned == 0 || aligned == 1);

    int32_t device;
    PLUGIN_CUASSERT(cudaGetDevice(&device));
    cudaDeviceProp props;
    PLUGIN_CUASSERT(cudaGetDeviceProperties(&props, device));

    mMaxThreadsPerBlock = props.maxThreadsPerBlock;
}

IPluginCapability* ROIAlignV3::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV3* ROIAlignV3::clone() noexcept
{
    try
    {
        auto plugin = std::make_unique<ROIAlignV3>(*this);
        return plugin.release();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

char const* ROIAlignV3::getPluginName() const noexcept
{
    return gRoialignPluginName;
}

char const* ROIAlignV3::getPluginVersion() const noexcept
{
    return gRoialignPluginVersion;
}

char const* ROIAlignV3::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

int32_t ROIAlignV3::getNbOutputs() const noexcept
{
    return 1;
}

int32_t ROIAlignV3::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return 0;
}

bool ROIAlignV3::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(inOut != nullptr);
    PLUGIN_ASSERT(pos >= 0 && pos <= 3);
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(nbOutputs == 1);

    PluginTensorDesc const& desc = inOut[pos].desc;
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    // first input should be float16 or float32
    if (pos == 0)
    {
        return (desc.type == nvinfer1::DataType::kFLOAT || desc.type == nvinfer1::DataType::kHALF);
    }

    // batch_indices always has to be int32
    if (pos == 2)
    {
        return (desc.type == nvinfer1::DataType::kINT32);
    }

    // rois and the output should have the same type as the first input
    return (desc.type == inOut[0].desc.type);
}

int32_t ROIAlignV3::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    PLUGIN_ASSERT(inputTypes != nullptr);
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(nbOutputs == 1);
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t ROIAlignV3::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(inputs != nullptr);
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(nbOutputs == 1);

    outputs[0].nbDims = 4;

    // mROICount
    outputs[0].d[0] = inputs[1].d[0];
    // mFeatureLength
    outputs[0].d[1] = inputs[0].d[1];
    // height
    auto const* height = exprBuilder.constant(mOutputHeight);
    PLUGIN_ASSERT(height != nullptr);
    outputs[0].d[2] = height;
    // width
    auto const* width = exprBuilder.constant(mOutputWidth);
    PLUGIN_ASSERT(width != nullptr);
    outputs[0].d[3] = width;

    return 0;
}

int32_t ROIAlignV3::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    PLUGIN_VALIDATE(inputDesc != nullptr && inputs != nullptr && outputs != nullptr);

    // No-op pass-through for empty ROIs
    if (mROICount == 0)
    {
        return 0;
    }

    auto type = inputDesc[0].type;

    PLUGIN_ASSERT(type == nvinfer1::DataType::kHALF || type == nvinfer1::DataType::kFLOAT);

    switch (type)
    {
    case nvinfer1::DataType::kFLOAT:
    {
        auto bottomData = static_cast<float const*>(inputs[0]);
        auto bottomRois = static_cast<float const*>(inputs[1]);
        auto batchIndicesPtr = static_cast<int32_t const*>(inputs[2]);
        auto topData = static_cast<float*>(outputs[0]);

        return RoiAlignImpl<float>(stream, mMaxThreadsPerBlock, bottomData, mSpatialScale, mROICount, mFeatureLength,
            mHeight, mWidth, mOutputHeight, mOutputWidth, mSamplingRatio, bottomRois, topData, mMode, batchIndicesPtr,
            mAligned);
    }
    break;
    case nvinfer1::DataType::kHALF:
    {
        auto bottomData = static_cast<__half const*>(inputs[0]);
        auto bottomRois = static_cast<__half const*>(inputs[1]);
        auto batchIndicesPtr = static_cast<int32_t const*>(inputs[2]);
        auto topData = static_cast<__half*>(outputs[0]);

        return RoiAlignImpl<__half>(stream, mMaxThreadsPerBlock, bottomData, mSpatialScale, mROICount, mFeatureLength,
            mHeight, mWidth, mOutputHeight, mOutputWidth, mSamplingRatio, bottomRois, topData, mMode, batchIndicesPtr,
            mAligned);
    }
    break;
    default: return -1;
    }

    return 0;
}

int32_t ROIAlignV3::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(in != nullptr);
    PLUGIN_ASSERT(out != nullptr);
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(nbInputs == 3);

    nvinfer1::Dims rois = in[1].dims;
    nvinfer1::Dims batchIndices = in[2].dims;

    PLUGIN_ASSERT(rois.nbDims == 2);
    PLUGIN_ASSERT(rois.d[1] == 4);

    PLUGIN_ASSERT(batchIndices.nbDims == 1);
    // Check batch_indices matches rois in length
    PLUGIN_ASSERT(rois.d[0] == batchIndices.d[0]);

    mFeatureLength = in[0].dims.d[1];
    mHeight = in[0].dims.d[2];
    mWidth = in[0].dims.d[3];

    mROICount = in[1].dims.d[0];
    return 0;
}

IPluginV3* ROIAlignV3::attachToContext(IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* ROIAlignV3::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("coordinate_transformation_mode", &mAligned, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("mode", &mMode, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("output_height", &mOutputHeight, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("output_width", &mOutputWidth, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("sampling_ratio", &mSamplingRatio, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("spatial_scale", &mSpatialScale, PluginFieldType::kFLOAT32, 1);
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

size_t ROIAlignV3::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

void ROIAlignV3::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_ASSERT(libNamespace != nullptr);
        mNameSpace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}
