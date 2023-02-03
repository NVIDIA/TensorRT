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
#include "roiAlignKernel.h"
#include "roiAlignPlugin.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::ROIAlign;
using nvinfer1::plugin::ROIAlignPluginCreator;

namespace
{
char const* kROIALIGN_PLUGIN_VERSION{"1"};
char const* kROIALIGN_PLUGIN_NAME{"ROIAlign_TRT"};
size_t constexpr kSERIALIZATION_SIZE{sizeof(int32_t) * 5 + sizeof(float) + sizeof(int32_t) * 4};
} // namespace

PluginFieldCollection ROIAlignPluginCreator::mFC{};
std::vector<PluginField> ROIAlignPluginCreator::mPluginAttributes;

ROIAlignPluginCreator::ROIAlignPluginCreator()
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> guard(mutex);
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

char const* ROIAlignPluginCreator::getPluginName() const noexcept
{
    return kROIALIGN_PLUGIN_NAME;
}

char const* ROIAlignPluginCreator::getPluginVersion() const noexcept
{
    return kROIALIGN_PLUGIN_VERSION;
}

PluginFieldCollection const* ROIAlignPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* ROIAlignPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
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
        return new ROIAlign(outputHeight, outputWidth, samplingRatio, mode, spatialScale, aligned);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* ROIAlignPluginCreator::deserializePlugin(
    char const* name, void const* data, size_t length) noexcept
{
    try
    {
        PLUGIN_VALIDATE(data != nullptr);
        return new ROIAlign(data, length);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

int32_t ROIAlign::getNbOutputs() const noexcept
{
    return 1;
}

int32_t ROIAlign::initialize() noexcept
{
    int32_t device;
    PLUGIN_CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp props;
    PLUGIN_CHECK_CUDA(cudaGetDeviceProperties(&props, device));

    mMaxThreadsPerBlock = props.maxThreadsPerBlock;

    return 0;
}

void ROIAlign::terminate() noexcept {}

void ROIAlign::destroy() noexcept
{
    delete this;
}

size_t ROIAlign::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

bool ROIAlign::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(inOut != nullptr);
    PLUGIN_ASSERT(pos >= 0 && pos <= 3);
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(nbOutputs == 1);

    PluginTensorDesc const& desc = inOut[pos];
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    // first input should be float16 or float32
    if(pos == 0)
    {
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF);
    }

    // batch_indices always has to be int32
    if(pos == 2)
    {
        return (inOut[pos].type == nvinfer1::DataType::kINT32);
    }

    // rois and the output should have the same type as the first input
    return (inOut[pos].type == inOut[0].type);
}

char const* ROIAlign::getPluginType() const noexcept
{
    return kROIALIGN_PLUGIN_NAME;
}

char const* ROIAlign::getPluginVersion() const noexcept
{
    return kROIALIGN_PLUGIN_VERSION;
}

IPluginV2DynamicExt* ROIAlign::clone() const noexcept
{
    try
    {
        auto plugin = new ROIAlign(*this);
        plugin->setPluginNamespace(mNameSpace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void ROIAlign::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_ASSERT(libNamespace != nullptr);
        mNameSpace = libNamespace;
    }
    catch (std::exception const& e)
    {
        gLogError << e.what() << std::endl;
    }
}

char const* ROIAlign::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

void ROIAlign::checkValidInputs(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputDims)
{
    PLUGIN_ASSERT(inputs != nullptr);
    PLUGIN_ASSERT(nbInputDims == 3);

    nvinfer1::Dims rois = inputs[1].desc.dims;
    nvinfer1::Dims batchIndices = inputs[2].desc.dims;

    PLUGIN_ASSERT(rois.nbDims == 2);
    PLUGIN_ASSERT(rois.d[1] == 4);

    PLUGIN_ASSERT(batchIndices.nbDims == 1);
    // Check batch_indices matches rois in length
    PLUGIN_ASSERT(rois.d[0] == batchIndices.d[0]);
}

void ROIAlign::validateAttributes(int32_t outputHeight, int32_t outputWidth, int32_t samplingRatio, int32_t mode, float spatialScale, int32_t aligned)
{
    PLUGIN_VALIDATE(outputHeight > 0);
    PLUGIN_VALIDATE(outputWidth > 0);
    PLUGIN_VALIDATE(samplingRatio >= 0);
    PLUGIN_VALIDATE(mode == 0 || mode == 1);
    PLUGIN_VALIDATE(spatialScale > 0.0F);
    PLUGIN_VALIDATE(aligned == 0 || aligned == 1);
}

DimsExprs ROIAlign::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(inputs != nullptr);
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(outputIndex == 0); // there is only one output

    nvinfer1::DimsExprs result;
    result.nbDims = 4;

    // mROICount
    result.d[0] = inputs[1].d[0];
    // mFeatureLength
    result.d[1] = inputs[0].d[1];
    // height
    auto const* height = exprBuilder.constant(mOutputHeight);
    PLUGIN_ASSERT(height != nullptr);
    result.d[2] = height;
    // width
    auto const* width = exprBuilder.constant(mOutputWidth);
    PLUGIN_ASSERT(width != nullptr);
    result.d[3] = width;

    return result;
}

int32_t ROIAlign::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    PLUGIN_ASSERT(inputDesc != nullptr);
    PLUGIN_ASSERT(inputs != nullptr);
    PLUGIN_ASSERT(outputs != nullptr);
    PLUGIN_ASSERT(outputDesc != nullptr);

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
            mHeight, mWidth, mOutputHeight, mOutputWidth, mSamplingRatio, bottomRois, topData, mMode, batchIndicesPtr, mAligned);
    }
    break;
    case nvinfer1::DataType::kHALF:
    {
        auto bottomData = static_cast<__half const*>(inputs[0]);
        auto bottomRois = static_cast<__half const*>(inputs[1]);
        auto batchIndicesPtr = static_cast<int32_t const*>(inputs[2]);
        auto topData = static_cast<__half*>(outputs[0]);

        return RoiAlignImpl<__half>(stream, mMaxThreadsPerBlock, bottomData, mSpatialScale, mROICount,
            mFeatureLength, mHeight, mWidth, mOutputHeight, mOutputWidth, mSamplingRatio, bottomRois, topData, mMode, batchIndicesPtr, mAligned);
    }
    break;
    default:
        return -1;
    }

    return 0;
}

size_t ROIAlign::getSerializationSize() const noexcept
{
    return kSERIALIZATION_SIZE;
}

void ROIAlign::serialize(void* buffer) const noexcept
{
    PLUGIN_VALIDATE(buffer != nullptr);
    char *d = static_cast<char*>(buffer);
    char *a = d;
    write(d, mAligned);       // int32_t
    write(d, mMode);          // int32_t
    write(d, mOutputHeight);  // int32_t
    write(d, mOutputWidth);   // int32_t
    write(d, mSamplingRatio); // int32_t
    write(d, mSpatialScale);  // float

    write(d, mROICount);      // int32_t
    write(d, mFeatureLength); // int32_t
    write(d, mHeight);        // int32_t
    write(d, mWidth);         // int32_t
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

ROIAlign::ROIAlign(
    int32_t outputHeight, int32_t outputWidth, int32_t samplingRatio, int32_t mode, float spatialScale, int32_t aligned)
    : mOutputHeight(outputHeight)
    , mOutputWidth(outputWidth)
    , mSamplingRatio(samplingRatio)
    , mSpatialScale(spatialScale)
    , mMode(mode)
    , mAligned(aligned)
{
    validateAttributes(mOutputHeight, mOutputWidth, mSamplingRatio, mMode, mSpatialScale, mAligned);
}

ROIAlign::ROIAlign(void const* data, size_t length)
{
    PLUGIN_VALIDATE(data != nullptr);
    PLUGIN_VALIDATE(length == kSERIALIZATION_SIZE);

    char const *d = static_cast<char const*>(data);
    char const *a = d;

    mAligned = read<int32_t>(d);
    mMode = read<int32_t>(d);
    mOutputHeight = read<int32_t>(d);
    mOutputWidth = read<int32_t>(d);
    mSamplingRatio = read<int32_t>(d);
    mSpatialScale = read<float>(d);

    mROICount = read<int32_t>(d);
    mFeatureLength = read<int32_t>(d);
    mHeight = read<int32_t>(d);
    mWidth = read<int32_t>(d);

    PLUGIN_VALIDATE(d == a + length);
    validateAttributes(mOutputHeight, mOutputWidth, mSamplingRatio, mMode, mSpatialScale, mAligned);
}

DataType ROIAlign::getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const
    noexcept
{
    PLUGIN_ASSERT(inputTypes != nullptr);
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(index == 0);
    return inputTypes[0];
}

void ROIAlign::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(in != nullptr);
    PLUGIN_ASSERT(out != nullptr);
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(nbInputs == 3);

    checkValidInputs(in, nbInputs);

    mFeatureLength = in[0].desc.dims.d[1];
    mHeight = in[0].desc.dims.d[2];
    mWidth = in[0].desc.dims.d[3];

    mROICount = in[1].desc.dims.d[0];
}
