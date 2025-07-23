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

#include "decodeBbox3D.h"
#include "common/bboxUtils.h"
#include "common/checkMacrosPlugin.h"
#include "common/kernels/kernel.h"
#include "common/templates.h"

namespace nvinfer1::plugin
{

using nvinfer1::plugin::DecodeBbox3DPlugin;
using nvinfer1::plugin::DecodeBbox3DPluginCreator;

namespace
{
static char const* const kPLUGIN_VERSION{"1"};
static char const* const kPLUGIN_NAME{"DecodeBbox3DPlugin"};
} // namespace

DecodeBbox3DPlugin::DecodeBbox3DPlugin(float xMin, float xMax, float yMin, float yMax, float zMin, float zMax,
    int32_t numDirBins, float dirOffset, float dirLimitOffset, std::vector<float> const& anchorBottomHeight,
    std::vector<float> const& anchors, float scoreThreshold)
    : mMinXRange(xMin)
    , mMaxXRange(xMax)
    , mMinYRange(yMin)
    , mMaxYRange(yMax)
    , mMinZRange(zMin)
    , mMaxZRange(zMax)
    , mNumDirBins(numDirBins)
    , mDirOffset(dirOffset)
    , mDirLimitOffset(dirLimitOffset)
    , mScoreThreashold(scoreThreshold)
{
    mAnchorBottomHeight = anchorBottomHeight;
    mAnchors = anchors;
    mNumClasses = static_cast<int32_t>(mAnchorBottomHeight.size());
    PLUGIN_VALIDATE(static_cast<size_t>(mNumClasses) * 2 * 4 == mAnchors.size());
}

DecodeBbox3DPlugin::DecodeBbox3DPlugin(float xMin, float xMax, float yMin, float yMax, float zMin, float zMax,
    int32_t numDirBins, float dirOffset, float dirLimitOffset, std::vector<float> const& anchorBottomHeight,
    std::vector<float> const& anchors, float scoreThreshold, int32_t feature_h, int32_t feature_w)
    : mMinXRange(xMin)
    , mMaxXRange(xMax)
    , mMinYRange(yMin)
    , mMaxYRange(yMax)
    , mMinZRange(zMin)
    , mMaxZRange(zMax)
    , mNumDirBins(numDirBins)
    , mDirOffset(dirOffset)
    , mDirLimitOffset(dirLimitOffset)
    , mScoreThreashold(scoreThreshold)
    , mFeatureH(feature_h)
    , mFeatureW(feature_w)
{
    mAnchorBottomHeight = anchorBottomHeight;
    mAnchors = anchors;
    mNumClasses = static_cast<int32_t>(mAnchorBottomHeight.size());
    PLUGIN_VALIDATE(static_cast<size_t>(mNumClasses) * 2 * 4 == mAnchors.size());
}

DecodeBbox3DPlugin::DecodeBbox3DPlugin(void const* data, size_t length)
{
    PLUGIN_VALIDATE(data != nullptr);
    auto const* d = reinterpret_cast<uint8_t const*>(data);
    mMinXRange = readFromBuffer<float>(d);
    mMaxXRange = readFromBuffer<float>(d);
    mMinYRange = readFromBuffer<float>(d);
    mMaxYRange = readFromBuffer<float>(d);
    mMinZRange = readFromBuffer<float>(d);
    mMaxZRange = readFromBuffer<float>(d);
    mNumDirBins = readFromBuffer<int32_t>(d);
    mDirOffset = readFromBuffer<float>(d);
    mDirLimitOffset = readFromBuffer<float>(d);
    mScoreThreashold = readFromBuffer<float>(d);
    mNumClasses = readFromBuffer<int32_t>(d);
    mFeatureH = readFromBuffer<int32_t>(d);
    mFeatureW = readFromBuffer<int32_t>(d);

    mAnchorBottomHeight.resize(mNumClasses);
    for (int32_t i = 0; i < mNumClasses; i++)
    {
        mAnchorBottomHeight[i] = readFromBuffer<float>(d);
    }

    mAnchors.resize(mNumClasses * 2 * 4);
    for (int32_t i = 0; i < mNumClasses * 2 * 4; i++)
    {
        mAnchors[i] = readFromBuffer<float>(d);
    }

    PLUGIN_VALIDATE(d == reinterpret_cast<uint8_t const*>(data) + length);
}

nvinfer1::IPluginV2DynamicExt* DecodeBbox3DPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new DecodeBbox3DPlugin(mMinXRange, mMaxXRange, mMinYRange, mMaxYRange, mMinZRange, mMaxZRange,
            mNumDirBins, mDirOffset, mDirLimitOffset, mAnchorBottomHeight, mAnchors, mScoreThreashold, mFeatureH,
            mFeatureW);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::DimsExprs DecodeBbox3DPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(getNbOutputs() == 2);
        PLUGIN_VALIDATE(outputIndex >= 0 && outputIndex < getNbOutputs());
        PLUGIN_VALIDATE(inputs != nullptr);
        auto const& featureH = inputs[0].d[1];
        auto const& featureW = inputs[0].d[2];
        auto const& batchSize = inputs[0].d[0];
        if (outputIndex == 0)
        {
            nvinfer1::DimsExprs dim0{};
            dim0.nbDims = 3;
            dim0.d[0] = batchSize;
            dim0.d[1] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, featureH[0],
                exprBuilder.operation(
                    nvinfer1::DimensionOperation::kPROD, featureW[0], exprBuilder.constant(mNumClasses * 2)[0])[0]);
            dim0.d[2] = exprBuilder.constant(9);
            return dim0;
        }
        nvinfer1::DimsExprs dim1{};
        dim1.nbDims = 1;
        dim1.d[0] = batchSize;
        return dim1;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nvinfer1::DimsExprs{};
}

bool DecodeBbox3DPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(nbInputs == 3);
        PLUGIN_VALIDATE(nbOutputs == 2);
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE((pos >= 0) && (pos < nbInputs + nbOutputs));
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return false;
    }

    PluginTensorDesc const& in = inOut[pos];
    if (pos == 0) // cls_preds
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1) // box_preds
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2) // dir_cls_preds
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 3) // boxes
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 4) // box_num
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    return false;
}

void DecodeBbox3DPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(in != nullptr);
        mFeatureH = in[0].desc.dims.d[1];
        mFeatureW = in[0].desc.dims.d[2];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

size_t DecodeBbox3DPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    size_t mAnchorsSize = mNumClasses * 2 * 4 * sizeof(float);
    size_t mAnchorBottomHeightSize = mNumClasses * sizeof(float);
    size_t workspaces[2];
    workspaces[0] = mAnchorsSize;
    workspaces[1] = mAnchorBottomHeightSize;
    return calculateTotalWorkspaceSize(workspaces, 2);
}

int32_t DecodeBbox3DPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* /* outputDesc */, void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr && inputs != nullptr && outputs != nullptr && workspace != nullptr);

        int32_t batchSize = inputDesc[0].dims.d[0];

        // Inputs
        auto const* clsInput = static_cast<float const*>(inputs[0]);
        auto const* boxInput = static_cast<float const*>(inputs[1]);
        auto const* dirClsInput = static_cast<float const*>(inputs[2]);

        // Outputs
        auto* bndboxOutput = static_cast<float*>(outputs[0]);
        auto* boxNum = static_cast<int32_t*>(outputs[1]);

        // Initialize workspaces
        auto* anchors = static_cast<float*>(workspace);
        size_t anchorsSize = mNumClasses * 2 * 4 * sizeof(float);
        auto* anchorBottomHeight
            = reinterpret_cast<float*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(anchors), anchorsSize));
        size_t anchorBottomHeightSize = mNumClasses * sizeof(float);
        PLUGIN_CUASSERT(cudaMemcpyAsync(anchors, &mAnchors[0], anchorsSize, cudaMemcpyHostToDevice, stream));
        PLUGIN_CUASSERT(cudaMemcpyAsync(
            anchorBottomHeight, &mAnchorBottomHeight[0], anchorBottomHeightSize, cudaMemcpyHostToDevice, stream));
        // Initialize boxNum to 0
        PLUGIN_CUASSERT(cudaMemsetAsync(boxNum, 0, batchSize * sizeof(int32_t), stream));

        decodeBbox3DLaunch(batchSize, clsInput, boxInput, dirClsInput, anchors, anchorBottomHeight, bndboxOutput,
            boxNum, mMinXRange, mMaxXRange, mMinYRange, mMaxYRange, mFeatureW, mFeatureH, mNumClasses * 2, mNumClasses,
            7, mScoreThreashold, mDirOffset, mDirLimitOffset, mNumDirBins, stream);
        return cudaPeekAtLastError();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

nvinfer1::DataType DecodeBbox3DPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputTypes != nullptr);
        if (index == 0)
        {
            return inputTypes[0];
        }
        return nvinfer1::DataType::kINT32;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nvinfer1::DataType{};
}

char const* DecodeBbox3DPlugin::getPluginType() const noexcept
{
    return kPLUGIN_NAME;
}

char const* DecodeBbox3DPlugin::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

int32_t DecodeBbox3DPlugin::getNbOutputs() const noexcept
{
    return 2;
}

int32_t DecodeBbox3DPlugin::initialize() noexcept
{
    return 0;
}

void DecodeBbox3DPlugin::terminate() noexcept {}

size_t DecodeBbox3DPlugin::getSerializationSize() const noexcept
{
    size_t scalarSize = 9 * sizeof(float) + 4 * sizeof(int32_t);
    size_t vectorSize = mNumClasses * 9 * sizeof(float);
    return scalarSize + vectorSize;
}

void DecodeBbox3DPlugin::serialize(void* buffer) const noexcept
{
    PLUGIN_ASSERT(buffer != nullptr);
    auto* d = reinterpret_cast<uint8_t*>(buffer);
    auto* const start = d;
    writeToBuffer<float>(d, mMinXRange);
    writeToBuffer<float>(d, mMaxXRange);
    writeToBuffer<float>(d, mMinYRange);
    writeToBuffer<float>(d, mMaxYRange);
    writeToBuffer<float>(d, mMinZRange);
    writeToBuffer<float>(d, mMaxZRange);
    writeToBuffer<int32_t>(d, mNumDirBins);
    writeToBuffer<float>(d, mDirOffset);
    writeToBuffer<float>(d, mDirLimitOffset);
    writeToBuffer<float>(d, mScoreThreashold);
    writeToBuffer<int32_t>(d, mNumClasses);
    writeToBuffer<int32_t>(d, mFeatureH);
    writeToBuffer<int32_t>(d, mFeatureW);
    for (int32_t i = 0; i < mNumClasses; i++)
    {
        writeToBuffer<float>(d, mAnchorBottomHeight[i]);
    }
    for (int32_t i = 0; i < mNumClasses * 2 * 4; i++)
    {
        writeToBuffer<float>(d, mAnchors[i]);
    }
    PLUGIN_ASSERT(d == start + getSerializationSize());
}

void DecodeBbox3DPlugin::destroy() noexcept
{
    delete this;
}

void DecodeBbox3DPlugin::setPluginNamespace(char const* libNamespace) noexcept
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

char const* DecodeBbox3DPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

DecodeBbox3DPluginCreator::DecodeBbox3DPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("point_cloud_range", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("anchors", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("anchor_bottom_height", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("dir_offset", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("dir_limit_offset", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_dir_bins", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("score_thresh", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* DecodeBbox3DPluginCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* DecodeBbox3DPluginCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

PluginFieldCollection const* DecodeBbox3DPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* DecodeBbox3DPluginCreator::createPlugin(char const* /*name*/, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const* fields = fc->fields;

        // Initialize default values for attributes.
        float pointCloudRange[6] = {0.F};
        std::vector<float> anchors{};
        std::vector<float> anchorBottomHeight{};
        float dirOffset = 0.78539F;
        float dirLimitOffset = 0.F;
        int32_t numDirBins = 2;
        float scoreThreshold = 0.F;

        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            char const* attr_name = fields[i].name;
            if (!strcmp(attr_name, "point_cloud_range"))
            {
                auto const* d = static_cast<float const*>(fields[i].data);
                for (int32_t pointCloudIdx = 0; pointCloudIdx < 6; pointCloudIdx++)
                {
                    pointCloudRange[pointCloudIdx] = d[pointCloudIdx];
                }
            }
            else if (!strcmp(attr_name, "anchors"))
            {
                auto const* d = static_cast<float const*>(fields[i].data);
                for (int32_t j = 0; j < fields[i].length; ++j)
                {
                    anchors.push_back(d[j]);
                }
            }
            else if (!strcmp(attr_name, "anchor_bottom_height"))
            {
                auto const* d = static_cast<float const*>(fields[i].data);
                for (int32_t j = 0; j < fields[i].length; ++j)
                {
                    anchorBottomHeight.push_back(d[j]);
                }
            }
            else if (!strcmp(attr_name, "dir_offset"))
            {
                auto const* d = static_cast<float const*>(fields[i].data);
                dirOffset = d[0];
            }
            else if (!strcmp(attr_name, "dir_limit_offset"))
            {
                auto const* d = static_cast<float const*>(fields[i].data);
                dirLimitOffset = d[0];
            }
            else if (!strcmp(attr_name, "num_dir_bins"))
            {
                auto const* d = static_cast<int32_t const*>(fields[i].data);
                numDirBins = d[0];
            }
            else if (!strcmp(attr_name, "score_thresh"))
            {
                auto const* d = static_cast<float const*>(fields[i].data);
                scoreThreshold = d[0];
            }
        }
        IPluginV2* plugin = new DecodeBbox3DPlugin(pointCloudRange[0], pointCloudRange[3], pointCloudRange[1],
            pointCloudRange[4], pointCloudRange[2], pointCloudRange[5], numDirBins, dirOffset, dirLimitOffset,
            anchorBottomHeight, anchors, scoreThreshold);
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* DecodeBbox3DPluginCreator::deserializePlugin(
    char const* /*name*/, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        return new DecodeBbox3DPlugin(serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void DecodeBbox3DPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
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

char const* DecodeBbox3DPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
} // namespace nvinfer1::plugin
