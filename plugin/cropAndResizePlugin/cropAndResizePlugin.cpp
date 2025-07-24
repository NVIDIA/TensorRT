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
#include "cropAndResizePlugin.h"
#include <cstring>
#include <vector>

namespace nvinfer1::plugin
{
// Plugin-specific constants
namespace
{
char const* const kCROP_AND_RESIZE_DYNAMIC_PLUGIN_VERSION{"2"};
char const* const kCROP_AND_RESIZE_DYNAMIC_PLUGIN_NAME{"CropAndResizeDynamic"};
} // namespace

CropAndResizeDynamicPlugin::CropAndResizeDynamicPlugin(int32_t cropWidth, int32_t cropHeight)
    : mCropWidth(cropWidth)
    , mCropHeight(cropHeight)
{
}

CropAndResizeDynamicPlugin::CropAndResizeDynamicPlugin(
    int32_t cropWidth, int32_t cropHeight, int32_t depth, int32_t inputWidth, int32_t inputHeight, int32_t maxBoxNum)
    : mCropWidth(cropWidth)
    , mCropHeight(cropHeight)
    , mDepth(depth)
    , mInputWidth(inputWidth)
    , mInputHeight(inputHeight)
    , mNumBoxes(maxBoxNum)
{
}

CropAndResizeDynamicPlugin::~CropAndResizeDynamicPlugin() noexcept {}

char const* CropAndResizeDynamicPlugin::getPluginName() const noexcept
{
    return kCROP_AND_RESIZE_DYNAMIC_PLUGIN_NAME;
}

char const* CropAndResizeDynamicPlugin::getPluginVersion() const noexcept
{
    return kCROP_AND_RESIZE_DYNAMIC_PLUGIN_VERSION;
}

int32_t CropAndResizeDynamicPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t CropAndResizeDynamicPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* shapeInputs, int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(outputs != nullptr);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(inputs[0].nbDims == 4);

        // 5D output (N, R, C, H, W)
        outputs[0].nbDims = 5;
        outputs[0].d[0] = inputs[0].d[0]; // N(batch)
        outputs[0].d[1] = inputs[1].d[1]; // R(MaxBoxNum)
        outputs[0].d[2] = inputs[0].d[1]; // C(channel)
        outputs[0].d[3] = exprBuilder.constant(mCropHeight);
        outputs[0].d[4] = exprBuilder.constant(mCropWidth);

        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

int32_t CropAndResizeDynamicPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr && inputs != nullptr && outputs != nullptr);

        // Our plugin outputs only one tensor
        void* output = outputs[0];

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

size_t CropAndResizeDynamicPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

IPluginV3* CropAndResizeDynamicPlugin::clone() noexcept
{
    try
    {
        auto plugin = std::make_unique<CropAndResizeDynamicPlugin>(
            mCropWidth, mCropHeight, mDepth, mInputWidth, mInputHeight, mNumBoxes);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginCapability* CropAndResizeDynamicPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

PluginFieldCollection const* CropAndResizeDynamicPlugin::getFieldsToSerialize() noexcept
{
    try
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back(PluginField("crop_width", &mCropWidth, PluginFieldType::kINT32, 1));
        mDataToSerialize.emplace_back(PluginField("crop_height", &mCropHeight, PluginFieldType::kINT32, 1));

        mFCToSerialize.nbFields = mDataToSerialize.size();
        mFCToSerialize.fields = mDataToSerialize.data();
        return &mFCToSerialize;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

int32_t CropAndResizeDynamicPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputTypes != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        // One output
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(outputTypes != nullptr);
        outputTypes[0] = DataType::kFLOAT;
        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

int32_t CropAndResizeDynamicPlugin::onShapeChange(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(inputs != nullptr);

        // Re-validate dimensions and update internal state if needed
        // Here we can update mDepth, mInputHeight, mInputWidth, mNumBoxes if they change
        mDepth = inputs[0].dims.d[1];
        mInputHeight = inputs[0].dims.d[2];
        mInputWidth = inputs[0].dims.d[3];
        mNumBoxes = inputs[1].dims.d[1];

        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

IPluginV3* CropAndResizeDynamicPlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    try
    {
        return clone(); // Simple clone is sufficient since no context resources are needed
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

bool CropAndResizeDynamicPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        // 2 inputs, 1 outputs, so 3 input/output in total
        PLUGIN_VALIDATE(0 <= pos && pos < 3);
        PLUGIN_VALIDATE(inOut != nullptr);
        auto const* in = inOut;
        auto const* out = inOut + nbInputs;
        bool const consistentFloatPrecision = (in[0].desc.type == in[pos].desc.type);
        switch (pos)
        {
        case 0:
            return (in[0].desc.type == DataType::kFLOAT && in[0].desc.format == PluginFormat::kLINEAR
                && consistentFloatPrecision);
        case 1:
            return (in[1].desc.type == DataType::kFLOAT && in[1].desc.format == PluginFormat::kLINEAR
                && consistentFloatPrecision);
        case 2:
            return (out[0].desc.type == DataType::kFLOAT && out[0].desc.format == PluginFormat::kLINEAR
                && consistentFloatPrecision);
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

void CropAndResizeDynamicPlugin::setPluginNamespace(char const* libNamespace) noexcept
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

int32_t CropAndResizeDynamicPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        // Validate input/output counts and update internal state based on input dimensions
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
        return STATUS_FAILURE;
    }
    return STATUS_SUCCESS;
}

char const* CropAndResizeDynamicPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

CropAndResizeDynamicPluginCreator::CropAndResizeDynamicPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("crop_width", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("crop_height", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* CropAndResizeDynamicPluginCreator::getPluginName() const noexcept
{
    return kCROP_AND_RESIZE_DYNAMIC_PLUGIN_NAME;
}

char const* CropAndResizeDynamicPluginCreator::getPluginVersion() const noexcept
{
    return kCROP_AND_RESIZE_DYNAMIC_PLUGIN_VERSION;
}

PluginFieldCollection const* CropAndResizeDynamicPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV3* CropAndResizeDynamicPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);
        PLUGIN_VALIDATE(fc->fields != nullptr);

        validateRequiredAttributesExist({"crop_width", "crop_height"}, fc);

        int32_t cropWidth = 0;
        int32_t cropHeight = 0;

        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            PluginField const& field = fc->fields[i];
            PLUGIN_VALIDATE(field.type == PluginFieldType::kINT32);

            std::string_view const fieldName{field.name};
            auto const value = *static_cast<int32_t const*>(field.data);

            if (fieldName == "crop_width")
            {
                cropWidth = value;
            }
            else if (fieldName == "crop_height")
            {
                cropHeight = value;
            }
        }

        PLUGIN_VALIDATE(cropWidth > 0);
        PLUGIN_VALIDATE(cropHeight > 0);

        auto plugin = std::make_unique<CropAndResizeDynamicPlugin>(cropWidth, cropHeight);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void CropAndResizeDynamicPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(pluginNamespace != nullptr);
        mNamespace = pluginNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* CropAndResizeDynamicPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

} // namespace nvinfer1::plugin
