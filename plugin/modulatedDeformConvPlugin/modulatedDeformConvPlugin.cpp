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

/*
 **************************************************************************
 * Modified from mmcv (https://github.com/open-mmlab/mmcv/tree/master/mmcv)
 * Copyright (c) OpenMMLab. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 [see LICENSE for details]
 * https://github.com/open-mmlab/mmcv/blob/master/LICENSE
 **************************************************************************
 */

#include "modulatedDeformConvPlugin.h"
#include <assert.h>
#include <chrono>

using namespace nvinfer1;
using nvinfer1::plugin::ModulatedDeformableConvPluginDynamic;
using nvinfer1::plugin::ModulatedDeformableConvPluginDynamicCreator;

void ModulatedDeformConvForwardCUDAKernelLauncherFloat(float const* input, float const* weight, float const* bias,
    float const* offset, float const* mask, float* output, void* workspace, int32_t batch, int32_t channels,
    int32_t height, int32_t width, int32_t channelsOut, int32_t kernelW, int32_t kernelH, int32_t strideW,
    int32_t strideH, int32_t padW, int32_t padH, int32_t dilationW, int32_t dilationH, int32_t group,
    int32_t deformableGroup, int32_t im2colStep, cublasHandle_t cublasHandle, cudaStream_t stream);

namespace
{
static char const* PLUGIN_VERSION{"1"};
static char const* PLUGIN_NAME{"ModulatedDeformConv2d"};
} // namespace

nvinfer1::PluginFieldCollection ModulatedDeformableConvPluginDynamicCreator::mFC{};
std::vector<nvinfer1::PluginField> ModulatedDeformableConvPluginDynamicCreator::mPluginAttributes;

ModulatedDeformableConvPluginDynamic::ModulatedDeformableConvPluginDynamic(std::string const& name,
    const nvinfer1::Dims stride, const nvinfer1::Dims padding, const nvinfer1::Dims dilation,
    int32_t const deformableGroup, int32_t const group)
    : mLayerName(name)
    , mStride(stride)
    , mPadding(padding)
    , mDilation(dilation)
    , mDeformableGroup(deformableGroup)
    , mGroup(group)
{
    mWithBias = false;
}

ModulatedDeformableConvPluginDynamic::ModulatedDeformableConvPluginDynamic(
    const std::string name, void const* data, size_t length)
    : mLayerName(name)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    mStride = read<nvinfer1::Dims>(d);
    mPadding = read<nvinfer1::Dims>(d);
    mDilation = read<nvinfer1::Dims>(d);
    mDeformableGroup = read<int32_t>(d);
    mGroup = read<int32_t>(d);
    PLUGIN_VALIDATE(d == a + length);
    mWithBias = false;
}

ModulatedDeformableConvPluginDynamic::~ModulatedDeformableConvPluginDynamic() {}

nvinfer1::IPluginV2DynamicExt* ModulatedDeformableConvPluginDynamic::clone() const noexcept
{
    try
    {
        ModulatedDeformableConvPluginDynamic* plugin = new ModulatedDeformableConvPluginDynamic(
            mLayerName, mStride, mPadding, mDilation, mDeformableGroup, mGroup);
        plugin->setPluginNamespace(getPluginNamespace());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::DimsExprs ModulatedDeformableConvPluginDynamic::getOutputDimensions(int32_t outputIndex,
    nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        nvinfer1::DimsExprs ret;
        ret.nbDims = 4;
        ret.d[0] = inputs[0].d[0];
        ret.d[1] = inputs[3].d[0];

        ret.d[2] = inputs[1].d[2];
        ret.d[3] = inputs[1].d[3];
        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool ModulatedDeformableConvPluginDynamic::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (pos == 0)
    {
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
    }
    else
    {
        return inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format;
    }
}

void ModulatedDeformableConvPluginDynamic::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
    int32_t nbInputs, nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    try
    {
        if (mCublasHandle == nullptr)
        {
            PLUGIN_CUBLASASSERT(cublasCreate(&mCublasHandle));
        }
        if (nbInputs == 5)
        {
            mWithBias = true;
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

size_t ModulatedDeformableConvPluginDynamic::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs,
    int32_t nbInputs, nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    int32_t sizeofDtype = nvinfer1::plugin::bert::getElementSize(outputs[0].type);

    int32_t nInputPlane = inputs[0].dims.d[1];
    int32_t outputHeight = outputs[0].dims.d[2];
    int32_t outputWidth = outputs[0].dims.d[3];
    int32_t kH = inputs[3].dims.d[2];
    int32_t kW = inputs[3].dims.d[3];

    int64_t colSize = divUp(nInputPlane * kW * kH * outputHeight * outputWidth * sizeofDtype, 16) * 16;

    return colSize;
}

int32_t ModulatedDeformableConvPluginDynamic::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workSpace,
    cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr
            && workSpace != nullptr);

        int32_t batch = inputDesc[0].dims.d[0];
        int32_t channels = inputDesc[0].dims.d[1];
        int32_t height = inputDesc[0].dims.d[2];
        int32_t width = inputDesc[0].dims.d[3];
        int32_t channelsOut = outputDesc[0].dims.d[1];
        int32_t kernelH = inputDesc[3].dims.d[2];
        int32_t kernelW = inputDesc[3].dims.d[3];

        void const* x = inputs[0];
        void const* offset = inputs[1];
        void const* mask = inputs[2];
        void const* weight = inputs[3];
        void const* bias = mWithBias ? inputs[4] : nullptr;
        void* output = outputs[0];
        int32_t im2colStep = std::min(batch, 32);

        auto data_type = inputDesc[0].type;
        switch (data_type)
        {
        case nvinfer1::DataType::kFLOAT:
            ModulatedDeformConvForwardCUDAKernelLauncherFloat((float*) x, (float*) weight, (float*) bias,
                (float*) offset, (float*) mask, (float*) output, workSpace, batch, channels, height, width, channelsOut,
                kernelW, kernelH, mStride.d[0], mStride.d[1], mPadding.d[0], mPadding.d[1], mDilation.d[0],
                mDilation.d[1], mGroup, mDeformableGroup, im2colStep, mCublasHandle, stream);
            break;
        default: return 1;
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }

    return 0;
}

nvinfer1::DataType ModulatedDeformableConvPluginDynamic::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

// IPluginV2 Methods
char const* ModulatedDeformableConvPluginDynamic::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

char const* ModulatedDeformableConvPluginDynamic::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int32_t ModulatedDeformableConvPluginDynamic::getNbOutputs() const noexcept
{
    return 1;
}

int32_t ModulatedDeformableConvPluginDynamic::initialize() noexcept
{
    return 0;
}

void ModulatedDeformableConvPluginDynamic::terminate() noexcept {}

size_t ModulatedDeformableConvPluginDynamic::getSerializationSize() const noexcept
{
    return sizeof(mStride) + sizeof(mPadding) + sizeof(mDilation) + sizeof(mDeformableGroup) + sizeof(mGroup);
}

void ModulatedDeformableConvPluginDynamic::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    write(d, mStride);
    write(d, mPadding);
    write(d, mDilation);
    write(d, mDeformableGroup);
    write(d, mGroup);
}

void ModulatedDeformableConvPluginDynamic::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void ModulatedDeformableConvPluginDynamic::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) noexcept {}

void ModulatedDeformableConvPluginDynamic::detachFromContext() noexcept
{
    PLUGIN_CUBLASASSERT(cublasDestroy(mCublasHandle));
}

void ModulatedDeformableConvPluginDynamic::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* ModulatedDeformableConvPluginDynamic::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

ModulatedDeformableConvPluginDynamicCreator::ModulatedDeformableConvPluginDynamicCreator()
{
    mPluginAttributes.emplace_back(nvinfer1::PluginField("stride"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("padding"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("dilation"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("group"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("deformable_group"));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* ModulatedDeformableConvPluginDynamicCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

char const* ModulatedDeformableConvPluginDynamicCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

nvinfer1::PluginFieldCollection const* ModulatedDeformableConvPluginDynamicCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV2* ModulatedDeformableConvPluginDynamicCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    try
    {
        nvinfer1::Dims stride{2, {1, 1}};
        nvinfer1::Dims padding{2, {0, 0}};
        nvinfer1::Dims dilation{2, {1, 1}};
        int32_t deformableGroup = 1;
        int32_t group = 1;
        plugin::validateRequiredAttributesExist({"deformable_group", "group", "stride", "padding", "dilation"}, fc);

        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            if (fc->fields[i].data == nullptr)
            {
                continue;
            }
            std::string field_name(fc->fields[i].name);

            if (field_name.compare("deformable_group") == 0)
            {
                PLUGIN_VALIDATE(fc->fields[i].type == PluginFieldType::kINT32);
                deformableGroup = static_cast<int32_t const*>(fc->fields[i].data)[0];
                PLUGIN_VALIDATE(deformableGroup > 0);
            }

            if (field_name.compare("group") == 0)
            {
                PLUGIN_VALIDATE(fc->fields[i].type == PluginFieldType::kINT32);
                group = static_cast<int32_t const*>(fc->fields[i].data)[0];
                PLUGIN_VALIDATE(group > 0);
            }

            if (field_name.compare("stride") == 0)
            {
                PLUGIN_VALIDATE(fc->fields[i].type == PluginFieldType::kINT32);
                stride.nbDims = 2;
                stride.d[0] = static_cast<int32_t const*>(fc->fields[i].data)[0];
                stride.d[1] = static_cast<int32_t const*>(fc->fields[i].data)[1];
                PLUGIN_VALIDATE(stride.d[0] > 0);
                PLUGIN_VALIDATE(stride.d[1] > 0);
            }

            if (field_name.compare("padding") == 0)
            {
                PLUGIN_VALIDATE(fc->fields[i].type == PluginFieldType::kINT32);
                padding.nbDims = 2;
                padding.d[0] = static_cast<int32_t const*>(fc->fields[i].data)[0];
                padding.d[1] = static_cast<int32_t const*>(fc->fields[i].data)[1];
                PLUGIN_VALIDATE(padding.d[0] >= 0);
                PLUGIN_VALIDATE(padding.d[1] >= 0);
            }

            if (field_name.compare("dilation") == 0)
            {
                PLUGIN_VALIDATE(fc->fields[i].type == PluginFieldType::kINT32);
                dilation.nbDims = 2;
                dilation.d[0] = static_cast<int32_t const*>(fc->fields[i].data)[0];
                dilation.d[1] = static_cast<int32_t const*>(fc->fields[i].data)[1];
                PLUGIN_VALIDATE(dilation.d[0] > 0);
                PLUGIN_VALIDATE(dilation.d[1] > 0);
            }
        }

        ModulatedDeformableConvPluginDynamic* plugin
            = new ModulatedDeformableConvPluginDynamic(name, stride, padding, dilation, deformableGroup, group);
        plugin->setPluginNamespace(getPluginNamespace());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::IPluginV2* ModulatedDeformableConvPluginDynamicCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto plugin = new ModulatedDeformableConvPluginDynamic(name, serialData, serialLength);
        plugin->setPluginNamespace(getPluginNamespace());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void ModulatedDeformableConvPluginDynamicCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* ModulatedDeformableConvPluginDynamicCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
