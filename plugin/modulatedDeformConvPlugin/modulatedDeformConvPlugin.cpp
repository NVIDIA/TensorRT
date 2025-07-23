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

/*
 **************************************************************************
 * Modified from mmcv (https://github.com/open-mmlab/mmcv/tree/master/mmcv)
 * Copyright (c) OpenMMLab. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 [see LICENSE for details]
 * https://github.com/open-mmlab/mmcv/blob/master/LICENSE
 **************************************************************************
 */

#include "modulatedDeformConvPlugin.h"
#include <algorithm>

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using nvinfer1::plugin::ModulatedDeformableConvPluginDynamic;
using nvinfer1::plugin::ModulatedDeformableConvPluginDynamicCreator;

void ModulatedDeformConvForwardCUDAKernelLauncherFloat(float const* input, float const* weight, float const* bias,
    float const* offset, float const* mask, float* output, void* workspace, int32_t batch, int32_t channels,
    int32_t height, int32_t width, int32_t channelsOut, int32_t kernelW, int32_t kernelH, int32_t strideW,
    int32_t strideH, int32_t padW, int32_t padH, int32_t dilationW, int32_t dilationH, int32_t group,
    int32_t deformableGroup, int32_t im2colStep, nvinfer1::pluginInternal::cublasHandle_t cublasHandle,
    cudaStream_t stream);

void ModulatedDeformConvForwardCUDAKernelLauncherHalf(half const* input, half const* weight, half const* bias,
    half const* offset, half const* mask, half* output, void* workspace, int32_t batch, int32_t channels,
    int32_t height, int32_t width, int32_t channelsOut, int32_t kernelW, int32_t kernelH, int32_t strideW,
    int32_t strideH, int32_t padW, int32_t padH, int32_t dilationW, int32_t dilationH, int32_t group,
    int32_t deformableGroup, int32_t im2colStep, nvinfer1::pluginInternal::cublasHandle_t cublasHandle,
    cudaStream_t stream);

namespace
{
static char const* PLUGIN_VERSION{"2"};
static char const* PLUGIN_NAME{"ModulatedDeformConv2d"};
} // namespace

ModulatedDeformableConvPluginDynamic::ModulatedDeformableConvPluginDynamic(std::string const& name,
    nvinfer1::Dims const stride, nvinfer1::Dims const padding, nvinfer1::Dims const dilation,
    int32_t const deformableGroup, int32_t const group)
    : mLayerName(name)
    , mStride(stride)
    , mPadding(padding)
    , mDilation(dilation)
    , mDeformableGroup(deformableGroup)
    , mGroup(group)
    , mWithBias(0)
{
}

ModulatedDeformableConvPluginDynamic::~ModulatedDeformableConvPluginDynamic() {}

nvinfer1::IPluginV3* ModulatedDeformableConvPluginDynamic::clone() noexcept
{
    try
    {
        auto* plugin = new ModulatedDeformableConvPluginDynamic(
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

IPluginCapability* ModulatedDeformableConvPluginDynamic::getCapabilityInterface(PluginCapabilityType type) noexcept
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

int32_t ModulatedDeformableConvPluginDynamic::getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
    nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr && outputs != nullptr);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(nbInputs == 4 || nbInputs == 5); // nbInputs depends on bias

        // Output shape is (N, C_out, H_out, W_out)
        // N = N_in (inputs[0].d[0])
        // C_out = C_weight (inputs[3].d[0])
        // H_out = H_offset (inputs[1].d[2])
        // W_out = W_offset (inputs[1].d[3])
        outputs[0].nbDims = 4;
        outputs[0].d[0] = inputs[0].d[0]; // Batch size
        outputs[0].d[1] = inputs[3].d[0]; // Output channels from weight tensor
        outputs[0].d[2] = inputs[1].d[2]; // Output height from offset tensor
        outputs[0].d[3] = inputs[1].d[3]; // Output width from offset tensor
        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

bool ModulatedDeformableConvPluginDynamic::supportsFormatCombination(
    int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        if (pos == 0)
        {
            // Input tensor must be FP32 or FP16 and linear format
            return ((inOut[pos].desc.type == nvinfer1::DataType::kFLOAT
                        || inOut[pos].desc.type == nvinfer1::DataType::kHALF)
                && inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR);
        }
        // All other tensors must have the same type and format as the input tensor
        return inOut[pos].desc.type == inOut[0].desc.type && inOut[pos].desc.format == inOut[0].desc.format;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

int32_t ModulatedDeformableConvPluginDynamic::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* /* in */,
    int32_t /* nbInputs */, nvinfer1::DynamicPluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept
{
    // Bias presence (mWithBias) is determined dynamically in onShapeChange based on nbInputs.
    // No other configuration needed here.
    return STATUS_SUCCESS;
}

int32_t ModulatedDeformableConvPluginDynamic::onShapeChange(nvinfer1::PluginTensorDesc const* /* inputs */,
    int32_t nbInputs, nvinfer1::PluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) noexcept
{
    try
    {
        // Determine if bias is present based on the number of inputs.
        mWithBias = (nbInputs == 5);
        // No specific shape-dependent updates needed for this plugin's internal state.
        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

size_t ModulatedDeformableConvPluginDynamic::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs,
    int32_t /* nbInputs */, nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t /* nbOutputs */) const noexcept
{
    // Calculate workspace size needed for the im2col buffer.
    int32_t const sizeOfDtype = nvinfer1::plugin::bert::getElementSize(outputs[0].desc.type);

    int32_t const nInputPlane = inputs[0].desc.dims.d[1]; // Input channels
    int32_t const outputHeight = outputs[0].desc.dims.d[2];
    int32_t const outputWidth = outputs[0].desc.dims.d[3];
    int32_t const kernelH = inputs[3].desc.dims.d[2]; // Weight kernel height
    int32_t const kernelW = inputs[3].desc.dims.d[3]; // Weight kernel width

    // Calculate size needed for the intermediate 'columns' buffer used in im2col + GEMM approach.
    int64_t const colSize
        = divUp(static_cast<int64_t>(nInputPlane) * kernelW * kernelH * outputHeight * outputWidth * sizeOfDtype, 16)
        * 16; // Align to 16 bytes

    return static_cast<size_t>(colSize);
}

int32_t ModulatedDeformableConvPluginDynamic::enqueue(nvinfer1::PluginTensorDesc const* inputDescs,
    nvinfer1::PluginTensorDesc const* outputDescs, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDescs != nullptr && outputDescs != nullptr && inputs != nullptr && outputs != nullptr
            && workspace != nullptr);

        // Extract dimensions
        int32_t const batch = inputDescs[0].dims.d[0];
        int32_t const channels = inputDescs[0].dims.d[1];
        int32_t const height = inputDescs[0].dims.d[2];
        int32_t const width = inputDescs[0].dims.d[3];
        int32_t const channelsOut = outputDescs[0].dims.d[1];
        int32_t const kernelH = inputDescs[3].dims.d[2]; // Weight kernel height
        int32_t const kernelW = inputDescs[3].dims.d[3]; // Weight kernel width

        // Get input/output pointers
        void const* inputTensor = inputs[0];
        void const* offsetTensor = inputs[1];
        void const* maskTensor = inputs[2];
        void const* weightTensor = inputs[3];
        void const* biasTensor = mWithBias ? inputs[4] : nullptr;
        void* outputTensor = outputs[0];

        // Determine im2col step size
        int32_t const im2colStep = std::min(batch, 32);

        DataType const dataType = inputDescs[0].type;
        switch (dataType)
        {
        case nvinfer1::DataType::kFLOAT:
            ModulatedDeformConvForwardCUDAKernelLauncherFloat(static_cast<float const*>(inputTensor),
                static_cast<float const*>(weightTensor), static_cast<float const*>(biasTensor),
                static_cast<float const*>(offsetTensor), static_cast<float const*>(maskTensor),
                static_cast<float*>(outputTensor), workspace, batch, channels, height, width, channelsOut, kernelW,
                kernelH, mStride.d[0], mStride.d[1], mPadding.d[0], mPadding.d[1], mDilation.d[0], mDilation.d[1],
                mGroup, mDeformableGroup, im2colStep, mCublasHandle, stream);
            break;
        case nvinfer1::DataType::kHALF:
            ModulatedDeformConvForwardCUDAKernelLauncherHalf(static_cast<half const*>(inputTensor),
                static_cast<half const*>(weightTensor), static_cast<half const*>(biasTensor),
                static_cast<half const*>(offsetTensor), static_cast<half const*>(maskTensor),
                static_cast<half*>(outputTensor), workspace, batch, channels, height, width, channelsOut, kernelW,
                kernelH, mStride.d[0], mStride.d[1], mPadding.d[0], mPadding.d[1], mDilation.d[0], mDilation.d[1],
                mGroup, mDeformableGroup, im2colStep, mCublasHandle, stream);
            break;
        default:
            // Unsupported data type
            return STATUS_FAILURE;
        }
        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

IPluginV3* ModulatedDeformableConvPluginDynamic::attachToContext(nvinfer1::IPluginResourceContext* context) noexcept
{
    try
    {
        auto* p = static_cast<ModulatedDeformableConvPluginDynamic*>(clone());
        // The clone has shared ownership of the underlying cublasWrapper instance
        // that is mapped to the current context.
        p->setCublasResources(nvinfer1::pluginInternal::createPluginCublasWrapper(context));
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void ModulatedDeformableConvPluginDynamic::setCublasResources(
    std::shared_ptr<nvinfer1::pluginInternal::CublasWrapper> cublasWrapper)
{
    mCublasWrapper = cublasWrapper;
    if (mCublasWrapper)
    {
        // The shared cublasWrapper resource owns the handle.
        // `this` instance has a non-owning pointer to the handle.
        // The cublasWrapper initializes the handle and checks for nullptr.
        mCublasHandle = mCublasWrapper->getCublasHandle();
    }
    // else: mCublasHandle remains nullptr, handle potential errors in enqueue
}

int32_t ModulatedDeformableConvPluginDynamic::getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(outputTypes != nullptr && inputTypes != nullptr);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(nbInputs == 4 || nbInputs == 5); // Depends on bias

        // Output type must match the input type
        outputTypes[0] = inputTypes[0];
        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

char const* ModulatedDeformableConvPluginDynamic::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

char const* ModulatedDeformableConvPluginDynamic::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

void ModulatedDeformableConvPluginDynamic::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = (pluginNamespace == nullptr) ? "" : pluginNamespace;
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

int32_t ModulatedDeformableConvPluginDynamic::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::PluginFieldCollection const* ModulatedDeformableConvPluginDynamic::getFieldsToSerialize() noexcept
{
    try
    {
        mDataToSerialize.clear();
        // stride, padding, dilation are stored natively as int64 in memory
        // even though the plugin exposes them as int32.
        // Therefore, during build time, we upcast them to int64.
        // During runtime, we serialize/deserialize them as int64.
        // See ModulatedDeformableConvPluginDynamicCreator::createPlugin() on how we handle this.
        mDataToSerialize.emplace_back("stride", mStride.d, PluginFieldType::kINT64, 2);
        mDataToSerialize.emplace_back("padding", mPadding.d, PluginFieldType::kINT64, 2);
        mDataToSerialize.emplace_back("dilation", mDilation.d, PluginFieldType::kINT64, 2);
        mDataToSerialize.emplace_back("group", &mGroup, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("deformable_group", &mDeformableGroup, PluginFieldType::kINT32, 1);

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

////////////////////// creator /////////////////////////////

ModulatedDeformableConvPluginDynamicCreator::ModulatedDeformableConvPluginDynamicCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(PluginField("padding", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(PluginField("dilation", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(PluginField("group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("deformable_group", nullptr, PluginFieldType::kINT32, 1));

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

nvinfer1::IPluginV3* ModulatedDeformableConvPluginDynamicCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);
        PLUGIN_VALIDATE(fc->fields != nullptr || fc->nbFields == 0);

        nvinfer1::Dims stride{2, {1, 1}};
        nvinfer1::Dims padding{2, {0, 0}};
        nvinfer1::Dims dilation{2, {1, 1}};
        int32_t deformableGroup = 1;
        int32_t group = 1;

        plugin::validateRequiredAttributesExist({"deformable_group", "group", "stride", "padding", "dilation"}, fc);

        bool const isBuildPhase = (phase == nvinfer1::TensorRTPhase::kBUILD);

        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            PluginField const& field = fc->fields[i];
            // Skip fields with null data pointer
            if (field.data == nullptr)
            {
                continue;
            }

            std::string const fieldName(field.name);

            if (fieldName == "deformable_group")
            {
                PLUGIN_VALIDATE(field.type == PluginFieldType::kINT32);
                PLUGIN_VALIDATE(field.length == 1);
                deformableGroup = *static_cast<int32_t const*>(field.data);
                PLUGIN_VALIDATE(deformableGroup > 0);
            }
            else if (fieldName == "group")
            {
                PLUGIN_VALIDATE(field.type == PluginFieldType::kINT32);
                PLUGIN_VALIDATE(field.length == 1);
                group = *static_cast<int32_t const*>(field.data);
                PLUGIN_VALIDATE(group > 0);
            }
            else if (bert::elem(fieldName, {"stride", "padding", "dilation"}))
            {
                nvinfer1::Dims* dimsPtr
                    = (fieldName == "stride") ? &stride : ((fieldName == "padding") ? &padding : &dilation);

                PluginFieldType const expectedFieldType
                    = isBuildPhase ? PluginFieldType::kINT32 : PluginFieldType::kINT64;
                PLUGIN_VALIDATE(field.type == expectedFieldType);
                PLUGIN_VALIDATE(field.length == 2);
                dimsPtr->nbDims = 2;

                // To stay consistent with this plugin's IO, we expose int32 stride, padding, dilation
                // during build but store and serialize/deserialize as int64.
                if (isBuildPhase)
                {
                    // During build time, data is INT32, upcast to int64 for internal storage (Dims uses int64_t).
                    auto const* dataPtr = static_cast<int32_t const*>(field.data);
                    dimsPtr->d[0] = dataPtr[0];
                    dimsPtr->d[1] = dataPtr[1];
                }
                else // Runtime phase
                {
                    // During runtime, data is deserialized as INT64.
                    PLUGIN_VALIDATE(phase == nvinfer1::TensorRTPhase::kRUNTIME);
                    auto const* dataPtr = static_cast<int64_t const*>(field.data);
                    dimsPtr->d[0] = dataPtr[0];
                    dimsPtr->d[1] = dataPtr[1];
                }

                // Validate values
                if (fieldName == "padding")
                {
                    PLUGIN_VALIDATE(dimsPtr->d[0] >= 0 && dimsPtr->d[1] >= 0);
                }
                else // stride or dilation
                {
                    // Stride and dilation must be positive
                    PLUGIN_VALIDATE(dimsPtr->d[0] > 0 && dimsPtr->d[1] > 0);
                }
            }
        }

        auto* plugin
            = new ModulatedDeformableConvPluginDynamic(name, stride, padding, dilation, deformableGroup, group);
        plugin->setPluginNamespace(mNamespace.c_str());
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
        mNamespace = (libNamespace == nullptr) ? "" : libNamespace;
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
