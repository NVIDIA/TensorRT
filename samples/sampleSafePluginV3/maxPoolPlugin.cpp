/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

#include "NvInferSafePlugin.h"
#include "NvInferSafeRuntime.h"

#include "maxPoolKernel.h"
#include "maxPoolPlugin.h"
#include "safeCommon.h"

namespace nvinfer1
{
namespace plugin
{

IPluginV3* MaxPoolPlugin::clone() noexcept
{
    try
    {
        return new MaxPoolPlugin{mParams};
    }
    catch (std::exception const& e)
    {
        std::cerr << "Error cloning MaxPoolPlugin: " << e.what() << std::endl;
        return nullptr;
    }
}

IPluginCapability* MaxPoolPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    switch (type)
    {
    case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneSafeBuildMSS*>(this);
    case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneSafeRuntime*>(this);
    case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneSafeCore*>(this);
    }
    return nullptr;
}

int32_t MaxPoolPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t MaxPoolPlugin::configurePlugin(
    TensorDescriptor const* in, int32_t nbInputs, TensorDescriptor const* out, int32_t nbOutputs) noexcept
{
    SAFE_ASSERT(in && nbInputs == 1);
    SAFE_ASSERT(out && nbOutputs == 1);
    SAFE_ASSERT(in[0].dataType == out[0].dataType);

    mParams.dtype = in[0].dataType;
    mParams.C = in[0].shape.d[1];
    mParams.H = in[0].shape.d[2];
    mParams.W = in[0].shape.d[3];
    mParams.H_out = out[0].shape.d[2];
    mParams.W_out = out[0].shape.d[3];

    switch (mParams.dtype)
    {
    case nvinfer1::DataType::kINT8: mParams.dtypeBytes = 1; break;
    case nvinfer1::DataType::kHALF: mParams.dtypeBytes = 2; break;
    case nvinfer1::DataType::kFLOAT: mParams.dtypeBytes = 4; break;
    default:
    {
        mRecorder->reportError(
            nvinfer1::ErrorCode::kFAILED_EXECUTION, "Failed to execute due to unavailable precision.");
        return 1;
    }
    }
    return 0;
}

bool MaxPoolPlugin::supportsFormatCombination(
    int32_t pos, TensorDescriptor const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // For this method inputs are numbered 0..(nbInputs-1) and outputs are
    // numbered nbInputs..(nbInputs+nbOutputs-1). Using this numbering, pos is
    // an index into InOut, where 0 <= pos < nbInputs+nbOutputs.
    if (!(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs))
    {
        return false;
    }

    // Check if the data type is supported
    bool const supportedDataType = (inOut[pos].dataType == nvinfer1::DataType::kFLOAT)
        || (inOut[pos].dataType == nvinfer1::DataType::kHALF) || (inOut[pos].dataType == nvinfer1::DataType::kINT8);

    if (!supportedDataType)
    {
        return false;
    }

    // Check if the format is supported (no vectorization)
    bool const supportedFormat = (inOut[pos].vectorizedDim == -1);

    if (!supportedFormat)
    {
        return false;
    }

    // For output tensors, ensure they match the input data type
    if (pos >= nbInputs) // This is an output tensor
    {
        // Output must match input data type
        return inOut[pos].dataType == inOut[0].dataType && inOut[pos].vectorizedDim == inOut[0].vectorizedDim;
    }

    // For input tensors, just check that the type and format are supported
    return true;
}

int32_t MaxPoolPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    SAFE_ASSERT(inputTypes && nbInputs == 1);
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t MaxPoolPlugin::getOutputShapes(
    Dims const* inputs, int32_t nbInputs, Dims* outputs, int32_t nbOutputs) const noexcept
{
    // Empty, will not be called
    return 0;
}

int32_t MaxPoolPlugin::getSymbolicOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs* outputs,
    int32_t nbOutputs, IExprBuilder& exprBuilder) const noexcept
{
    SAFE_ASSERT(inputs && nbInputs == 1 && inputs[0U].nbDims == 4);
    SAFE_ASSERT(outputs && nbOutputs == 1);
    outputs[0U].nbDims = inputs[0U].nbDims;
    outputs[0U].d[0U] = inputs[0U].d[0U];
    outputs[0U].d[1U] = inputs[0U].d[1U];

    // Calculate height: (input_height + pad_y*2 - kernel_y) / stride_y + 1
    auto const* padY2
        = exprBuilder.operation(DimensionOperation::kPROD, *exprBuilder.constant(mParams.Py), *exprBuilder.constant(2));
    SAFE_ASSERT(padY2);
    SAFE_ASSERT(inputs[0U].d[2U]);
    auto const* inputPlusPadY = exprBuilder.operation(DimensionOperation::kSUM, *inputs[0U].d[2U], *padY2);
    SAFE_ASSERT(inputPlusPadY);
    auto const* heightMinusKernel
        = exprBuilder.operation(DimensionOperation::kSUB, *inputPlusPadY, *exprBuilder.constant(mParams.Ky));
    SAFE_ASSERT(heightMinusKernel);
    auto const* heightDivStride
        = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *heightMinusKernel, *exprBuilder.constant(mParams.Sy));
    SAFE_ASSERT(heightDivStride);
    outputs[0U].d[2U] = exprBuilder.operation(DimensionOperation::kSUM, *heightDivStride, *exprBuilder.constant(1));
    SAFE_ASSERT(outputs[0U].d[2U]);

    // Calculate width: (input_width + pad_x*2 - kernel_x) / stride_x + 1
    auto const* padX2
        = exprBuilder.operation(DimensionOperation::kPROD, *exprBuilder.constant(mParams.Px), *exprBuilder.constant(2));
    SAFE_ASSERT(padX2);
    SAFE_ASSERT(inputs[0U].d[3U]);
    auto const* inputPlusPadX = exprBuilder.operation(DimensionOperation::kSUM, *inputs[0U].d[3U], *padX2);
    SAFE_ASSERT(inputPlusPadX);
    auto const* widthMinusKernel
        = exprBuilder.operation(DimensionOperation::kSUB, *inputPlusPadX, *exprBuilder.constant(mParams.Kx));
    SAFE_ASSERT(widthMinusKernel);
    auto const* widthDivStride
        = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *widthMinusKernel, *exprBuilder.constant(mParams.Sx));
    SAFE_ASSERT(widthDivStride);
    outputs[0U].d[3U] = exprBuilder.operation(DimensionOperation::kSUM, *widthDivStride, *exprBuilder.constant(1));
    SAFE_ASSERT(outputs);

    return 0;
}

size_t MaxPoolPlugin::getWorkspaceSize(
    TensorDescriptor const* inputs, int32_t nbInputs, TensorDescriptor const* outputs, int32_t nbOutputs) const noexcept
{
    // MaxPool doesn't need any workspace memory
    return 0;
}

} // namespace plugin
} // namespace nvinfer1

extern "C" nvinfer2::safe::consistency::IPluginChecker* getPluginChecker(char const* name)
{
    if (name && strcmp(name, "MaxPoolPlugin1") == 0)
    {
        auto checker = std::make_unique<nvinfer2::safe::consistency::MaxPoolPluginChecker>();
        return checker.release();
    }
    return nullptr;
}
