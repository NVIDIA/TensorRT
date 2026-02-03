/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "maxPoolPluginRuntime.h"
#include "safeCommon.h"

namespace nvinfer1
{
namespace plugin
{

MaxPoolPluginRuntime::MaxPoolPluginRuntime(PoolParameters const& params)
    : mParams{params}
    , mRecorder{nullptr}
{
    initFieldsToSerialize();
}

void MaxPoolPluginRuntime::initFieldsToSerialize()
{
    // Serialize MaxPoolParameters.
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(
        nvinfer1::PluginField("parameters", &mParams, PluginFieldType::kUNKNOWN, sizeof(PoolParameters)));
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
}

IPluginCapability* MaxPoolPluginRuntime::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    switch (type)
    {
    case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneSafeRuntime*>(this);
    case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneSafeCore*>(this);
    case PluginCapabilityType::kBUILD: return nullptr; // MaxPoolPluginRuntime does not support build capability
    }
    return nullptr;
}

IPluginV3* MaxPoolPluginRuntime::clone() noexcept
{
    try
    {
        return new MaxPoolPluginRuntime{mParams};
    }
    catch (...)
    {
        return nullptr;
    }
}

AsciiChar const* MaxPoolPluginRuntime::getPluginName() const noexcept
{
    return kMAX_POOL_PLUGIN_NAME;
}

AsciiChar const* MaxPoolPluginRuntime::getPluginVersion() const noexcept
{
    return kMAX_POOL_PLUGIN_VERSION;
}

AsciiChar const* MaxPoolPluginRuntime::getPluginNamespace() const noexcept
{
    return kMAX_POOL_PLUGIN_NAMESPACE;
}

int32_t MaxPoolPluginRuntime::enqueue(TensorDescriptor const* inputDesc, TensorDescriptor const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        int32_t ret = 0;
        SAFE_ASSERT(inputs);
        SAFE_ASSERT(outputs);
        int32_t batch_size = inputDesc[0].shape.d[0];
        SAFE_ASSERT(batch_size > 0);
        switch (mParams.dtype)
        {
        case nvinfer1::DataType::kFLOAT:
            ret = maxPoolFloat(stream, batch_size, mParams.C, mParams.H, mParams.W, inputs[0], outputs[0], mParams.Kx,
                mParams.Sx, mParams.Px);
            break;
        case nvinfer1::DataType::kHALF:
            ret = maxPoolHalf(stream, batch_size, mParams.C, mParams.H, mParams.W, inputs[0], outputs[0], mParams.Kx,
                mParams.Sx, mParams.Px);
            break;
        case nvinfer1::DataType::kINT8:
            ret = maxPoolInt8(stream, batch_size, mParams.C, mParams.H, mParams.W, inputs[0], outputs[0], mParams.Kx,
                mParams.Sx, mParams.Px);
            break;
        default:
        {
            if (mRecorder)
            {
                mRecorder->reportError(
                    nvinfer1::ErrorCode::kFAILED_EXECUTION, "Failed to execute due to unavailable precision.");
            }
            ret = 1;
        }
        }
        return ret;
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << '\n';
    }
    return -1;
}

int32_t MaxPoolPluginRuntime::initResource(ISafePluginResourceContext const* context) noexcept
{
    return 0;
}

PluginFieldCollection const* MaxPoolPluginRuntime::getFieldsToSerialize() noexcept
{
    return &mFCToSerialize;
}

ISafeRecorder* MaxPoolPluginRuntime::getSafeRecorder() const noexcept
{
    return mRecorder;
}

void MaxPoolPluginRuntime::setSafeRecorder(ISafeRecorder& recorder) noexcept
{
    mRecorder = &recorder;
}
} // namespace plugin
} // namespace nvinfer1
