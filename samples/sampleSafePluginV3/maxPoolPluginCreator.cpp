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

#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>

#include "NvInferSafePlugin.h"
#include "NvInferSafeRuntime.h"

#include "maxPoolPluginCreator.h"

namespace nvinfer1
{
namespace plugin
{

IPluginV3* MaxPoolCreator::createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    // The build phase and the deserialization phase are handled differently.
    if (phase == TensorRTPhase::kBUILD)
    {
        nvinfer1::PluginField const* fields{fc->fields};
        int32_t nbFields{fc->nbFields};

        std::vector<int32_t> kernelShape{};
        std::vector<int32_t> strides{};
        std::vector<int32_t> pads{};
        PoolingType pType{PoolingType::kMAX}; // Default to MAX pooling

        for (int32_t i{0}; i < nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "kernel_shape"))
            {
                int32_t const* const kernelShapeData{static_cast<int32_t const*>(fields[i].data)};
                for (int32_t j{0}; j < fields[i].length; ++j)
                {
                    kernelShape.push_back(kernelShapeData[j]);
                }
            }
            if (!strcmp(attrName, "strides"))
            {
                int32_t const* const stridesData{static_cast<int32_t const*>(fields[i].data)};
                for (int32_t j{0}; j < fields[i].length; ++j)
                {
                    strides.push_back(stridesData[j]);
                }
            }
            if (!strcmp(attrName, "pads"))
            {
                int32_t const* const padsData{static_cast<int32_t const*>(fields[i].data)};
                for (int32_t j{0}; j < fields[i].length; ++j)
                {
                    pads.push_back(padsData[j]);
                }
            }
            if (!strcmp(attrName, "pType"))
            {
                if (fields[i].data != nullptr)
                {
                    pType = *(static_cast<PoolingType const*>(fields[i].data));
                }
            }
        }

        PoolParameters params;
        params.pType = pType;
        params.Sx = strides[0];
        params.Sy = strides[1];
        params.Kx = kernelShape[0];
        params.Ky = kernelShape[1];
        params.Px = pads[0];
        params.Py = pads[1];

        MaxPoolPlugin* const plugin{new MaxPoolPlugin(params)};
        return plugin;
    }
    else if (phase == TensorRTPhase::kRUNTIME)
    {
        nvinfer1::PluginField const* fields{fc->fields};

        PoolParameters params{*(static_cast<PoolParameters const*>(fields[0].data))};

        MaxPoolPlugin* const plugin{new MaxPoolPlugin(params)};
        return plugin;
    }
    return nullptr;
}
} // namespace plugin
} // namespace nvinfer1
