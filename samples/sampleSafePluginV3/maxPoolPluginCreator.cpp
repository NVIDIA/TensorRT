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
#include <string_view>

#include "NvInferSafePlugin.h"
#include "NvInferSafeRuntime.h"

#include "maxPoolPluginCreator.h"

namespace nvinfer1::plugin
{

namespace
{

using namespace std::string_view_literals;

//! Like C++23's `container.append_range(std::span(data, count))`.
template <typename Container, typename T>
void appendRange(Container& container, T* data, int64_t count)
{
    container.insert(container.end(), data, data + count);
}

//! \return PoolParameters parsed from the given `PluginField`s.
//! \param fields, nbFields: A span of fields.
[[nodiscard]] PoolParameters parsePoolParameters(nvinfer1::PluginField const* const fields, int32_t const nbFields)
{
    std::vector<int32_t> kernelShape;
    std::vector<int32_t> strides;
    std::vector<int32_t> pads;
    PoolingType pType{PoolingType::kMAX}; // Default to MAX pooling
    for (int32_t i{0}; i < nbFields; ++i)
    {
        auto const& field = fields[i];
        std::string_view const attrName = field.name;
        if (attrName == "kernel_shape"sv)
        {
            appendRange(kernelShape, static_cast<int32_t const*>(field.data), field.length);
        }
        else if (attrName == "strides"sv)
        {
            appendRange(strides, static_cast<int32_t const*>(field.data), field.length);
        }
        else if (attrName == "pads"sv)
        {
            appendRange(pads, static_cast<int32_t const*>(field.data), field.length);
        }
        else if (attrName == "pType"sv && field.data != nullptr)
        {
            pType = *static_cast<PoolingType const*>(field.data);
        }
    }

    PoolParameters result{};
    result.pType = pType;
    result.Sx = strides.at(0);
    result.Sy = strides.at(1);
    result.Kx = kernelShape.at(0);
    result.Ky = kernelShape.at(1);
    result.Px = pads.at(0);
    result.Py = pads.at(1);
    return result;
}
} // namespace

IPluginV3* MaxPoolCreator::createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    // The build phase and the deserialization phase are handled differently.
    switch (phase)
    {
    case TensorRTPhase::kBUILD:
        return std::make_unique<MaxPoolPlugin>(parsePoolParameters(fc->fields, fc->nbFields)).release();
    case TensorRTPhase::kRUNTIME:
        return std::make_unique<MaxPoolPlugin>(*static_cast<PoolParameters const*>(fc->fields[0].data)).release();
    }
    return nullptr;
}
} // namespace nvinfer1::plugin
