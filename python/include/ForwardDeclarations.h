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

#ifndef TRT_PYTHON_FORWARD_DECLARATIONS_H
#define TRT_PYTHON_FORWARD_DECLARATIONS_H

// clang-format off
// Hack for missing declarations on Windows.
// These headers must be included before pybind11.h as some dependencies are missing.
#ifdef _MSC_VER
#include <cstdint>
using ssize_t = int64_t;
#endif // _MSC_VER
// clang-format on

#include <pybind11/pybind11.h>

// True if we are building full TensorRT bindings (as opposed to lean/dispatch bindings).
#if defined(tensorrt_EXPORTS)
#define EXPORT_ALL_BINDINGS 1
#else
#define EXPORT_ALL_BINDINGS 0
#endif

#include "NvInferRuntimeCommon.h"

// We need to avoid making copies of PluginField because it does not own any of it's members.
// When there are multiple PluginFields pointing to the same data in Python, bad things happen.
// Making this opaque allows us to create lists of PluginFields without creating unwanted copies.
PYBIND11_MAKE_OPAQUE(std::vector<nvinfer1::PluginField>)

namespace tensorrt
{
// Set some global namespace aliases.
namespace py = pybind11;
// This is for literal operators (like _a for default args)
using namespace pybind11::literals;
// Hack for situations where the C++ object does not own a member string/const char*.
// Cannot reference python strings, so we make a copy and keep it alive on the C++ side.
struct FallbackString
{
    FallbackString() = default;
    FallbackString(std::string const& other)
        : mData{other}
    {
    }
    FallbackString(py::str other)
        : mData{std::string(other)}
    {
    }
    const char* c_str() const
    {
        return mData.c_str();
    }
    const char* c_str()
    {
        return mData.c_str();
    }
    std::string mData{};
};

// Infer
void bindFoundationalTypes(py::module& m);
void bindPlugin(py::module& m);
#if EXPORT_ALL_BINDINGS
void bindInt8(py::module& m);
void bindGraph(py::module& m);
void bindAlgorithm(py::module& m);
#endif
void bindCore(py::module& m);
// Parsers
#if EXPORT_ALL_BINDINGS
void bindOnnx(py::module& m);
#endif
} // namespace tensorrt

#endif // TRT_PYTHON_FORWARD_DECLARATIONS_H
