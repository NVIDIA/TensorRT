/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// This file contains top level bindings and defines the whole TRT Python package.
#include "ForwardDeclarations.h"
#include "NvInfer.h"
#include "pyTensorRTDoc.h"
#include <pybind11/stl_bind.h>

namespace tensorrt
{
PYBIND11_MODULE(TENSORRT_MODULE, m)
{
    // Python strings can be automatically converted to FallbackStrings,
    // whose lifetime is tied to TRT objects that reference, but do not own, a string.
    // See ForwardDeclarations.h for more information about FallbackString.
    // Note that we cannot allow Python to deallocate this string, hence the py::nodelete.
    //
    // Important: *All* Python enum and class interfaces exported by this module *must* be
    // declared with py::module_local() scope, so that multiple TRT runtime modules
    // (e.g. tensorrt_lean and tensorrt_dispatch) may be imported by the same Python script
    // without conflicts.
    //
    // See https://pybind11.readthedocs.io/en/stable/advanced/classes.html#module-local-class-bindings
    // for more information.
    py::class_<FallbackString, std::unique_ptr<FallbackString, py::nodelete>>(m, "FallbackString", py::module_local())
        .def(py::init<std::string>())
        .def(py::init<py::str>());
    py::implicitly_convertible<std::string, FallbackString>();
    py::implicitly_convertible<py::str, FallbackString>();

    // Make it so that we can use lists of PluginFields without creating unwanted copies.
    // This is declared opaque in ForwardDeclarations.h
    py::bind_vector<std::vector<nvinfer1::PluginField>>(m, "PluginFieldCollection");

    // Order matters here - Dependencies must be resolved properly!
    // TODO: Maybe use actual forward declarations and define functions later.
    bindFoundationalTypes(m);
    bindPlugin(m);
#if EXPORT_ALL_BINDINGS
    bindInt8(m);
    bindGraph(m);
    bindAlgorithm(m);
#endif
    bindCore(m);
#if EXPORT_ALL_BINDINGS
    // Parsers
    bindOnnx(m);
#endif
}
} // namespace tensorrt
