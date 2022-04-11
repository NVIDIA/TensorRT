/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "utils.h"

namespace tensorrt
{
namespace utils
{

void issueDeprecationWarning(const char* useInstead)
{
    std::string msg{"Use " + std::string{useInstead} + " instead."};

    py::gil_scoped_acquire acquire{};
    PyErr_WarnEx(PyExc_DeprecationWarning, msg.c_str(), 1);
}

// The following is a helper WAR to "throw py::index_error()", which results in an incompatibility
// with Tensorflow 2.5 and above--on Windows only--when Tensorflow is imported after TensorRT.
// The TF library fast_module_type.pyd hooks on to IndexErrors thrown through py::index_error()
// resulting in hangs at unpacking operations and out-of-bounds index accesses.
void throwPyIndexError(std::string message)
{
    PyErr_SetString(PyExc_IndexError, message.data());
    throw py::error_already_set();
}

} // namespace utils
} // namespace tensorrt
