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

#include "utils.h"

namespace tensorrt
{
namespace utils
{

void issueDeprecationWarning(char const* useInstead)
{
    std::string msg{"Use " + std::string{useInstead} + " instead."};

    py::gil_scoped_acquire acquire{};
    PyErr_WarnEx(PyExc_DeprecationWarning, msg.c_str(), 1);
}

// Returns the size in bytes of the specified data type.
size_t size(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kBOOL: return 1;
    case nvinfer1::DataType::kUINT8: return 1;
    case nvinfer1::DataType::kFP8: return 1;
    case nvinfer1::DataType::kBF16: return 2;
    case nvinfer1::DataType::kINT4:
        break; // TRT-22011 - need to address sub-byte element size
    }
    return -1;
}

std::unique_ptr<py::dtype> nptype(nvinfer1::DataType type)
{
    auto const makeDtype = [](char const* typeStr) { return std::make_unique<py::dtype>(typeStr); };

    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: return makeDtype("f4");
    case nvinfer1::DataType::kHALF: return makeDtype("f2");
    case nvinfer1::DataType::kINT8: return makeDtype("i1");
    case nvinfer1::DataType::kINT32: return makeDtype("i4");
    case nvinfer1::DataType::kINT64: return makeDtype("i8");
    case nvinfer1::DataType::kBOOL: return makeDtype("b1");
    case nvinfer1::DataType::kUINT8: return makeDtype("u1");
    case nvinfer1::DataType::kFP8:
    case nvinfer1::DataType::kBF16:
    case nvinfer1::DataType::kINT4:
        return nullptr;
    }
    return nullptr;
}

nvinfer1::DataType type(py::dtype const& type)
{
    if (type.is(py::dtype("f4")))
    {
        return nvinfer1::DataType::kFLOAT;
    }
    else if (type.is(py::dtype("f2")))
    {
        return nvinfer1::DataType::kHALF;
    }
    else if (type.is(py::dtype("i8")))
    {
        return nvinfer1::DataType::kINT64;
    }
    else if (type.is(py::dtype("i4")))
    {
        return nvinfer1::DataType::kINT32;
    }
    else if (type.is(py::dtype("i1")))
    {
        return nvinfer1::DataType::kINT8;
    }
    else if (type.is(py::dtype("b1")))
    {
        return nvinfer1::DataType::kBOOL;
    }
    else if (type.is(py::dtype("u1")))
    {
        return nvinfer1::DataType::kUINT8;
    }
    int32_t constexpr kBITS_PER_BYTE{8};
    std::stringstream ss{};
    ss << "[TRT] [E] Could not implicitly convert NumPy data type: " << type.kind()
       << (type.itemsize() * kBITS_PER_BYTE) << " to TensorRT.";
    std::cerr << ss.str() << std::endl;
    PY_ASSERT_VALUE_ERROR(false, ss.str());
    return nvinfer1::DataType::kFLOAT;
}

void throwPyError(PyObject* type, std::string const& message)
{
    PyErr_SetString(type, message.data());
    throw py::error_already_set();
}

} // namespace utils
} // namespace tensorrt
