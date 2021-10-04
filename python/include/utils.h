/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_PYTHON_UTILS_H
#define TRT_PYTHON_UTILS_H

// These headers must be included before pybind11.h as some dependencies are otherwise missing on Windows.
// clang-format off
#include "ForwardDeclarations.h"
// clang-format on

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "NvInfer.h"
#include <functional>
#include <iostream>
#include <string>

namespace tensorrt
{
namespace utils
{

namespace py = pybind11;

// Returns the size in bytes of the specified data type.
inline size_t size(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kBOOL: return 1;
    }
    return -1;
}

// Converts a TRT datatype to its corresponding numpy dtype.
inline py::dtype nptype(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: return py::dtype("f4");
    case nvinfer1::DataType::kHALF: return py::dtype("f2");
    case nvinfer1::DataType::kINT8: return py::dtype("i1");
    case nvinfer1::DataType::kINT32: return py::dtype("i4");
    case nvinfer1::DataType::kBOOL: return py::dtype("b1");
    }
    return py::dtype("unknown");
}

// Returns the TRT type corresponding to the specified numpy type.
inline nvinfer1::DataType type(const py::dtype& type)
{
    if (type.is(py::dtype("f4")))
    {
        return nvinfer1::DataType::kFLOAT;
    }
    else if (type.is(py::dtype("f2")))
    {
        return nvinfer1::DataType::kHALF;
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
    std::cout << "[ERROR] Unsupported numpy data type: " << type.kind() << type.itemsize() * 8
              << ". Cannot implicitly convert to tensorrt.Weights." << std::endl;
    throw std::invalid_argument{"Unsupported data type"};
}

// Return a numpy array (that doesn't own the data, but rather refers to it)
static const auto weights_to_numpy = [](const nvinfer1::Weights& self) {
    // The py::cast(self) allows us to return the buffer by reference rather than by copy.
    // See https://stackoverflow.com/questions/49181258/pybind11-create-numpy-view-of-data
    return py::array{nptype(self.type), self.count, self.values, py::cast(self)};
};

inline size_t volume(const nvinfer1::Dims& dims)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
}

// Method for calling the python function and returning the value (returned from python) used in cpp trampoline
// classes. Prints an error if no such method is overriden in python.
// T* must NOT be a trampoline class!
template <typename T>
py::function getOverride(const T* self, const std::string& overloadName, bool showWarning = true)
{
    py::function overload = py::get_override(self, overloadName.c_str());
    if (!overload && showWarning)
    {
        std::cerr << "Method: " << overloadName
                  << " was not overriden. Please provide an implementation for this method.";
    }
    return overload;
}

// Deprecation helpers
void issueDeprecationWarning(const char* useInstead);

// TODO: Figure out how to de-duplicate these two
template <typename RetVal, typename... Args>
struct DeprecatedFunc
{
    using Func = RetVal (*)(Args...);

    RetVal operator()(Args... args) const
    {
        issueDeprecationWarning(useInstead);
        return (*func)(std::forward<Args>(args)...);
    }

    const Func func;
    const char* useInstead;
};

template <typename RetVal, typename... Args>
constexpr auto deprecate(RetVal (*func)(Args...), const char* useInstead) -> DeprecatedFunc<RetVal, Args...>
{
    return DeprecatedFunc<RetVal, Args...>{func, useInstead};
}

template <bool isConst, typename RetVal, typename Cls, typename... Args>
struct DeprecatedMemberFunc
{
    using Func = typename std::conditional<isConst, RetVal (Cls::*)(Args...) const, RetVal (Cls::*)(Args...)>::type;

    RetVal operator()(Cls& self, Args... args) const
    {
        issueDeprecationWarning(useInstead);
        return (std::forward<Cls>(self).*func)(std::forward<Args>(args)...);
    }

    const Func func;
    const char* useInstead;
};

template <typename RetVal, typename Cls, typename... Args>
constexpr auto deprecateMember(RetVal (Cls::*func)(Args...) const, const char* useInstead)
    -> DeprecatedMemberFunc</*isConst=*/true, RetVal, Cls, Args...>
{
    return DeprecatedMemberFunc</*isConst=*/true, RetVal, Cls, Args...>{func, useInstead};
}

template <typename RetVal, typename Cls, typename... Args>
constexpr auto deprecateMember(RetVal (Cls::*func)(Args...), const char* useInstead)
    -> DeprecatedMemberFunc</*isConst=*/false, RetVal, Cls, Args...>
{
    return DeprecatedMemberFunc</*isConst=*/false, RetVal, Cls, Args...>{func, useInstead};
}

template <typename T>
void doNothingDel(const T& self)
{
    issueDeprecationWarning("del obj");
}

} // namespace utils
} // namespace tensorrt

#endif // TRT_PYTHON_UTILS_H
