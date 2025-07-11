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
#include <memory>
#include <string>

#if defined(__GLIBC__)
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

#define CUDA_LIB_NAME "cuda"

//! Macro evaluates the expression \p EXPR, and if not equal to `CUDA_SUCCESS`, logs to `std::cerr` and evaluates return
//! RET_ON_FAIL.
#define CUDA_CALL_WITH_RET(EXPR, RET_ON_FAIL)                                                                          \
    if (CUresult retCode = (EXPR); retCode != CUDA_SUCCESS)                                                            \
    {                                                                                                                  \
        std::cerr << "[ERROR] Failed to " << #EXPR << " with error " << retCode << std::endl;                          \
        return RET_ON_FAIL;                                                                                            \
    }

namespace tensorrt
{
namespace utils
{

namespace py = pybind11;

// Wrapper function for dlopen/dlclose/dlsym, support both Windows and Linux
//! \brief Attempts to open the library
//!
//! \param libName. The returned library handle may be nullptr on failure and must be closed with `dllClose`.
[[nodiscard]] void* nvdllOpen(char const* libName);

//! Closes a library opened with `nvdllOpen`.
void dllClose(void* handle);

//! \brief get symbol from the library
//!
//! \param name in a dll
//! \param handle, loaded by `nvdllOpen`.
//!
//! \return the pointer to the symbol named
[[nodiscard]] void* dllGetSym(void* handle, char const* name);

// Returns the size in bytes of the specified data type.
size_t size(nvinfer1::DataType type);

// Converts a TRT datatype to its corresponding numpy dtype.
// Returns nullptr if the type could not be converted to NumPy.
std::unique_ptr<py::dtype> nptype(nvinfer1::DataType type);

// Returns the TRT type corresponding to the specified numpy type.
nvinfer1::DataType type(py::dtype const& type);

// Return a numpy array (that doesn't own the data, but rather refers to it)
static const auto weights_to_numpy = [](nvinfer1::Weights const& self) -> py::object {
    // The py::cast(self) allows us to return the buffer by reference rather than by copy.
    // See https://stackoverflow.com/questions/49181258/pybind11-create-numpy-view-of-data
    auto const npType = nptype(self.type);
    if (npType)
    {
        return py::array{*npType, self.count, self.values, py::cast(self)};
    }
    return py::cast(self);
};

inline int64_t volume(nvinfer1::Dims const& dims)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}

// Method for calling the python function and returning the value (returned from python) used in cpp trampoline
// classes. Prints an error if no such method is overriden in python.
// T* must NOT be a trampoline class!
template <typename T>
py::function getOverride(const T* self, std::string const& overloadName, bool showWarning = true)
{
    py::function overload = py::get_override(self, overloadName.c_str());
    if (!overload && showWarning)
    {
        std::cerr << "Method: " << overloadName
                  << " was not overriden. Please provide an implementation for this method." << std::endl;
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
    using Func = typename std::conditional_t<isConst, RetVal (Cls::*)(Args...) const, RetVal (Cls::*)(Args...)>;

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

// https://nvbugs/3479811 Create a wrapper for C++ to python throw
void throwPyError(PyObject* type, std::string const& message = "python error");

} // namespace utils

#define PY_ASSERT_RUNTIME_ERROR(assertion, msg)                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(assertion))                                                                                              \
        {                                                                                                              \
            utils::throwPyError(PyExc_RuntimeError, msg);                                                              \
        }                                                                                                              \
    } while (false)

#define PY_ASSERT_INDEX_ERROR(assertion)                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(assertion))                                                                                              \
        {                                                                                                              \
            utils::throwPyError(PyExc_IndexError, "Out of bounds");                                                    \
        }                                                                                                              \
    } while (false)

#define PY_ASSERT_VALUE_ERROR(assertion, msg)                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(assertion))                                                                                              \
        {                                                                                                              \
            utils::throwPyError(PyExc_ValueError, msg);                                                                \
        }                                                                                                              \
    } while (false)

} // namespace tensorrt

#endif // TRT_PYTHON_UTILS_H
