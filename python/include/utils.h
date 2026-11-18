/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "address.h"
#include "invocableTraits.h"
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>

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

namespace tensorrt::utils
{
//! pybind11 caster for an opaque CUDA handle such as cudaStream_t or cudaEvent_t, which the TensorRT Python API
//! passes as a plain int. Letting pybind11 convert it means a method taking the handle can be bound directly
//! instead of through a lambda that re-declares its parameters and reinterpret_casts an integer.
//!
//! \tparam Handle the pointer type of the handle, e.g. cudaStream_t.
template <typename Handle>
struct OpaqueHandleCaster
{
    static_assert(std::is_pointer_v<Handle>);

    //! Accepts any object usable as an index (int, numpy integer, a handle type with __index__), as pybind11's
    //! own integer caster does; rejects None, floats, and negative values.
    bool load(pybind11::handle src, bool)
    {
        if (!src || src.is_none())
        {
            return false;
        }
        auto index = pybind11::reinterpret_steal<pybind11::object>(PyNumber_Index(src.ptr()));
        if (!index)
        {
            PyErr_Clear();
            return false;
        }
        // Unsigned conversion so a negative value raises instead of wrapping, as PyLong_AsVoidPtr would let it.
        auto const bits = static_cast<uintptr_t>(PyLong_AsUnsignedLongLong(index.ptr()));
        if (PyErr_Occurred())
        {
            PyErr_Clear();
            return false;
        }
        value = reinterpret_cast<Handle>(bits);
        return true;
    }

    static pybind11::handle cast(Handle src, pybind11::return_value_policy, pybind11::handle)
    {
        return PyLong_FromVoidPtr(src);
    }

    template <typename T>
    using cast_op_type = Handle&;
    explicit operator Handle&()
    {
        return value;
    }
    static constexpr auto name = pybind11::detail::const_name("int");

private:
    Handle value{nullptr};
};
} // namespace tensorrt::utils

//! Specializing pybind11::detail::type_caster is pybind11's documented mechanism for custom conversions; see
//! https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html. The specializations are on the pointee
//! types because pybind11 keys casters on the pointee. Every binding translation unit includes this header, so no
//! TU ever instantiates the default caster for these types.
template <>
struct pybind11::detail::type_caster<std::remove_pointer_t<cudaStream_t>>
    : tensorrt::utils::OpaqueHandleCaster<cudaStream_t>
{
};

template <>
struct pybind11::detail::type_caster<std::remove_pointer_t<cudaEvent_t>>
    : tensorrt::utils::OpaqueHandleCaster<cudaEvent_t>
{
};

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

// https://nvbugs/3479811 Create a wrapper for C++ to python throw
[[noreturn]] void throwPyError(PyObject* type, std::string const& message = "python error");

//! Find the Python override named \p overloadName of the object \p self is part of, printing an error if there is
//! none and \p showWarning. The caller passes the py::gil_scoped_acquire it holds as proof that it has the GIL,
//! which py::get_override requires; it is taken by non-const lvalue reference so a temporary (which would destruct
//! before use) cannot bind to it. \p T may be the trampoline itself: pybind11 registers a trampoline as the class
//! it is the alias of, so the lookup resolves to the instance's own class, at any depth of the interface hierarchy.
//! Casting to the interface that declares the method would not: IGpuAllocator::reallocate on an IGpuAsyncAllocator
//! finds no instance registered as an IGpuAllocator.
template <typename T>
[[nodiscard]] py::function getOverride(
    py::gil_scoped_acquire& /*gil*/, T const* self, char const* overloadName, bool showWarning = true)
{
    py::function overload = py::get_override(self, overloadName);
    if (!overload && showWarning)
    {
        std::cerr << "Method: " << overloadName
                  << " was not overriden. Please provide an implementation for this method." << std::endl;
    }
    return overload;
}

//! Like getOverride, but throws a Python RuntimeError naming \p overloadName instead of returning a falsy
//! py::function when there is no override. Use where the caller requires the override to exist.
//!
//! \post the returned function is truthy.
template <typename T>
[[nodiscard]] py::function requireOverride(py::gil_scoped_acquire& gil, T const* self, char const* overloadName)
{
    py::function overload = getOverride(gil, self, overloadName);
    if (!overload)
    {
        throwPyError(PyExc_RuntimeError, std::string("no implementation provided for ") + overloadName + "()");
    }
    return overload;
}

//! Convert one argument of an interface method into the form its Python override expects.
//!
//! Everything goes through pybind11's casters, including cudaStream_t and cudaEvent_t via OpaqueHandleCaster,
//! except void pointers: they are device or host memory addresses, which the API passes as ints, and pybind11's
//! own void* caster would produce a capsule instead. void* is not a type of our own, so that rule cannot be a caster.
//! A value that already is a Python object, such as the bytes IStreamWriter::write passes, goes through as is.
template <typename T>
[[nodiscard]] py::object toPythonArg(T const& arg)
{
    if constexpr (VoidPointer<T>)
    {
        return py::int_(addressToPyInt(arg));
    }
    else if constexpr (std::derived_from<T, py::object>)
    {
        return arg;
    }
    else
    {
        return py::cast(arg);
    }
}

//! What invokeOverride yields for a method returning \p R: monostate stands in for void, so an override that ran is
//! still told apart from one that did not.
template <typename R>
using OverrideResult = std::optional<std::conditional_t<std::is_void_v<R>, std::monostate, R>>;

//! Convert a Python override's return value to the interface method's return type. The counterpart of
//! toPythonArg: void pointers come back as ints, everything else through pybind11's casters, and a void method's
//! result is dropped unexamined.
template <typename R>
[[nodiscard]] typename OverrideResult<R>::value_type fromPythonResult([[maybe_unused]] py::object const& result)
{
    if constexpr (std::is_void_v<R>)
    {
        return std::monostate{};
    }
    else if constexpr (VoidPointer<R>)
    {
        return reinterpret_cast<R>(result.cast<uintptr_t>());
    }
    else
    {
        return result.cast<R>();
    }
}

//! Invoke the Python override named \p pyName on \p self. \p TInvocable is the signature the override has: the
//! type of a pointer to the interface method when Python receives the method's own parameters, or a function
//! type R(Interface&, Params...) when the Python parameters differ from the C++ ones.
//!
//! Python receives \p args positionally, so they must be exactly the signature's parameters, in order: a wrong
//! count, a wrong type, or two arguments swapped fails the constraints instead of reaching Python as a
//! plausible-looking int (https://nvbugs/6157565). \p self is normally the trampoline; see getOverride.
//!
//! \return the override's result, monostate for a void signature; nullopt if there is no override (printed as an
//! error when \p warnIfMissing), or if the call threw or returned a value of the wrong type, which are always
//! printed.
template <typename TInvocable, typename TSelf, typename... Args>
    requires Implements<TSelf, TInvocable> && ArgumentsOf<TInvocable, Args...>
[[nodiscard]] OverrideResult<typename InvocableTraits<TInvocable>::Return> invokeOverride(
    TSelf* self, char const* pyName, bool warnIfMissing, Args const&... args) noexcept
{
    using Traits = InvocableTraits<TInvocable>;
    using Return = typename Traits::Return;
    try
    {
        py::gil_scoped_acquire gil{};
        py::function pyOverride = getOverride(gil, self, pyName, warnIfMissing);
        if (!pyOverride)
        {
            return std::nullopt;
        }
        py::object result = pyOverride(toPythonArg(args)...);
        try
        {
            return fromPythonResult<Return>(result);
        }
        catch (py::cast_error const&)
        {
            std::cerr << "[ERROR] Return value of " << pyName << "() could not be interpreted as "
                      << (std::is_pointer_v<Return> ? "an int" : "the expected type") << std::endl;
        }
    }
    catch (std::exception const& e)
    {
        std::cerr << "[ERROR] Exception caught in " << pyName << "(): " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[ERROR] Exception caught in " << pyName << "()" << std::endl;
    }
    return std::nullopt;
}

//! Invoke the Python override of the interface method \p Method on \p self, passing \p args, which must be
//! exactly the method's parameters. The override is looked up as kPYTHON_NAME<Method>, the same string the
//! pybind11 .def uses, so the two cannot drift apart. A missing override is printed as an error.
//!
//! \return the override's result, or nullopt if the override is missing or failed, as for the overload above.
template <auto Method, typename TSelf, typename... Args>
    requires HasPythonName<Method> && Implements<TSelf, decltype(Method)> && ArgumentsOf<decltype(Method), Args...>
[[nodiscard]] OverrideResult<typename InvocableTraits<decltype(Method)>::Return> invokeOverride(
    TSelf* self, Args const&... args) noexcept
{
    return invokeOverride<decltype(Method)>(self, kPYTHON_NAME<Method>, true, args...);
}

//! Whether \p self's Python class overrides the interface method \p Method, for methods a class may omit.
template <auto Method, typename TSelf>
    requires HasPythonName<Method> && Implements<TSelf, decltype(Method)>
[[nodiscard]] bool hasOverride(TSelf const* self) noexcept
{
    try
    {
        py::gil_scoped_acquire gil{};
        return static_cast<bool>(getOverride(gil, self, kPYTHON_NAME<Method>, false));
    }
    catch (...)
    {
        return false;
    }
}

// Deprecation helpers

//! Raise a Python DeprecationWarning that directs the caller to \p useInstead. The Deprecated* wrappers below call
//! it each time a deprecated binding is invoked.
void issueDeprecationWarning(char const* useInstead);

//! Warn via issueDeprecationWarning, then invoke \p func on \p args, which for a member-pointer \p func starts
//! with the instance. The Deprecated specializations below are just this behind a call operator whose signature
//! matches \p func's, since pybind11 needs a concrete, non-template operator() to derive a binding's signature.
template <typename Func, typename... Args>
auto invokeDeprecated(char const* useInstead, Func func, Args&&... args)
{
    issueDeprecationWarning(useInstead);
    return std::invoke(func, std::forward<Args>(args)...);
}

template <typename Func>
struct Deprecated;

template <typename RetVal, typename... Args>
struct Deprecated<RetVal (*)(Args...)>
{
    RetVal operator()(Args... args) const
    {
        return invokeDeprecated(useInstead, func, std::forward<Args>(args)...);
    }

    RetVal (*const func)(Args...);
    char const* const useInstead;
};

template <typename RetVal, typename... Args>
struct Deprecated<RetVal (*)(Args...) noexcept>
{
    RetVal operator()(Args... args) const
    {
        return invokeDeprecated(useInstead, func, std::forward<Args>(args)...);
    }

    RetVal (*const func)(Args...) noexcept;
    char const* const useInstead;
};

template <typename RetVal, typename Cls, typename... Args>
struct Deprecated<RetVal (Cls::*)(Args...)>
{
    RetVal operator()(Cls& self, Args... args) const
    {
        return invokeDeprecated(useInstead, func, self, std::forward<Args>(args)...);
    }

    RetVal (Cls::*const func)(Args...);
    char const* const useInstead;
};

template <typename RetVal, typename Cls, typename... Args>
struct Deprecated<RetVal (Cls::*)(Args...) noexcept>
{
    RetVal operator()(Cls& self, Args... args) const
    {
        return invokeDeprecated(useInstead, func, self, std::forward<Args>(args)...);
    }

    RetVal (Cls::*const func)(Args...) noexcept;
    char const* const useInstead;
};

template <typename RetVal, typename Cls, typename... Args>
struct Deprecated<RetVal (Cls::*)(Args...) const>
{
    RetVal operator()(Cls const& self, Args... args) const
    {
        return invokeDeprecated(useInstead, func, self, std::forward<Args>(args)...);
    }

    RetVal (Cls::*const func)(Args...) const;
    char const* const useInstead;
};

template <typename RetVal, typename Cls, typename... Args>
struct Deprecated<RetVal (Cls::*)(Args...) const noexcept>
{
    RetVal operator()(Cls const& self, Args... args) const
    {
        return invokeDeprecated(useInstead, func, self, std::forward<Args>(args)...);
    }

    RetVal (Cls::*const func)(Args...) const noexcept;
    char const* const useInstead;
};

template <typename Func>
constexpr auto deprecate(Func func, char const* useInstead) -> Deprecated<Func>
{
    return Deprecated<Func>{func, useInstead};
}

template <typename T>
constexpr auto deprecateInTrtRtxOnly(T&& func, char const* /*unused*/) -> T&&
{
    return std::forward<T>(func);
}

template <typename T>
void doNothingDel(T const& self)
{
    issueDeprecationWarning("del obj");
}

//! \brief Validate a Dims returned by a TensorRT API.
//!
//! Several TensorRT APIs report failure by returning an invalid Dims (nbDims < 0) rather than raising an
//! error. Surface that as a Python exception so callers get a clear failure instead of an unusable object.
//!
//! \param dims The Dims to validate.
//! \param message The error message to raise when \p dims is invalid.
//! \return \p dims unchanged when it is valid.
[[nodiscard]] inline nvinfer1::Dims checkDims(nvinfer1::Dims const& dims, std::string const& message)
{
    if (dims.nbDims < 0)
    {
        throwPyError(PyExc_RuntimeError, message);
    }
    return dims;
}

//! \brief Wrap a no-argument Dims getter so an invalid result raises a Python exception.
//!
//! \param getter Pointer to the member function being wrapped.
//! \param name Human-readable name of the queried value, used in the error message.
//! \return A callable suitable for binding as a pybind11 property getter.
template <typename Cls>
[[nodiscard]] auto throwingDimsGetter(nvinfer1::Dims (Cls::*getter)() const noexcept, std::string name)
{
    return [getter, name](Cls& self) { return checkDims(std::invoke(getter, self), "Could not get " + name + "."); };
}

//! \brief Wrap a name-keyed Dims getter so an invalid result raises a Python exception.
//!
//! \param getter Pointer to the member function being wrapped.
//! \param what Human-readable description of the queried value, used in the error message.
//! \return A callable suitable for binding as a pybind11 method.
template <typename Cls>
[[nodiscard]] auto throwingNamedDimsGetter(nvinfer1::Dims (Cls::*getter)(char const*) const noexcept, std::string what)
{
    return [getter, what](Cls& self, char const* name) {
        return checkDims(std::invoke(getter, self, name),
            "Could not get " + what + " for tensor '" + std::string{name} + "'. Is the tensor name correct?");
    };
}

//! \brief Wrap a no-argument Dims getter so an invalid result maps to None.
//!
//! Some TensorRT getters return an invalid Dims (nbDims < 0) to signal a legitimate state, e.g. the value is
//! supplied dynamically through an input tensor rather than statically. Map that to None rather than exposing
//! an unusable Dims object.
//!
//! \param getter Pointer to the member function being wrapped.
//! \return A callable suitable for binding as a pybind11 property getter.
template <typename Cls>
[[nodiscard]] auto optionalDimsGetter(nvinfer1::Dims (Cls::*getter)() const noexcept)
{
    return [getter](Cls& self) -> py::object {
        nvinfer1::Dims const dims = std::invoke(getter, self);
        if (dims.nbDims < 0)
        {
            return py::none();
        }
        return py::cast(dims);
    };
}

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
