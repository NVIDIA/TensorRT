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

#ifndef TRT_PYTHON_INVOCABLE_TRAITS_H
#define TRT_PYTHON_INVOCABLE_TRAITS_H

#include <concepts>
#include <tuple>
#include <type_traits>

//! \file Compile-time description of an invocable that implements an interface method. Plain C++, independent of
//! pybind11 and of TensorRT's internal headers, which the Python bindings cannot include.

namespace tensorrt::utils
{

//! Signature of an invocable that implements an interface method: the interface class, the return type, and the
//! decayed parameter types after the interface object itself. Accepts a pointer to a member function of the
//! interface; a pointer to a free function whose first parameter is the interface, or a functor or lambda of that
//! shape; or a function type of that shape, which names a signature directly. The same description serves the
//! trampoline and the pybind11 .def that exposes the method.
template <typename F>
struct InvocableTraits;

//! Traits of the free-function form R(Interface&, Params...).
template <typename Sig>
struct FreeInvocableTraits;

template <typename R, typename Self, typename... Params>
struct FreeInvocableTraits<R(Self&, Params...)>
{
    using Interface = std::remove_const_t<Self>;
    using Return = R;
    using Parameters = std::tuple<std::decay_t<Params>...>;
};

template <typename R, typename C, typename... Params>
struct InvocableTraits<R (C::*)(Params...)>
{
    using Interface = C;
    using Return = R;
    using Parameters = std::tuple<std::decay_t<Params>...>;
};

template <typename R, typename C, typename... Params>
struct InvocableTraits<R (C::*)(Params...) noexcept> : InvocableTraits<R (C::*)(Params...)>
{
};

template <typename R, typename C, typename... Params>
struct InvocableTraits<R (C::*)(Params...) const> : InvocableTraits<R (C::*)(Params...)>
{
};

template <typename R, typename C, typename... Params>
struct InvocableTraits<R (C::*)(Params...) const noexcept> : InvocableTraits<R (C::*)(Params...)>
{
};

//! A function type describes an override whose Python parameters differ from the C++ method's, e.g.,
//! IStreamWriter::write(void const*, int64_t) is write(bytes) in Python.
template <typename R, typename Self, typename... Params>
struct InvocableTraits<R(Self&, Params...)> : FreeInvocableTraits<R(Self&, Params...)>
{
};

template <typename R, typename... Args>
struct InvocableTraits<R (*)(Args...)> : FreeInvocableTraits<R(Args...)>
{
};

template <typename R, typename... Args>
struct InvocableTraits<R (*)(Args...) noexcept> : FreeInvocableTraits<R(Args...)>
{
};

//! The free-function signature of a functor's call operator, with its class stripped.
template <typename CallOperator>
struct CallOperatorTraits;

template <typename R, typename F, typename... Args>
struct CallOperatorTraits<R (F::*)(Args...) const> : FreeInvocableTraits<R(Args...)>
{
};

template <typename R, typename F, typename... Args>
struct CallOperatorTraits<R (F::*)(Args...) const noexcept> : FreeInvocableTraits<R(Args...)>
{
};

template <typename R, typename F, typename... Args>
struct CallOperatorTraits<R (F::*)(Args...)> : FreeInvocableTraits<R(Args...)>
{
};

template <typename R, typename F, typename... Args>
struct CallOperatorTraits<R (F::*)(Args...) noexcept> : FreeInvocableTraits<R(Args...)>
{
};

template <typename F>
    requires requires { &F::operator(); }
struct InvocableTraits<F> : CallOperatorTraits<decltype(&F::operator())>
{
};

//! \p TSelf is (or derives from) the interface that \p TInvocable implements a method of.
template <typename TSelf, typename TInvocable>
concept Implements = std::derived_from<TSelf, typename InvocableTraits<TInvocable>::Interface>;

//! \p Args are exactly the parameters of the interface method \p TInvocable implements, in the method's order.
//! Exact types, not convertible ones: a stream is a cudaStream_t, never an integer that happens to hold one.
template <typename TInvocable, typename... Args>
concept ArgumentsOf = std::same_as<std::tuple<std::decay_t<Args>...>, typename InvocableTraits<TInvocable>::Parameters>;

//! The Python name of an interface method, spelled once and shared by the pybind11 .def that exposes the method
//! and the trampoline that forwards to its override. Specialize per method: the pointer to member is the key.
template <auto Method>
inline constexpr char const* kPYTHON_NAME = nullptr;

//! A method that has a Python name registered through \p kPYTHON_NAME.
template <auto Method>
concept HasPythonName = kPYTHON_NAME<Method> != nullptr;

//! A pointer to (const) void: an untyped memory address, which the TensorRT Python API passes as an int.
template <typename T>
concept VoidPointer = std::is_pointer_v<T> && std::is_void_v<std::remove_pointer_t<T>>;

//! A no-op with the signature of the interface method \p Method, parameters decayed, for the pybind11 .def that
//! gives a method Python is expected to override its Python signature and docstring. The trampoline, not the stub,
//! is what the runtime calls, so the stub only returns a default value.
template <auto Method, typename Parameters = typename InvocableTraits<decltype(Method)>::Parameters>
struct DocStub;

template <auto Method, typename... Params>
struct DocStub<Method, std::tuple<Params...>>
{
    using Traits = InvocableTraits<decltype(Method)>;

    static typename Traits::Return call(typename Traits::Interface& /*self*/, Params... /*args*/)
    {
        if constexpr (!std::is_void_v<typename Traits::Return>)
        {
            return typename Traits::Return{};
        }
    }
};

} // namespace tensorrt::utils

#endif // TRT_PYTHON_INVOCABLE_TRAITS_H
