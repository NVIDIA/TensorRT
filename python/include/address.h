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

#ifndef TRT_PYTHON_ADDRESS_H
#define TRT_PYTHON_ADDRESS_H

#include "invocableTraits.h"

#include <cstdint>
#include <span>
#include <vector>

//! \file The one place the Python bindings turn an untyped memory address into the integer the Python API passes.
//! Plain C++, independent of pybind11.
namespace tensorrt::utils
{

//! The int the Python API uses for the address \p pointer holds. Only void pointers qualify: a typed pointer is
//! not an address the API hands to Python, and a handle such as cudaStream_t goes through its own caster.
[[nodiscard]] uintptr_t addressToPyInt(VoidPointer auto pointer) noexcept
{
    return reinterpret_cast<uintptr_t>(pointer);
}

//! The ints for the addresses \p pointers hold, in order: the list a Python enqueue() receives.
template <VoidPointer T>
[[nodiscard]] std::vector<uintptr_t> addressesToPyInts(std::span<T const> pointers)
{
    std::vector<uintptr_t> addresses;
    addresses.reserve(pointers.size());
    for (T const pointer : pointers)
    {
        addresses.push_back(addressToPyInt(pointer));
    }
    return addresses;
}

} // namespace tensorrt::utils

#endif // TRT_PYTHON_ADDRESS_H
