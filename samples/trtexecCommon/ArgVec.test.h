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

#ifndef TRT_SAMPLES_COMMON_ARGVEC_TEST_H
#define TRT_SAMPLES_COMMON_ARGVEC_TEST_H

#include <cstdint>
#include <string>
#include <vector>

//! Wraps a list of argument strings as argc/argv for use with argument-parsing functions.
//!
//! CharPtrT controls the constness of the argv pointers:
//!   - ArgVec<char*>        for APIs that take char** (e.g. argsToArgumentsMap)
//!   - ArgVec<char const*>  for APIs that take char const* const* (e.g. getOptions)
template <typename CharPtrT>
struct ArgVec
{
    explicit ArgVec(std::initializer_list<char const*> strs)
    {
        mArgs.emplace_back("prog"); // argv[0] is the program name; argument parsers skip it.
        mArgs.insert(mArgs.end(), strs.begin(), strs.end());
        for (auto& s : mArgs)
        {
            mArgv.push_back(s.data());
        }
    }

    [[nodiscard]] int32_t argc() const
    {
        return static_cast<int32_t>(mArgv.size());
    }

    [[nodiscard]] CharPtrT* argv()
    {
        return mArgv.data();
    }

private:
    std::vector<std::string> mArgs;
    std::vector<CharPtrT> mArgv;
};

#endif // TRT_SAMPLES_COMMON_ARGVEC_TEST_H
