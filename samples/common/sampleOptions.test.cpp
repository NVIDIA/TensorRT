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

#include "sampleOptions.h"
#include "ArgVec.test.h"

#include <gtest/gtest.h>

#include <string_view>

using namespace sample;
using namespace std::string_view_literals;

using TestArgVec = ArgVec<char*>;

TEST(ArgsToArgumentsMap, Empty)
{
    TestArgVec av{};
    auto const args = argsToArgumentsMap(av.argc(), av.argv());
    EXPECT_TRUE(args.empty());
}

TEST(ArgsToArgumentsMap, FlagArg)
{
    TestArgVec av{"--verbose"};
    auto const args = argsToArgumentsMap(av.argc(), av.argv());
    ASSERT_EQ(args.count("--verbose"), 1U);
    EXPECT_EQ(args.find("--verbose")->second.first, ""sv);
}

TEST(ArgsToArgumentsMap, KeyValueArg)
{
    TestArgVec av{"--onnx=model.onnx"};
    auto const args = argsToArgumentsMap(av.argc(), av.argv());
    ASSERT_EQ(args.count("--onnx"), 1U);
    EXPECT_EQ(args.find("--onnx")->second.first, "model.onnx"sv);
}

TEST(ArgsToArgumentsMap, MultipleArgs)
{
    TestArgVec av{"--onnx=model.onnx", "--fp16", "--batch=4"};
    auto const args = argsToArgumentsMap(av.argc(), av.argv());
    ASSERT_EQ(args.count("--onnx"), 1U);
    ASSERT_EQ(args.count("--fp16"), 1U);
    ASSERT_EQ(args.count("--batch"), 1U);
    EXPECT_EQ(args.find("--onnx")->second.first, "model.onnx"sv);
    EXPECT_EQ(args.find("--fp16")->second.first, ""sv);
    EXPECT_EQ(args.find("--batch")->second.first, "4"sv);
}

TEST(ArgsToArgumentsMap, ValueWithEquals)
{
    // Values can themselves contain '='; only the first '=' is the key/value separator.
    TestArgVec av{"--key=a=b"};
    auto const args = argsToArgumentsMap(av.argc(), av.argv());
    ASSERT_EQ(args.count("--key"), 1U);
    EXPECT_EQ(args.find("--key")->second.first, "a=b"sv);
}

TEST(ArgsToArgumentsMap, ArgPositionRecorded)
{
    // argsToArgumentsMap records the original argv index (1-based, skipping argv[0]).
    TestArgVec av{"--onnx=model.onnx", "--fp16"};
    auto const args = argsToArgumentsMap(av.argc(), av.argv());
    EXPECT_EQ(args.find("--onnx")->second.second, 1);
    EXPECT_EQ(args.find("--fp16")->second.second, 2);
}
