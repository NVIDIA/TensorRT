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

#include "getOptions.h"
#include "ArgVec.test.h"

#include <gtest/gtest.h>

#include <string_view>

using nvinfer1::utility::getOptions;
using nvinfer1::utility::TRTOption;
using namespace std::string_view_literals;

using TestArgVec = ArgVec<char const*>;

// The options used by several tests below, matching the header's worked example.
static std::vector<TRTOption> const kEXAMPLE_OPTIONS{
    {'a', "", false},
    {'b', "", false},
    {'\0', "cee", false},
    {'d', "", true},
    {'e', "", true},
    {'f', "foo", true},
};

TEST(GetOptions, PositionalArgs)
{
    TestArgVec av{"hello", "world"};
    auto const result = getOptions(av.argc(), av.argv(), kEXAMPLE_OPTIONS);
    EXPECT_TRUE(result.errMsg.empty());
    ASSERT_EQ(result.positionalArgs.size(), 2U);
    EXPECT_EQ(result.positionalArgs[0], "hello"sv);
    EXPECT_EQ(result.positionalArgs[1], "world"sv);
}

TEST(GetOptions, ShortFlag)
{
    TestArgVec av{"-a"};
    auto const result = getOptions(av.argc(), av.argv(), kEXAMPLE_OPTIONS);
    EXPECT_TRUE(result.errMsg.empty());
    EXPECT_EQ(result.values[0].occurrences, 1); // 'a' is index 0
}

TEST(GetOptions, ShortFlagRepeated)
{
    TestArgVec av{"-a", "-a"};
    auto const result = getOptions(av.argc(), av.argv(), kEXAMPLE_OPTIONS);
    EXPECT_TRUE(result.errMsg.empty());
    EXPECT_EQ(result.values[0].occurrences, 2);
}

TEST(GetOptions, LongFlag)
{
    TestArgVec av{"--cee"};
    auto const result = getOptions(av.argc(), av.argv(), kEXAMPLE_OPTIONS);
    EXPECT_TRUE(result.errMsg.empty());
    EXPECT_EQ(result.values[2].occurrences, 1); // "cee" is index 2
}

TEST(GetOptions, ShortValueSpaceSeparated)
{
    TestArgVec av{"-d", "12"};
    auto const result = getOptions(av.argc(), av.argv(), kEXAMPLE_OPTIONS);
    EXPECT_TRUE(result.errMsg.empty());
    ASSERT_EQ(result.values[3].occurrences, 1); // 'd' is index 3
    EXPECT_EQ(result.values[3].values[0], "12"sv);
}

TEST(GetOptions, LongValueEqualsSign)
{
    TestArgVec av{"--foo=34"};
    auto const result = getOptions(av.argc(), av.argv(), kEXAMPLE_OPTIONS);
    EXPECT_TRUE(result.errMsg.empty());
    ASSERT_EQ(result.values[5].occurrences, 1); // "foo" is index 5
    EXPECT_EQ(result.values[5].values[0], "34"sv);
}

TEST(GetOptions, ExactExampleFromHeader)
{
    // ./main hello world -a -a --cee -d 12 -f 34
    TestArgVec av{"hello", "world", "-a", "-a", "--cee", "-d", "12", "-f", "34"};
    auto const result = getOptions(av.argc(), av.argv(), kEXAMPLE_OPTIONS);
    EXPECT_TRUE(result.errMsg.empty());
    EXPECT_EQ(result.values[0].occurrences, 2); // 'a'
    EXPECT_EQ(result.values[1].occurrences, 0); // 'b'
    EXPECT_EQ(result.values[2].occurrences, 1); // "cee"
    ASSERT_EQ(result.values[3].occurrences, 1); // 'd'
    EXPECT_EQ(result.values[3].values[0], "12"sv);
    EXPECT_EQ(result.values[4].occurrences, 0); // 'e'
    ASSERT_EQ(result.values[5].occurrences, 1); // "foo"/"f"
    EXPECT_EQ(result.values[5].values[0], "34"sv);
    ASSERT_EQ(result.positionalArgs.size(), 2U);
    EXPECT_EQ(result.positionalArgs[0], "hello"sv);
    EXPECT_EQ(result.positionalArgs[1], "world"sv);
}

TEST(GetOptions, UnknownOptionsIgnored)
{
    TestArgVec av{"--unknown-flag"};
    auto const result = getOptions(av.argc(), av.argv(), kEXAMPLE_OPTIONS);
    EXPECT_TRUE(result.errMsg.empty());
    for (auto const& v : result.values)
    {
        EXPECT_EQ(v.occurrences, 0);
    }
}

TEST(GetOptions, MissingRequiredValue)
{
    TestArgVec av{"-d"}; // 'd' requires a value but none is given
    auto const result = getOptions(av.argc(), av.argv(), kEXAMPLE_OPTIONS);
    EXPECT_FALSE(result.errMsg.empty());
}

TEST(GetOptions, DuplicateShortName)
{
    std::vector<TRTOption> const opts{{'a', "", false}, {'a', "other", false}};
    TestArgVec av{};
    auto const result = getOptions(av.argc(), av.argv(), opts);
    EXPECT_FALSE(result.errMsg.empty());
}

TEST(GetOptions, EmptyOptions)
{
    TestArgVec av{"hello"};
    auto const result = getOptions(av.argc(), av.argv(), {});
    EXPECT_TRUE(result.errMsg.empty());
    ASSERT_EQ(result.positionalArgs.size(), 1U);
    EXPECT_EQ(result.positionalArgs[0], "hello"sv);
}
