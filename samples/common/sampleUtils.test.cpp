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

#include "sampleUtils.h"

#include <gtest/gtest.h>

#include <string_view>

using namespace sample;
using namespace std::string_view_literals;

TEST(RoundUp, ExactMultiple)
{
    EXPECT_EQ(roundUp(4, 4), 4);
    EXPECT_EQ(roundUp(8, 4), 8);
    EXPECT_EQ(roundUp(0, 4), 0);
}

TEST(RoundUp, NeedsRounding)
{
    EXPECT_EQ(roundUp(1, 4), 4);
    EXPECT_EQ(roundUp(5, 4), 8);
    EXPECT_EQ(roundUp(7, 4), 8);
}

TEST(SplitToStringVec, SingleToken)
{
    auto const v = splitToStringVec("hello", ',');
    ASSERT_EQ(v.size(), 1U);
    EXPECT_EQ(v[0], "hello"sv);
}

TEST(SplitToStringVec, MultipleTokens)
{
    auto const v = splitToStringVec("a,b,c", ',');
    ASSERT_EQ(v.size(), 3U);
    EXPECT_EQ(v[0], "a"sv);
    EXPECT_EQ(v[1], "b"sv);
    EXPECT_EQ(v[2], "c"sv);
}

TEST(SplitToStringVec, EmptyString)
{
    auto const v = splitToStringVec("", ',');
    EXPECT_TRUE(v.empty());
}

TEST(SplitToStringVec, MaxSplit)
{
    // maxSplit=1 means at most one split; the rest of the string is the second element.
    auto const v = splitToStringVec("a:b:c", ':', 1);
    ASSERT_EQ(v.size(), 2U);
    EXPECT_EQ(v[0], "a"sv);
    EXPECT_EQ(v[1], "b:c"sv);
}

TEST(SplitToStringVec, TrailingSeparator)
{
    auto const v = splitToStringVec("a,b,", ',');
    ASSERT_EQ(v.size(), 3U);
    EXPECT_EQ(v[0], "a"sv);
    EXPECT_EQ(v[1], "b"sv);
    EXPECT_EQ(v[2], ""sv);
}

TEST(MatchStringWithOneWildcard, ExactMatch)
{
    EXPECT_TRUE(matchStringWithOneWildcard("hello", "hello"));
    EXPECT_FALSE(matchStringWithOneWildcard("hello", "world"));
    EXPECT_FALSE(matchStringWithOneWildcard("hello", "hello2"));
}

TEST(MatchStringWithOneWildcard, WildcardMatchesAnything)
{
    EXPECT_TRUE(matchStringWithOneWildcard("*", "anything"));
    EXPECT_TRUE(matchStringWithOneWildcard("*", ""));
}

TEST(MatchStringWithOneWildcard, PrefixWildcard)
{
    EXPECT_TRUE(matchStringWithOneWildcard("hello*", "hello"));
    EXPECT_TRUE(matchStringWithOneWildcard("hello*", "hello world"));
    EXPECT_FALSE(matchStringWithOneWildcard("hello*", "world"));
}

TEST(MatchStringWithOneWildcard, SuffixWildcard)
{
    EXPECT_TRUE(matchStringWithOneWildcard("*world", "world"));
    EXPECT_TRUE(matchStringWithOneWildcard("*world", "hello world"));
    EXPECT_FALSE(matchStringWithOneWildcard("*world", "hello"));
}

TEST(MatchStringWithOneWildcard, MiddleWildcard)
{
    EXPECT_TRUE(matchStringWithOneWildcard("he*ld", "held"));
    EXPECT_TRUE(matchStringWithOneWildcard("he*ld", "hello world"));
    EXPECT_FALSE(matchStringWithOneWildcard("he*ld", "hello"));
}

TEST(NormalizeDirectoryPath, AlreadyNormalized)
{
    EXPECT_EQ(normalizeDirectoryPath("/some/path/"), "/some/path/"sv);
}

TEST(NormalizeDirectoryPath, MissingTrailingSlash)
{
    EXPECT_EQ(normalizeDirectoryPath("/some/path"), "/some/path/"sv);
}

TEST(NormalizeDirectoryPath, EmptyString)
{
    EXPECT_EQ(normalizeDirectoryPath(""), ""sv);
}

TEST(SanitizeRemoteAutoTuningConfig, Empty)
{
    EXPECT_EQ(sanitizeRemoteAutoTuningConfig(""), ""sv);
}

TEST(SanitizeRemoteAutoTuningConfig, NoCredentials)
{
    // No @ means no credentials section; returned as-is.
    EXPECT_EQ(sanitizeRemoteAutoTuningConfig("ssh://host:22"), "ssh://host:22"sv);
}

TEST(SanitizeRemoteAutoTuningConfig, UsernameOnly)
{
    EXPECT_EQ(sanitizeRemoteAutoTuningConfig("ssh://user@host:22"), "ssh://***@host:22"sv);
}

TEST(SanitizeRemoteAutoTuningConfig, UsernameAndPassword)
{
    EXPECT_EQ(sanitizeRemoteAutoTuningConfig("ssh://user:pass@host:22"), "ssh://***@host:22"sv);
}

TEST(SanitizeRemoteAutoTuningConfig, WithQueryParams)
{
    EXPECT_EQ(sanitizeRemoteAutoTuningConfig("ssh://admin:secret@server.com:22?timeout=30"),
        "ssh://***@server.com:22?timeout=30"sv);
}
