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
#include "ArgVec.test.h"

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

TEST(SanitizeRemoteConfig, Empty)
{
    EXPECT_EQ(sanitizeRemoteConfig(""), ""sv);
}

TEST(SanitizeRemoteConfig, NoCredentials)
{
    // No @ means no credentials section; returned as-is.
    EXPECT_EQ(sanitizeRemoteConfig("ssh://host:22"), "ssh://host:22"sv);
}

TEST(SanitizeRemoteConfig, UsernameOnly)
{
    EXPECT_EQ(sanitizeRemoteConfig("ssh://user@host:22"), "ssh://***@host:22"sv);
}

TEST(SanitizeRemoteConfig, UsernameAndPassword)
{
    EXPECT_EQ(sanitizeRemoteConfig("ssh://user:pass@host:22"), "ssh://***@host:22"sv);
}

TEST(SanitizeRemoteConfig, WithQueryParams)
{
    EXPECT_EQ(
        sanitizeRemoteConfig("ssh://admin:secret@server.com:22?timeout=30"), "ssh://***@server.com:22?timeout=30"sv);
}

TEST(SanitizeArgv, MasksRemoteConfigCredentials)
{
    ArgVec<char*> av{"--remoteConfig=ssh://user:pass@host:22", "--safe"};
    auto const sanitized = sanitizeArgv(av.argc(), av.argv());
    ASSERT_EQ(sanitized.size(), 3U);
    EXPECT_EQ(sanitized[1], "--remoteConfig=ssh://***@host:22"sv);
    EXPECT_EQ(sanitized[2], "--safe"sv);
}

TEST(SanitizeArgv, MasksAliasCredentials)
{
    ArgVec<char*> av{"--remoteAutoTuningConfig=ssh://user:pass@host:22"};
    auto const sanitized = sanitizeArgv(av.argc(), av.argv());
    ASSERT_EQ(sanitized.size(), 2U);
    EXPECT_EQ(sanitized[1], "--remoteAutoTuningConfig=ssh://***@host:22"sv);
}

TEST(SanitizeArgv, LeavesOtherArgumentsUntouched)
{
    ArgVec<char*> av{"--onnx=model.onnx", "--remoteConfig="};
    auto const sanitized = sanitizeArgv(av.argc(), av.argv());
    ASSERT_EQ(sanitized.size(), 3U);
    EXPECT_EQ(sanitized[1], "--onnx=model.onnx"sv);
    EXPECT_EQ(sanitized[2], "--remoteConfig="sv);
}
