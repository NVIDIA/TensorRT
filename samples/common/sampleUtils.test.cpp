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
