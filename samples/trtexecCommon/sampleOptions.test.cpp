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

#include <stdexcept>
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

#if !TRT_WINML
TEST(SystemOptions, DLAWorkspaceAllocationStrategyDefaultsToDefault)
{
    SystemOptions const options;
    EXPECT_EQ(options.dlaWorkspaceAllocationStrategy, nvinfer1::DLAWorkspaceAllocationStrategy::kDEFAULT);
}

TEST(SystemOptions, ParsesDLAWorkspaceAllocationStrategy)
{
    auto const parseStrategy = [](char const* argument) {
        TestArgVec av{argument};
        auto args = argsToArgumentsMap(av.argc(), av.argv());
        SystemOptions options;

        options.parse(args);

        EXPECT_TRUE(args.empty());
        return options.dlaWorkspaceAllocationStrategy;
    };

    EXPECT_EQ(
        parseStrategy("--dlaWorkspaceAllocationStrategy=default"), nvinfer1::DLAWorkspaceAllocationStrategy::kDEFAULT);
    EXPECT_EQ(parseStrategy("--dlaWorkspaceAllocationStrategy=sharedStatic"),
        nvinfer1::DLAWorkspaceAllocationStrategy::kSHARED_STATIC);
}

TEST(SystemOptions, RejectsUnknownDLAWorkspaceAllocationStrategy)
{
    TestArgVec av{"--dlaWorkspaceAllocationStrategy=invalid"};
    auto args = argsToArgumentsMap(av.argc(), av.argv());
    SystemOptions options;

    EXPECT_THROW(options.parse(args), std::invalid_argument);
}

namespace
{

//! \return the BuildOptions produced by parsing \p av.
[[nodiscard]] BuildOptions parseBuildOptions(TestArgVec& av)
{
    auto args = argsToArgumentsMap(av.argc(), av.argv());
    BuildOptions options{};
    options.parse(args);
    return options;
}

} // namespace

TEST(BuildOptionsParse, RemoteConfig)
{
    TestArgVec av{"--safe", "--remoteConfig=ssh://host:22"};
    EXPECT_EQ(parseBuildOptions(av).remoteConfig, "ssh://host:22"sv);
}

TEST(BuildOptionsParse, RemoteConfigAlias)
{
    TestArgVec av{"--safe", "--remoteAutoTuningConfig=ssh://host:22"};
    EXPECT_EQ(parseBuildOptions(av).remoteConfig, "ssh://host:22"sv);
}

TEST(BuildOptionsParse, RemoteConfigAliasAgreeingValues)
{
    TestArgVec av{"--safe", "--remoteConfig=ssh://host:22", "--remoteAutoTuningConfig=ssh://host:22"};
    EXPECT_EQ(parseBuildOptions(av).remoteConfig, "ssh://host:22"sv);
}

TEST(BuildOptionsParse, RemoteConfigAliasConflict)
{
    TestArgVec av{"--safe", "--remoteConfig=ssh://host:22", "--remoteAutoTuningConfig=ssh://other:22"};
    EXPECT_THROW(static_cast<void>(parseBuildOptions(av)), std::invalid_argument);
}

TEST(BuildOptionsParse, DumpCheckerBlob)
{
    TestArgVec av{"--safe", "--dumpCheckerBlob"};
    EXPECT_TRUE(parseBuildOptions(av).dumpCheckerBlob);
}

//! --dumpKernelText is the name the flag shipped under, so scripts still spell it that way.
TEST(BuildOptionsParse, DumpCheckerBlobAlias)
{
    TestArgVec av{"--safe", "--dumpKernelText"};
    EXPECT_TRUE(parseBuildOptions(av).dumpCheckerBlob);
}

//! Both spellings must be consumed even when only one of them sets the flag, or trtexec rejects the
//! survivor as an unknown option.
TEST(BuildOptionsParse, DumpCheckerBlobAliasConsumesBothSpellings)
{
    TestArgVec av{"--safe", "--dumpCheckerBlob", "--dumpKernelText"};
    auto args = argsToArgumentsMap(av.argc(), av.argv());
    BuildOptions options{};
    options.parse(args);

    EXPECT_TRUE(options.dumpCheckerBlob);
    EXPECT_TRUE(args.empty());
}

TEST(BuildOptionsParse, DumpCheckerBlobAliasRequiresSafe)
{
    TestArgVec av{"--dumpKernelText"};
    EXPECT_THROW(static_cast<void>(parseBuildOptions(av)), std::invalid_argument);
}

TEST(BuildOptionsParse, Reference)
{
    TestArgVec av{"--safe", "--reference", "--loadEngine=e.mvmf", "--remoteConfig=ssh://host:22"};
    EXPECT_TRUE(parseBuildOptions(av).reference);
}

//! The metadata a check reads is written beside the engine at build time, so checking is a separate run
//! over a saved engine. Asking to build and check at once fails rather than checking nothing.
TEST(BuildOptionsParse, ReferenceRequiresLoadEngine)
{
    TestArgVec av{"--safe", "--reference", "--remoteConfig=ssh://host:22"};
    EXPECT_THROW(static_cast<void>(parseBuildOptions(av)), std::invalid_argument);
}

//! Falls out of --reference requiring --loadEngine, since load and save are already exclusive.
TEST(BuildOptionsParse, ReferenceCannotSaveAnEngine)
{
    TestArgVec av{"--safe", "--reference", "--saveEngine=e.mvmf", "--remoteConfig=ssh://host:22"};
    EXPECT_THROW(static_cast<void>(parseBuildOptions(av)), std::invalid_argument);
}

//! The blob path is never derived from the engine path, so a check reads only the blob it was given.
TEST(BuildOptionsParse, LoadCheckerBlob)
{
    TestArgVec av{"--safe", "--reference", "--loadEngine=e.mvmf", "--loadCheckerBlob=/tmp/kt.txt",
        "--remoteConfig=ssh://host:22"};
    EXPECT_EQ(parseBuildOptions(av).checkerBlob, "/tmp/kt.txt");
}

//! Omitting the flag leaves the check to report that the engine it read needed a blob it did not get.
TEST(BuildOptionsParse, LoadCheckerBlobDefaultsEmpty)
{
    TestArgVec av{"--safe", "--reference", "--loadEngine=e.mvmf", "--remoteConfig=ssh://host:22"};
    EXPECT_TRUE(parseBuildOptions(av).checkerBlob.empty());
}

TEST(BuildOptionsParse, LoadCheckerBlobRequiresReference)
{
    TestArgVec av{"--safe", "--loadEngine=e.mvmf", "--loadCheckerBlob=/tmp/kt.txt"};
    EXPECT_THROW(static_cast<void>(parseBuildOptions(av)), std::invalid_argument);
}

#if !HOS_WIN_BUILDER
//! A remote target is only meaningful for a safe build, which is the same build the reference checker
//! needs, so the two flags cannot be separated.
TEST(BuildOptionsParse, RemoteConfigRequiresSafe)
{
    TestArgVec av{"--reference", "--remoteConfig=ssh://host:22"};
    EXPECT_THROW(static_cast<void>(parseBuildOptions(av)), std::invalid_argument);
}
#endif // !HOS_WIN_BUILDER

//! --reference is off unless asked for, so a safe build that does not want it pays nothing.
TEST(BuildOptionsParse, ReferenceDefaultsOff)
{
    TestArgVec av{"--safe"};
    EXPECT_FALSE(parseBuildOptions(av).reference);
}
#endif // !TRT_WINML
