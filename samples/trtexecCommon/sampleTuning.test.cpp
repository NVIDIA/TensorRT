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

#include "sampleTuning.h"
#include "ArgVec.test.h"
#include "sampleUtils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

using namespace sample;

namespace
{
//! \brief Construct synthetic multi-valued knobs with an explicit fixed override.
[[nodiscard]] TuningContext makeContext(
    int32_t count, TuningSearchAlgorithm algorithm = TuningSearchAlgorithm::kFAST, std::string const& cacheFile = {})
{
    TuningContext ctx;
    ctx.searchStrategy = createSearchStrategy(algorithm, cacheFile);
    for (int32_t i = 0; i < count; ++i)
    {
        ctx.parsedExprs.push_back({"-knob" + std::to_string(i), {"off", "on", "other"}, false});
        ctx.defaultValues.push_back("off");
    }
    ctx.parsedExprs.push_back({"-fixed", {"override"}, true});
    ctx.defaultValues.push_back("default");
    return ctx;
}

//! \brief Enumerate a bounded search starting directly at the requested index.
//! This test helper collects small expected sequences; production code consumes one route at a time.
[[nodiscard]] std::vector<std::string> routes(TuningContext& ctx, BigInt index = BigInt{0})
{
    std::vector<std::string> result;
    for (auto route = ctx.getBuildRouteAtIndex(index); route; route = ctx.getBuildRouteAtIndex(++index))
    {
        result.push_back(*route);
    }
    return result;
}

//! \brief Own a temporary JSON-lines tuning cache.
class TuningCacheTest : public ::testing::Test
{
protected:
    //! \brief Create a separate cache with a valid header for each test.
    void SetUp() override
    {
        mPath = (std::filesystem::temp_directory_path()
            / ("trt-tuning-cache-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
                    .string();
        std::ofstream(mPath) << R"({"argv":["trtexec"],"tuning_expr":"-a=[on|off]"})" << '\n';
    }

    //! \brief Remove this test's cache, including after an assertion failure.
    void TearDown() override
    {
        std::filesystem::remove(mPath);
    }

    //! \brief Persist a fast scan with specified improvements and a 1000 ms baseline.
    //! Unlisted variations receive 2000 ms so only the supplied candidates can improve on baseline.
    void writeMeasurements(TuningContext const& ctx, std::vector<MixedSearchKnobResult> const& improvements)
    {
        FastSearch fast;
        BigInt const count = *fast.count(ctx);
        for (BigInt i{0}; i < count; ++i)
        {
            double gpuTimeMs = i.isZero() ? 1000.0 : 2000.0;
            if (auto const knob = identifyVariedKnob(ctx, i))
            {
                for (auto const& candidate : improvements)
                {
                    if (candidate.knobIndex == knob->first && candidate.value == knob->second)
                    {
                        gpuTimeMs = candidate.gpuTimeMs;
                    }
                }
            }
            ASSERT_TRUE(writeTuningCacheIteration(
                mPath, i.toUint64(), *fast.getBuildRouteAtIndex(ctx, i), false, "", {}, gpuTimeMs));
        }
    }

    std::string mPath;
};
} // namespace

TEST(TuningSearch, FullAndFastKeepExistingOrder)
{
    auto ctx = makeContext(2, TuningSearchAlgorithm::kEXHAUSTIVE);
    EXPECT_EQ(ctx.count(), BigInt{9});
    EXPECT_EQ(ctx.getBuildRouteAtIndex(BigInt{0}), "-knob0=off -knob1=off -fixed=override");
    EXPECT_EQ(ctx.getBuildRouteAtIndex(BigInt{1}), "-knob0=off -knob1=on -fixed=override");
    EXPECT_EQ(ctx.getBuildRouteAtIndex(BigInt{3}), "-knob0=on -knob1=off -fixed=override");
    EXPECT_EQ(ctx.getBuildRouteAtIndex(BigInt{8}), "-knob0=other -knob1=other -fixed=override");
    EXPECT_FALSE(ctx.getBuildRouteAtIndex(BigInt{9}));
    ctx.searchStrategy = std::make_unique<FastSearch>();
    EXPECT_EQ(routes(ctx),
        (std::vector<std::string>{"-knob0=off -knob1=off -fixed=override", "-knob0=off -knob1=on -fixed=override",
            "-knob0=off -knob1=other -fixed=override", "-knob0=on -knob1=off -fixed=override",
            "-knob0=other -knob1=off -fixed=override"}));
}

TEST(TuningSearch, ComputesLargeIndicesWithoutExpandingRoutes)
{
    auto ctx = makeContext(70, TuningSearchAlgorithm::kEXHAUSTIVE);
    for (auto& expr : ctx.parsedExprs)
    {
        if (!expr.mIsFixed)
        {
            expr.mValues = {"off", "on"};
        }
    }
    BigInt const count{"1180591620717411303424"}; // 2^70.
    EXPECT_EQ(ctx.count(), count);
    auto const route = ctx.getBuildRouteAtIndex(count - BigInt{1});
    ASSERT_TRUE(route);
    EXPECT_EQ(route->find("=off"), std::string::npos);
    EXPECT_NE(route->find("-knob69=on -fixed=override"), std::string::npos);
    EXPECT_FALSE(ctx.getBuildRouteAtIndex(count));
    EXPECT_EQ(ctx.globalIndex, BigInt{0});
}

TEST(TuningSearch, CustomStrategyUsesTheSameContextInterface)
{
    class CustomSearch : public ISearchStrategy
    {
    public:
        //! \brief Supply one route without consulting or advancing the context's execution index.
        std::optional<std::string> getBuildRouteAtIndex(TuningContext const&, BigInt const& index) override
        {
            return index.isZero() ? std::make_optional<std::string>("-custom=on") : std::nullopt;
        }
        //! \brief Report the custom search's single route through the common interface.
        std::optional<BigInt> count(TuningContext const&) const override
        {
            return BigInt{1};
        }
    };
    TuningContext ctx;
    ctx.searchStrategy = std::make_unique<CustomSearch>();
    ctx.globalIndex = BigInt{42};
    EXPECT_EQ(ctx.count(), BigInt{1});
    EXPECT_EQ(ctx.getBuildRouteAtIndex(BigInt{0}), "-custom=on");
    EXPECT_FALSE(ctx.getBuildRouteAtIndex(BigInt{1}));
    EXPECT_EQ(ctx.globalIndex, BigInt{42});
}

TEST(TuningSearch, MixedInitialRoutesMatchFastWithoutReadingCache)
{
    for (auto const algorithm : {TuningSearchAlgorithm::kMIXED, TuningSearchAlgorithm::kMIXED_FAST})
    {
        auto ctx = makeContext(3, algorithm, "/missing/cache");
        FastSearch fast;
        EXPECT_FALSE(ctx.count());
        for (BigInt i{0}; i < *fast.count(ctx); ++i)
        {
            EXPECT_EQ(ctx.getBuildRouteAtIndex(i), fast.getBuildRouteAtIndex(ctx, i));
        }
        EXPECT_THROW(ctx.getBuildRouteAtIndex(*fast.count(ctx)), std::runtime_error);
        EXPECT_FALSE(ctx.count());
    }
}

TEST_F(TuningCacheTest, ThreeKnobsRunInDescendingRankedSubsetOrder)
{
    auto ctx = makeContext(4, TuningSearchAlgorithm::kMIXED_FAST, mPath);
    writeMeasurements(ctx, {{0, "on", 99.0}, {1, "other", 98.0}, {2, "on", 97.0}});
    BigInt const scanCount = *FastSearch{}.count(ctx);
    EXPECT_EQ(routes(ctx, scanCount),
        (std::vector<std::string>{"-knob0=on -knob1=other -knob2=on -knob3=off -fixed=override",
            "-knob0=off -knob1=other -knob2=on -knob3=off -fixed=override",
            "-knob0=on -knob1=off -knob2=on -knob3=off -fixed=override",
            "-knob0=on -knob1=other -knob2=off -knob3=off -fixed=override"}));
    EXPECT_EQ(ctx.count(), scanCount + BigInt{4});
    EXPECT_EQ(ctx.globalIndex, BigInt{0});
}

TEST_F(TuningCacheTest, SelectsBestValueBeforeApplyingDistinctKnobLimit)
{
    auto ctx = makeContext(12, TuningSearchAlgorithm::kMIXED_FAST, mPath);
    std::vector<MixedSearchKnobResult> positive{{0, "other", 0.5}, {0, "on", 0.7}};
    for (int32_t i = 1; i < 12; ++i)
    {
        positive.push_back({i, "on", static_cast<double>(i)});
    }
    writeMeasurements(ctx, positive);
    BigInt const scanCount = *FastSearch{}.count(ctx);
    auto const combinations = routes(ctx, scanCount);
    EXPECT_EQ(combinations.size(), 120U);
    EXPECT_EQ(std::set<std::string>(combinations.begin(), combinations.end()).size(), 120U);
    EXPECT_NE(combinations.front().find("-knob0=other"), std::string::npos);
    for (auto const& route : combinations)
    {
        EXPECT_EQ(route.find("-knob0=on"), std::string::npos);
        EXPECT_NE(route.find("-knob7=off -knob8=off"), std::string::npos);
    }
    ctx.searchStrategy = std::make_unique<MixedSearch>(mPath);
    EXPECT_EQ(ctx.getBuildRouteAtIndex(scanCount), FastSearch{}.getBuildRouteAtIndex(ctx, BigInt{0}));
    EXPECT_EQ(ctx.count(), scanCount + BigInt{59049}); // All three values of ten selected knobs.
    auto const last = ctx.getBuildRouteAtIndex(*ctx.count() - BigInt{1});
    ASSERT_TRUE(last);
    EXPECT_NE(last->find("-knob9=other -knob10=off -knob11=off"), std::string::npos);
}

TEST_F(TuningCacheTest, CountsZeroThroughTenImprovingKnobs)
{
    for (int32_t n = 0; n <= 10; ++n)
    {
        SCOPED_TRACE(n);
        std::ofstream(mPath) << "{}\n";
        auto ctx = makeContext(n, TuningSearchAlgorithm::kMIXED_FAST, mPath);
        std::vector<MixedSearchKnobResult> positive;
        for (int32_t i = 0; i < n; ++i)
        {
            positive.push_back({i, "on", static_cast<double>(i + 1)});
        }
        writeMeasurements(ctx, positive);
        uint32_t const selected = std::min(static_cast<uint32_t>(n), MixedFastSearch::kMAX_POSITIVE_KNOBS);
        uint64_t const expected = (uint64_t{1} << selected) - 1 - selected;
        BigInt const scanCount = *FastSearch{}.count(ctx);
        EXPECT_EQ(routes(ctx, scanCount).size(), expected);
        EXPECT_EQ(ctx.count(), scanCount + BigInt{expected});
    }
}

TEST_F(TuningCacheTest, DirectResumeAndEqualTimeTiesAreDeterministic)
{
    auto ctx = makeContext(8, TuningSearchAlgorithm::kMIXED_FAST, mPath);
    std::vector<MixedSearchKnobResult> positive;
    for (int32_t i = 0; i < 8; ++i)
    {
        positive.push_back({i, "other", 1.0});
        positive.push_back({i, "on", 1.0});
    }
    writeMeasurements(ctx, positive);
    BigInt const scanCount = *FastSearch{}.count(ctx);
    auto const expected = routes(ctx, scanCount);
    EXPECT_NE(expected.front().find("-knob0=on"), std::string::npos);
    EXPECT_NE(expected.front().find("-knob7=off"), std::string::npos);
    for (size_t completed = 0; completed <= expected.size(); ++completed)
    {
        auto resumed = makeContext(8, TuningSearchAlgorithm::kMIXED_FAST, mPath);
        resumed.globalIndex = scanCount + BigInt{completed};
        EXPECT_EQ(routes(resumed, resumed.globalIndex),
            (std::vector<std::string>{expected.begin() + completed, expected.end()}));
    }
    // Loading the bounded selection once is sufficient for subsequent route requests.
    std::filesystem::remove(mPath);
    EXPECT_EQ(routes(ctx, scanCount), expected);
}

TEST_F(TuningCacheTest, SelectionIgnoresLaterResultsAndDoesNotValidateSavedRoutes)
{
    auto ctx = makeContext(2, TuningSearchAlgorithm::kMIXED_FAST, mPath);
    std::vector<double> const times{100.0, 97.0, 96.0, 99.0, 98.0};
    for (uint64_t i = 0; i < times.size(); ++i)
    {
        ASSERT_TRUE(writeTuningCacheIteration(
            mPath, 42, "stored route", false, i == 2 ? "Failed accuracy validation" : "", {}, times[i]));
    }
    std::ofstream(mPath, std::ios::app) << "invalid later row\n";
    EXPECT_EQ(ctx.getBuildRouteAtIndex(BigInt{5}), "-knob0=other -knob1=on -fixed=override");
    EXPECT_EQ(ctx.count(), BigInt{6});
}

TEST_F(TuningCacheTest, IncompleteMeasurementsAreAnErrorRatherThanEndOfSearch)
{
    auto ctx = makeContext(2, TuningSearchAlgorithm::kMIXED_FAST, mPath);
    ASSERT_TRUE(writeTuningCacheIteration(mPath, 0, "baseline", false, "", {}, 100));
    EXPECT_THROW(ctx.getBuildRouteAtIndex(BigInt{5}), std::runtime_error);
    EXPECT_FALSE(ctx.count());
}

TEST_F(TuningCacheTest, MixedRetainsOriginalDomainsIncludingMissingDefaults)
{
    auto ctx = makeContext(2, TuningSearchAlgorithm::kMIXED, mPath);
    ctx.parsedExprs[0].mValues = {"on", "other"};
    ctx.parsedExprs[1].mValues = {"other"};
    writeMeasurements(ctx, {{0, "on", 2.0}, {0, "other", 1.0}, {1, "other", 3.0}});
    EXPECT_EQ(routes(ctx, BigInt{4}),
        (std::vector<std::string>{
            "-knob0=on -knob1=other -fixed=override", "-knob0=other -knob1=other -fixed=override"}));
    ctx.searchStrategy = std::make_unique<MixedFastSearch>(mPath);
    EXPECT_EQ(routes(ctx, BigInt{4}), (std::vector<std::string>{"-knob0=other -knob1=other -fixed=override"}));
}

TEST_F(TuningCacheTest, FixedAndDefaultOnlySearchesFinishAfterBaseline)
{
    for (auto const algorithm : {TuningSearchAlgorithm::kMIXED, TuningSearchAlgorithm::kMIXED_FAST})
    {
        std::ofstream(mPath) << "{}\n";
        auto ctx = makeContext(1, algorithm, mPath);
        ctx.parsedExprs[0].mValues = {"off"};
        writeMeasurements(ctx, {});
        EXPECT_EQ(routes(ctx), (std::vector<std::string>{"-knob0=off -fixed=override"}));
        EXPECT_EQ(ctx.count(), BigInt{1});
    }
    auto fixed = makeContext(0);
    EXPECT_EQ(routes(fixed), (std::vector<std::string>{"-fixed=override"}));
}

TEST(TuningSearch, InvalidOrFailedMeasurementsAreNotPositive)
{
    auto const ctx = makeContext(2);
    std::vector<MixedSearchKnobResult> positive;
    collectPositiveKnobFromResult(true, 1.0, 5.0, BigInt{1}, ctx, positive);
    for (double const time :
        {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), 5.0, 6.0})
    {
        collectPositiveKnobFromResult(false, time, 5.0, BigInt{1}, ctx, positive);
    }
    collectPositiveKnobFromResult(false, 1.0, std::numeric_limits<double>::infinity(), BigInt{1}, ctx, positive);
    EXPECT_TRUE(positive.empty());
    collectPositiveKnobFromResult(false, 1.0, 5.0, BigInt{1}, ctx, positive);
    ASSERT_EQ(positive.size(), 1U);
    EXPECT_EQ(positive.front().knobIndex, 1);
    EXPECT_EQ(positive.front().value, "on");
}

TEST(MixedFast, OptionsRoundTripAndRejectDryRun)
{
    ArgVec<char*> args{"--tuningSearch=mixed-fast", "--tuneBuildRoutes=-a=[on|off]"};
    auto arguments = argsToArgumentsMap(args.argc(), args.argv());
    TuningOptions options;
    options.parse(arguments);
    EXPECT_EQ(options.tuningSearchAlgorithm, TuningSearchAlgorithm::kMIXED_FAST);
    EXPECT_EQ(toString(options.tuningSearchAlgorithm), "mixed-fast");
    ArgVec<char*> dryArgs{"--tuningSearch=mixed-fast", "--dryRun"};
    auto dryArguments = argsToArgumentsMap(dryArgs.argc(), dryArgs.argv());
    EXPECT_THROW(options.parse(dryArguments), std::invalid_argument);
}

TEST(MixedSearch, RequiresCacheForCombinations)
{
    for (auto const* algorithm : {"--tuningSearch=mixed", "--tuningSearch=mixed-fast"})
    {
        ArgVec<char*> args{algorithm, "--tuningCacheFile="};
        auto arguments = argsToArgumentsMap(args.argc(), args.argv());
        TuningOptions options;
        EXPECT_THROW(options.parse(arguments), std::invalid_argument);
    }
}

TEST_F(TuningCacheTest, HeaderOnlyAndPartialCacheRoundTrip)
{
    auto header = readTuningCacheHeader(mPath);
    ASSERT_TRUE(header);
    EXPECT_EQ(header->completedIterations, 0);
    std::ifstream empty(mPath);
    std::string headerLine;
    ASSERT_TRUE(std::getline(empty, headerLine));
    EXPECT_FALSE(readCachedIterationResult(empty));
    writeTuningCacheIteration(mPath, 0, "-a=off", false, "", {}, 5.0, "/tmp/best.plan");
    writeTuningCacheIteration(mPath, 1, "-a=on", false, "child exited with status 1", {}, 1.0);
    std::ifstream file(mPath);
    ASSERT_TRUE(std::getline(file, headerLine));
    auto first = readCachedIterationResult(file);
    ASSERT_TRUE(first);
    EXPECT_TRUE(first->succeeded());
    EXPECT_EQ(first->enginePath, "/tmp/best.plan");
    auto second = readCachedIterationResult(file);
    ASSERT_TRUE(second);
    EXPECT_FALSE(second->succeeded());
    EXPECT_EQ(second->buildRoute, "-a=on");
    EXPECT_FALSE(readCachedIterationResult(file));
}

TEST_F(TuningCacheTest, ResumeSummaryUsesRecordCountAndBestSuccessfulMeasurement)
{
    ASSERT_TRUE(writeTuningCacheIteration(mPath, 42, "saved first route", false, "", {}, 5.0, "/tmp/first.plan"));
    ASSERT_TRUE(writeTuningCacheIteration(mPath, 0, "failed route", false, "Accuracy failed", {}, 1.0));
    ASSERT_TRUE(writeTuningCacheIteration(mPath, 0, "saved best route", false, "", {}, 4.0, "/tmp/best.plan"));
    ASSERT_TRUE(writeTuningCacheIteration(mPath, 0, "tied route", false, "", {}, 4.0, "/tmp/tied.plan"));
    ASSERT_TRUE(writeTuningCacheIteration(mPath, 99, "crashed route", true, "Child failed", {}, 0.0));
    auto const summary = readTuningCacheHeader(mPath);
    ASSERT_TRUE(summary);
    EXPECT_EQ(summary->completedIterations, BigInt{5});
    EXPECT_EQ(summary->successCount, BigInt{3});
    ASSERT_TRUE(summary->bestResult);
    EXPECT_DOUBLE_EQ(summary->bestResult->gpuTimeMs, 4.0);
    EXPECT_EQ(summary->bestResult->buildRoute, "saved best route");
    EXPECT_EQ(summary->bestResult->enginePath, "/tmp/best.plan");
}

TEST_F(TuningCacheTest, ResumeSummaryRejectsMalformedLateRecord)
{
    ASSERT_TRUE(writeTuningCacheIteration(mPath, 0, "saved route", false, "", {}, 5.0));
    std::ofstream(mPath, std::ios::app) << "{\"iter\":\n";
    EXPECT_FALSE(readTuningCacheHeader(mPath));
}

TEST_F(TuningCacheTest, CacheReadFailureIsNotEndOfSearch)
{
    std::istringstream file;
    file.setstate(std::ios::badbit);
    EXPECT_THROW((void) readCachedIterationResult(file), std::runtime_error);
}

TEST_F(TuningCacheTest, MalformedRowCannotSilentlySkipWork)
{
    std::istringstream file(R"({"iter":0)");
    EXPECT_THROW((void) readCachedIterationResult(file), std::runtime_error);
}

TEST_F(TuningCacheTest, ReaderConsumesOnlyOneRow)
{
    std::string const row = R"({"iter":0,"build_route":"-a=off","crash":false,"error_message":"","gpu_time":1})";
    std::istringstream file("\n" + row + "\ninvalid next row\n");
    auto const result = readCachedIterationResult(file);
    ASSERT_TRUE(result);
    EXPECT_TRUE(result->succeeded());
    EXPECT_EQ(file.tellg(), static_cast<std::streamoff>(row.size() + 2));
    EXPECT_THROW((void) readCachedIterationResult(file), std::runtime_error);
}

TEST_F(TuningCacheTest, StreamsLargeCache)
{
    int32_t constexpr kROW_COUNT = 10000;
    {
        std::ofstream file(mPath, std::ios::app);
        std::string const message(1024, 'x');
        for (int32_t i = 0; i < kROW_COUNT; ++i)
        {
            file << "{\"iter\":" << i << ",\"build_route\":\"-a=off\",\"crash\":true,\"error_message\":\"" << message
                 << "\",\"gpu_time\":null}\n";
        }
    }
    auto const header = readTuningCacheHeader(mPath);
    ASSERT_TRUE(header);
    EXPECT_EQ(header->completedIterations, kROW_COUNT);
    std::ifstream file(mPath);
    std::string headerLine;
    ASSERT_TRUE(std::getline(file, headerLine));
    for (int32_t i = 0; i < kROW_COUNT; ++i)
    {
        auto const result = readCachedIterationResult(file);
        ASSERT_TRUE(result);
        EXPECT_EQ(result->iteration, static_cast<uint64_t>(i));
        EXPECT_FALSE(result->succeeded());
        EXPECT_EQ(result->errorMessage.size(), 1024U);
    }
    EXPECT_FALSE(readCachedIterationResult(file));
}

TEST_F(TuningCacheTest, WriteFailuresAreReported)
{
    AllOptions options;
    ArgVec<char*> args{"trtexec"};
    auto const missingDirectory = mPath + "/cache.jsonl";
    EXPECT_FALSE(writeTuningCacheHeader(missingDirectory, options, args.argc(), args.argv(), "test", ""));
    EXPECT_FALSE(writeTuningCacheIteration(missingDirectory, 0, "-a=on", false, "", {}, 1.0));
}

#if defined(__linux__)
TEST_F(TuningCacheTest, FlushFailuresAreReported)
{
    // /dev/full opens successfully but rejects writes, including buffered data flushed by close().
    std::ofstream file("/dev/full");
    ASSERT_TRUE(file.is_open());
    file.close();

    AllOptions options;
    ArgVec<char*> args{"trtexec"};
    EXPECT_FALSE(writeTuningCacheHeader("/dev/full", options, args.argc(), args.argv(), "test", ""));
    EXPECT_FALSE(writeTuningCacheIteration("/dev/full", 0, "-a=on", false, "", {}, 1.0));
}
#endif

TEST_F(TuningCacheTest, MissingOutputPathRemainsRelative)
{
    auto const relative = std::filesystem::path(mPath).filename();
    EXPECT_EQ(resolveAbsolutePath(relative.string()), relative.string());
}

TEST_F(TuningCacheTest, CheckerBlobArgumentRemainsUnchanged)
{
    AllOptions options;
    options.tuning.tuningExpr = "-a=[on|off]";
    // Use an existing file so path resolution would change the relative argument.
    auto const checkerBlobArg = "--loadCheckerBlob=" + std::filesystem::relative(mPath).string();
    ArgVec<char*> args{checkerBlobArg.c_str()};
    ASSERT_TRUE(writeTuningCacheHeader(mPath, options, args.argc(), args.argv(), "test", ""));
    auto const header = readTuningCacheHeader(mPath);
    ASSERT_TRUE(header);
    EXPECT_EQ(header->argv.back(), checkerBlobArg);
    auto const resumed = reconstructArgvFromCacheHeader(*header, "trtexec", mPath);
    EXPECT_EQ(resumed.at(1), checkerBlobArg);
}

TEST_F(TuningCacheTest, ResumePreservesReferencePairsAndRelativeTimingCache)
{
    AllOptions options;
    options.tuning.tuningExpr = "-a=[on|off] -fixed=off";
    auto const timingCacheArg = "--timingCacheFile=" + std::filesystem::relative(mPath).string();
    ArgVec<char*> args{"--tuneBuildRouteFile=routes.txt", "--tuningSearch=mixed-fast", "--onnx=model.onnx",
        "--saveEngine=best.plan", timingCacheArg.c_str(), "--refPair=0", "--loadInputs=x:input 0.bin,z:aux.bin",
        "--loadRefOutputs=y:ref 0.bin", "--refPair=1", "--loadInputs=x:input 1.bin", "--loadRefOutputs=y:ref 1.bin",
        "--accuracyAlgorithm=l2", "--accuracyThreshold=0.001"};
    ASSERT_TRUE(writeTuningCacheHeader(mPath, options, args.argc(), args.argv(), "test", ""));
    auto const header = readTuningCacheHeader(mPath);
    ASSERT_TRUE(header);
    auto const resumed = reconstructArgvFromCacheHeader(*header, "new-trtexec", mPath);
    std::vector<std::string> const expected{"new-trtexec", "--tuningSearch=mixed-fast",
        "--onnx=" + resolveAbsolutePath("model.onnx"), "--saveEngine=" + resolveAbsolutePath("best.plan"),
        timingCacheArg, "--refPair=0",
        "--loadInputs=x:" + resolveAbsolutePath("input 0.bin") + ",z:" + resolveAbsolutePath("aux.bin"),
        "--loadRefOutputs=y:" + resolveAbsolutePath("ref 0.bin"), "--refPair=1",
        "--loadInputs=x:" + resolveAbsolutePath("input 1.bin"),
        "--loadRefOutputs=y:" + resolveAbsolutePath("ref 1.bin"), "--accuracyAlgorithm=l2", "--accuracyThreshold=0.001",
        "--tuneBuildRoutes=-a=[on|off] -fixed=off", "--tuningCacheFile=" + mPath};
    EXPECT_EQ(resumed, expected);
}

TEST_F(TuningCacheTest, RejectsStructurallyInvalidRows)
{
    for (auto const* row : {R"({"iter":-1,"build_route":"-a=off","crash":false,"error_message":"","gpu_time":1})",
             R"({"iter":0,"build_route":12,"crash":false,"error_message":"","gpu_time":1})",
             R"({"iter":0,"build_route":"-a=off","crash":"false","error_message":"","gpu_time":1})",
             R"({"iter":0,"build_route":"-a=off","crash":false,"gpu_time":1})"})
    {
        SCOPED_TRACE(row);
        std::istringstream file(row);
        EXPECT_THROW((void) readCachedIterationResult(file), std::runtime_error);
    }
}
