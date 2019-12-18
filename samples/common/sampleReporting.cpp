/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <utility>
#include <algorithm>
#include <numeric>

#include "sampleOptions.h"
#include "sampleInference.h"
#include "sampleReporting.h"

using namespace nvinfer1;

namespace sample
{

namespace
{

//!
//! \brief Find percentile in an ascending sequence of timings
//!
template <typename T>
float findPercentile(float percentage, const std::vector<InferenceTime>& timings, const T& toFloat)
{
    const int all = static_cast<int>(timings.size());
    const int exclude = static_cast<int>((1 - percentage / 100) * all);
    if (0 <= exclude && exclude <= all)
    {
        return toFloat(timings[std::max(all - 1 - exclude, 0)]);
    }
    return std::numeric_limits<float>::infinity();
}

//!
//! \brief Find median in a sorted sequence of timings
//!
template <typename T>
float findMedian(const std::vector<InferenceTime>& timings, const T& toFloat)
{
    if (timings.empty())
    {
        return std::numeric_limits<float>::infinity();
    }

    const int m = timings.size()/2;
    if (timings.size() % 2)
    {
        return toFloat(timings[m]);
    }

    return (toFloat(timings[m-1]) + toFloat(timings[m])) / 2;
}

inline InferenceTime traceToTiming(const InferenceTrace& a)
{
    return InferenceTime((a.inEnd - a.inStart), (a.computeEnd - a.computeStart), (a.outEnd - a.outStart), (a.outEnd - a.inStart));
};

} // namespace

void printProlog(int warmups, int timings, float warmupMs, float benchTimeMs, std::ostream& os)
{
    os << "Warmup completed " << warmups << " queries over " << warmupMs << " ms" << std::endl;
    os << "Timing trace has " << timings << " queries over " << benchTimeMs / 1000 << " s" << std::endl;
}

void printTiming(const std::vector<InferenceTime>& timings, int runsPerAvg, std::ostream& os)
{
    int count = 0;
    InferenceTime sum;

    os << "Trace averages of " << runsPerAvg << " runs:" << std::endl;
    for (const auto& t : timings)
    {
        sum += t;

        if (++count == runsPerAvg)
        {
// clang off
            os << "Average on " << runsPerAvg << " runs - GPU latency: " << sum.compute / runsPerAvg
               << " ms - Host latency: " << sum.latency() / runsPerAvg << " ms (end to end "
               << sum.e2e / runsPerAvg << " ms)" << std::endl;
// clang on
            count = 0;
            sum.in = 0;
            sum.compute = 0;
            sum.out = 0;
            sum.e2e = 0;
        }
    }
}

void printEpilog(std::vector<InferenceTime> timings, float walltimeMs, float percentile, int queries, std::ostream& os)
{
    const InferenceTime totalTime = std::accumulate(timings.begin(), timings.end(), InferenceTime());

    const auto getLatency = [](const InferenceTime& t) { return t.latency(); };
    const auto cmpLatency = [](const InferenceTime& a, const InferenceTime& b) { return a.latency() < b.latency(); };
    std::sort(timings.begin(), timings.end(), cmpLatency);
    const float latencyMin = timings.front().latency();
    const float latencyMax = timings.back().latency();
    const float latencyMedian = findMedian(timings, getLatency);
    const float latencyPercentile = findPercentile(percentile, timings, getLatency);
    const float latencyThroughput = queries * timings.size() / walltimeMs * 1000;

    const auto getEndToEnd = [](const InferenceTime& t) { return t.e2e; };
    const auto cmpEndToEnd = [](const InferenceTime& a, const InferenceTime& b) { return a.e2e < b.e2e; };
    std::sort(timings.begin(), timings.end(), cmpEndToEnd);
    const float endToEndMin = timings.front().e2e;
    const float endToEndMax = timings.back().e2e;
    const float endToEndMedian = findMedian(timings, getEndToEnd);
    const float endToEndPercentile = findPercentile(percentile, timings, getEndToEnd);

    const auto getCompute = [](const InferenceTime& t) { return t.compute; };
    const auto cmpCompute = [](const InferenceTime& a, const InferenceTime& b) { return a.compute < b.compute; };
    std::sort(timings.begin(), timings.end(), cmpCompute);
    const float gpuMin = timings.front().compute;
    const float gpuMax = timings.back().compute;
    const float gpuMedian = findMedian(timings, getCompute);
    const float gpuPercentile = findPercentile(percentile, timings, getCompute);

// clang off
    os << "Host latency"                                                           << std::endl <<
          "min: "                << latencyMin                           << " ms "
          "(end to end "         << endToEndMin                          << " ms)" << std::endl <<
          "max: "                << latencyMax                           << " ms "
          "(end to end "         << endToEndMax                          << " ms)" << std::endl <<
          "mean: "               << totalTime.latency() / timings.size() << " ms "
          "(end to end "         << totalTime.e2e / timings.size()       << " ms)" << std::endl <<
          "median: "             << latencyMedian                        << " ms "
          "(end to end "         << endToEndMedian                       << " ms)" << std::endl <<
          "percentile: "         << latencyPercentile                    << " ms "
          "at "                  << percentile                           << "% "
          "(end to end "         << endToEndPercentile                   << " ms "
          "at "                  << percentile                           << "%)"   << std::endl <<
          "throughput: "         << latencyThroughput                    << " qps" << std::endl <<
          "walltime: "           << walltimeMs / 1000                    << " s"   << std::endl <<
          "GPU Compute"                                                            << std::endl <<
          "min: "                << gpuMin                               << " ms"  << std::endl <<
          "max: "                << gpuMax                               << " ms"  << std::endl <<
          "mean: "               << totalTime.compute / timings.size()   << " ms"  << std::endl <<
          "median: "             << gpuMedian                            << " ms"  << std::endl <<
          "percentile: "         << gpuPercentile                        << " ms "
          "at "                  << percentile                           << "%"    << std::endl <<
          "total compute time: " << totalTime.compute / 1000             << " s"   << std::endl;
// clang on
}

void printPerformanceReport(const std::vector<InferenceTrace>& trace, const ReportingOptions& reporting, float warmupMs, int queries, std::ostream& os)
{
    const auto isNotWarmup = [&warmupMs](const InferenceTrace& a) { return a.computeStart >= warmupMs; };
    const auto noWarmup = std::find_if(trace.begin(), trace.end(), isNotWarmup);
    const int warmups = noWarmup - trace.begin();
    const float benchTime = trace.back().outEnd - noWarmup->inStart;
    printProlog(warmups * queries, (trace.size() - warmups) * queries, warmupMs, benchTime, os);

    std::vector<InferenceTime> timings(trace.size() - warmups);
    std::transform(noWarmup, trace.end(), timings.begin(), traceToTiming);
    printTiming(timings, reporting.avgs, os);
    printEpilog(timings, benchTime, reporting.percentile, queries, os);
}

//! Printed format:
//! [ value, ...]
//! value ::= { "start in" : time, "end in" : time, "start compute" : time, "end compute" : time, "start out" : time,
//!             "in" : time, "compute" : time, "out" : time, "latency" : time, "end to end" : time}
//!
void exportJSONTrace(const std::vector<InferenceTrace>& trace, const std::string& fileName)
{
    std::ofstream os(fileName, std::ofstream::trunc);
    os << "[" << std::endl;
    const char* sep = "  ";
    for (const auto& t : trace)
    {
        const InferenceTime it(traceToTiming(t));
        os << sep << "{ ";
        sep = ", ";
// clang off
        os << "\"startInMs\" : "      << t.inStart      << sep << "\"endInMs\" : "      << t.inEnd      << sep
           << "\"startComputeMs\" : " << t.computeStart << sep << "\"endComputeMs\" : " << t.computeEnd << sep
           << "\"startOutMs\" : "     << t.outStart     << sep << "\"endOutMs\" : "     << t.outEnd     << sep
           << "\"inMs\" : "           << it.in          << sep << "\"computeMs\" : "    << it.compute   << sep
           << "\"outMs\" : "          << it.out         << sep << "\"latencyMs\" : "    << it.latency() << sep
           << "\"endToEndMs\" : "     << it.e2e         << " }"                                         << std::endl;
// clang on
    }
    os << "]" << std::endl;
}

void Profiler::reportLayerTime(const char* layerName, float timeMs)
{
    if (mIterator == mLayers.end())
    {
        const bool first = !mLayers.empty() && mLayers.begin()->name == layerName;
        mUpdatesCount += mLayers.empty() || first;
        if (first)
        {
            mIterator = mLayers.begin();
        }
        else
        {
            mLayers.emplace_back();
            mLayers.back().name = layerName;
            mIterator = mLayers.end() - 1;
        }
    }

    mIterator->timeMs += timeMs;
    ++mIterator;
}

void Profiler::print(std::ostream& os) const
{
    const std::string nameHdr("Layer");
    const std::string timeHdr("   Time (ms)");
    const std::string avgHdr("   Avg. Time (ms)");
    const std::string percentageHdr("   Time \%");

    const float totalTimeMs = getTotalTime();

    const auto cmpLayer = [](const LayerProfile& a, const LayerProfile& b)
    {
        return a.name.size() < b.name.size();
    };
    const auto longestName = std::max_element(mLayers.begin(), mLayers.end(), cmpLayer);
    const auto nameLength = std::max(longestName->name.size() + 1, nameHdr.size());
    const auto timeLength = timeHdr.size();
    const auto avgLength = avgHdr.size();
    const auto percentageLength = percentageHdr.size();

    os << std::endl << "=== Profile (" << mUpdatesCount << " iterations ) ===" << std::endl
       << std::setw(nameLength) << nameHdr << timeHdr << avgHdr << percentageHdr << std::endl;

    for (const auto& p : mLayers)
    {
// clang off
        os << std::setw(nameLength)                                             << p.name
           << std::setw(timeLength)       << std::fixed << std::setprecision(2) << p.timeMs
           << std::setw(avgLength)        << std::fixed << std::setprecision(2) << p.timeMs / mUpdatesCount
           << std::setw(percentageLength) << std::fixed << std::setprecision(1) << p.timeMs / totalTimeMs * 100
           << std::endl;
    }
    {
        os << std::setw(nameLength)                                             << "Total"
           << std::setw(timeLength)       << std::fixed << std::setprecision(2) << totalTimeMs
           << std::setw(avgLength)        << std::fixed << std::setprecision(2) << totalTimeMs / mUpdatesCount
           << std::setw(percentageLength) << std::fixed << std::setprecision(1) << 100.0
           << std::endl;
// clang on
    }
    os << std::endl;

}

void Profiler::exportJSONProfile(const std::string& fileName) const
{
    std::ofstream os(fileName, std::ofstream::trunc);
    os << "[" << std::endl
       << "  { \"count\" : " << mUpdatesCount << " }" << std::endl;

    const auto totalTimeMs = getTotalTime();

    for (const auto& l : mLayers)
    {
// clang off
        os << ", {" << " \"name\" : \""      << l.name << "\""
                       ", \"timeMs\" : "     << l.timeMs
           <<          ", \"averageMs\" : "  << l.timeMs / mUpdatesCount
           <<          ", \"percentage\" : " << l.timeMs / totalTimeMs * 100
           << " }"  << std::endl;
// clang on
    }
    os << "]" << std::endl;
}

void dumpInputs(const nvinfer1::IExecutionContext& context, const Bindings& bindings, std::ostream& os)
{
    os << "Input Tensors:" << std::endl;
    bindings.dumpInputs(context, os);
}

void dumpOutputs(const nvinfer1::IExecutionContext& context, const Bindings& bindings, std::ostream& os)
{
    os << "Output Tensors:" << std::endl;
    bindings.dumpOutputs(context, os);
}

void exportJSONOutput(const nvinfer1::IExecutionContext& context, const Bindings& bindings, const std::string& fileName)
{
    std::ofstream os(fileName, std::ofstream::trunc);
    std::string sep ="  ";
    const auto output = bindings.getOutputBindings();
    os << "[" << std::endl;
    for (const auto& binding : output)
    {
// clang off
        os << sep << "{ \"name\" : \"" << binding.first << "\"" << std::endl;
        sep = ", ";
        os << "  " << sep << "\"dimensions\" : \"";
        bindings.dumpBindingDimensions(binding.second, context, os);
        os << "\"" << std::endl;
        os << "  " << sep << "\"values\" : [ ";
        bindings.dumpBindingValues(binding.second, os, sep);
        os << " ]" << std::endl << "  }"  << std::endl;
// clang on
    }
    os << "]" << std::endl;
}

} // namespace sample
