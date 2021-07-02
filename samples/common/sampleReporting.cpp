/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <algorithm>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <utility>

#include "sampleInference.h"
#include "sampleOptions.h"
#include "sampleReporting.h"

using namespace nvinfer1;

namespace sample
{

namespace
{

//!
//! \brief Find percentile in an ascending sequence of timings
//! \note percentile must be in [0, 100]. Otherwise, an exception is thrown.
//!
template <typename T>
float findPercentile(float percentile, const std::vector<InferenceTime>& timings, const T& toFloat)
{
    const int all = static_cast<int>(timings.size());
    const int exclude = static_cast<int>((1 - percentile / 100) * all);
    if (timings.empty())
    {
        return std::numeric_limits<float>::infinity();
    }
    if (percentile < 0.0f || percentile > 100.0f)
    {
        throw std::runtime_error("percentile is not in [0, 100]!");
    }
    return toFloat(timings[std::max(all - 1 - exclude, 0)]);
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

    const int m = timings.size() / 2;
    if (timings.size() % 2)
    {
        return toFloat(timings[m]);
    }

    return (toFloat(timings[m - 1]) + toFloat(timings[m])) / 2;
}

inline InferenceTime traceToTiming(const InferenceTrace& a)
{
    return InferenceTime((a.enqEnd - a.enqStart), (a.h2dEnd - a.h2dStart), (a.computeEnd - a.computeStart),
        (a.d2hEnd - a.d2hStart), (a.d2hEnd - a.h2dStart));
}

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

    os << std::endl;
    os << "=== Trace details ===" << std::endl;
    os << "Trace averages of " << runsPerAvg << " runs:" << std::endl;
    for (const auto& t : timings)
    {
        sum += t;

        if (++count == runsPerAvg)
        {
            // clang-format off
            os << "Average on " << runsPerAvg << " runs - GPU latency: " << sum.compute / runsPerAvg
               << " ms - Host latency: " << sum.latency() / runsPerAvg << " ms (end to end " << sum.e2e / runsPerAvg
               << " ms, enqueue " << sum.enq / runsPerAvg << " ms)" << std::endl;
            // clang-format on
            count = 0;
            sum.enq = 0;
            sum.h2d = 0;
            sum.compute = 0;
            sum.d2h = 0;
            sum.e2e = 0;
        }
    }
}

void printMetricExplanations(std::ostream& os)
{
    os << std::endl;
    os << "=== Explanations of the performance metrics ===" << std::endl;
    os << "Total Host Walltime: the host walltime from when the first query (after warmups) is enqueued to when the "
          "last query is completed."
       << std::endl;
    os << "GPU Compute Time: the GPU latency to execute the kernels for a query." << std::endl;
    os << "Total GPU Compute Time: the summation of the GPU Compute Time of all the queries. If this is significantly "
          "shorter than Total Host Walltime, the GPU may be under-utilized because of host-side overheads or data "
          "transfers."
       << std::endl;
    os << "Throughput: the observed throughput computed by dividing the number of queries by the Total Host Walltime. "
          "If this is significantly lower than the reciprocal of GPU Compute Time, the GPU may be under-utilized "
          "because of host-side overheads or data transfers."
       << std::endl;
    os << "Enqueue Time: the host latency to enqueue a query. If this is longer than GPU Compute Time, the GPU may be "
          "under-utilized."
       << std::endl;
    os << "H2D Latency: the latency for host-to-device data transfers for input tensors of a single query."
       << std::endl;
    os << "D2H Latency: the latency for device-to-host data transfers for output tensors of a single query."
       << std::endl;
    os << "Latency: the summation of H2D Latency, GPU Compute Time, and D2H Latency. This is the latency to infer a "
          "single query."
       << std::endl;
    os << "End-to-End Host Latency: the duration from when the H2D of a query is called to when the D2H of the same "
          "query is completed, which includes the latency to wait for the completion of the previous query. This is "
          "the latency of a query if multiple queries are enqueued consecutively."
       << std::endl;
}

PerformanceResult getPerformanceResult(const std::vector<InferenceTime>& timings,
    std::function<float(const InferenceTime&)> metricGetter, float percentile)
{
    const auto metricComparator
        = [metricGetter](const InferenceTime& a, const InferenceTime& b) { return metricGetter(a) < metricGetter(b); };
    const auto metricAccumulator = [metricGetter](float acc, const InferenceTime& a) { return acc + metricGetter(a); };
    std::vector<InferenceTime> newTimings = timings;
    std::sort(newTimings.begin(), newTimings.end(), metricComparator);
    PerformanceResult result;
    result.min = metricGetter(newTimings.front());
    result.max = metricGetter(newTimings.back());
    result.mean = std::accumulate(newTimings.begin(), newTimings.end(), 0.0f, metricAccumulator) / newTimings.size();
    result.median = findMedian(newTimings, metricGetter);
    result.percentile = findPercentile(percentile, newTimings, metricGetter);
    return result;
}

void printEpilog(const std::vector<InferenceTime>& timings, float walltimeMs, float percentile, int batchSize,
    std::ostream& osInfo, std::ostream& osWarning, std::ostream& osVerbose)
{
    const float throughput = batchSize * timings.size() / walltimeMs * 1000;

    const auto getLatency = [](const InferenceTime& t) { return t.latency(); };
    const auto latencyResult = getPerformanceResult(timings, getLatency, percentile);

    const auto getEndToEnd = [](const InferenceTime& t) { return t.e2e; };
    const auto e2eLatencyResult = getPerformanceResult(timings, getEndToEnd, percentile);

    const auto getEnqueue = [](const InferenceTime& t) { return t.enq; };
    const auto enqueueResult = getPerformanceResult(timings, getEnqueue, percentile);

    const auto getH2d = [](const InferenceTime& t) { return t.h2d; };
    const auto h2dResult = getPerformanceResult(timings, getH2d, percentile);

    const auto getCompute = [](const InferenceTime& t) { return t.compute; };
    const auto gpuComputeResult = getPerformanceResult(timings, getCompute, percentile);

    const auto getD2h = [](const InferenceTime& t) { return t.d2h; };
    const auto d2hResult = getPerformanceResult(timings, getD2h, percentile);

    const auto toPerfString = [percentile](const PerformanceResult& r) {
        std::stringstream s;
        s << "min = " << r.min << " ms, max = " << r.max << " ms, mean = " << r.mean << " ms, "
          << "median = " << r.median << " ms, percentile(" << percentile << "%) = " << r.percentile << " ms";
        return s.str();
    };

    osInfo << std::endl;
    osInfo << "=== Performance summary ===" << std::endl;
    osInfo << "Throughput: " << throughput << " qps" << std::endl;
    osInfo << "Latency: " << toPerfString(latencyResult) << std::endl;
    osInfo << "End-to-End Host Latency: " << toPerfString(e2eLatencyResult) << std::endl;
    osInfo << "Enqueue Time: " << toPerfString(enqueueResult) << std::endl;
    osInfo << "H2D Latency: " << toPerfString(h2dResult) << std::endl;
    osInfo << "GPU Compute Time: " << toPerfString(gpuComputeResult) << std::endl;
    osInfo << "D2H Latency: " << toPerfString(d2hResult) << std::endl;
    osInfo << "Total Host Walltime: " << walltimeMs / 1000 << " s" << std::endl;
    osInfo << "Total GPU Compute Time: " << gpuComputeResult.mean * timings.size() / 1000 << " s" << std::endl;

    // Report warnings if the throughput is bound by other factors than GPU
    // Compute Time.
    constexpr float enqueueBoundReportingThreshold{0.8f};
    if (enqueueResult.median > enqueueBoundReportingThreshold * gpuComputeResult.median)
    {
        osWarning
            << "* Throughput may be bound by Enqueue Time rather than GPU Compute and the GPU may be under-utilized."
            << std::endl;
        osWarning << "  If not already in use, --useCudaGraph (utilize CUDA graphs where possible) may increase the "
                     "throughput."
                  << std::endl;
    }
    if (h2dResult.median >= gpuComputeResult.median)
    {
        osWarning << "* Throughput may be bound by host-to-device transfers for the inputs rather than GPU Compute and "
                     "the GPU may be under-utilized."
                  << std::endl;
        osWarning << "  Add --noDataTransfers flag to disable data transfers." << std::endl;
    }
    if (d2hResult.median >= gpuComputeResult.median)
    {
        osWarning << "* Throughput may be bound by device-to-host transfers for the outputs rather than GPU Compute "
                     "and the GPU may be under-utilized."
                  << std::endl;
        osWarning << "  Add --noDataTransfers flag to disable data transfers." << std::endl;
    }

    // Explain what the metrics mean.
    osInfo << "Explanations of the performance metrics are printed in the verbose logs." << std::endl;
    printMetricExplanations(osVerbose);

    osInfo << std::endl;
}

void printPerformanceReport(const std::vector<InferenceTrace>& trace, const ReportingOptions& reporting, float warmupMs,
    int batchSize, std::ostream& osInfo, std::ostream& osWarning, std::ostream& osVerbose)
{
    const auto isNotWarmup = [&warmupMs](const InferenceTrace& a) { return a.computeStart >= warmupMs; };
    const auto noWarmup = std::find_if(trace.begin(), trace.end(), isNotWarmup);
    const int warmups = noWarmup - trace.begin();
    const float benchTime = trace.back().d2hEnd - noWarmup->h2dStart;
    // when implicit batch used, batchSize = options.inference.batch, which is parsed through --batch
    // when explicit batch used, batchSize = options.inference.batch = 0
    // treat inference with explicit batch as a single query and report the throughput
    batchSize = batchSize ? batchSize : 1;
    printProlog(warmups * batchSize, (trace.size() - warmups) * batchSize, warmupMs, benchTime, osInfo);

    std::vector<InferenceTime> timings(trace.size() - warmups);
    std::transform(noWarmup, trace.end(), timings.begin(), traceToTiming);
    printTiming(timings, reporting.avgs, osInfo);
    printEpilog(timings, benchTime, reporting.percentile, batchSize, osInfo, osWarning, osVerbose);

    if (!reporting.exportTimes.empty())
    {
        exportJSONTrace(trace, reporting.exportTimes);
    }
}

//! Printed format:
//! [ value, ...]
//! value ::= { "start enq : time, "end enq" : time, "start h2d" : time, "end h2d" : time, "start compute" : time,
//!             "end compute" : time, "start d2h" : time, "end d2h" : time, "h2d" : time, "compute" : time,
//!             "d2h" : time, "latency" : time, "end to end" : time }
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
        // clang-format off
        os << "\"startEnqMs\" : "     << t.enqStart     << sep << "\"endEnqMs\" : "     << t.enqEnd     << sep
           << "\"startH2dMs\" : "     << t.h2dStart     << sep << "\"endH2dMs\" : "     << t.h2dEnd     << sep
           << "\"startComputeMs\" : " << t.computeStart << sep << "\"endComputeMs\" : " << t.computeEnd << sep
           << "\"startD2hMs\" : "     << t.d2hStart     << sep << "\"endD2hMs\" : "     << t.d2hEnd     << sep
           << "\"h2dMs\" : "          << it.h2d         << sep << "\"computeMs\" : "    << it.compute   << sep
           << "\"d2hMs\" : "          << it.d2h         << sep << "\"latencyMs\" : "    << it.latency() << sep
           << "\"endToEndMs\" : "     << it.e2e         << " }"                                         << std::endl;
        // clang-format on
    }
    os << "]" << std::endl;
}

void Profiler::reportLayerTime(const char* layerName, float timeMs) noexcept
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

void Profiler::print(std::ostream& os) const noexcept
{
    const std::string nameHdr("Layer");
    const std::string timeHdr("   Time (ms)");
    const std::string avgHdr("   Avg. Time (ms)");
    const std::string percentageHdr("   Time %");

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

    os << std::endl
       << "=== Profile (" << mUpdatesCount << " iterations ) ===" << std::endl
       << std::setw(nameLength) << nameHdr << timeHdr << avgHdr << percentageHdr << std::endl;

    for (const auto& p : mLayers)
    {
        // clang-format off
        os << std::setw(nameLength) << p.name << std::setw(timeLength) << std::fixed << std::setprecision(2) << p.timeMs
           << std::setw(avgLength) << std::fixed << std::setprecision(4) << p.timeMs / mUpdatesCount
           << std::setw(percentageLength) << std::fixed << std::setprecision(1) << p.timeMs / totalTimeMs * 100
           << std::endl;
    }
    {
        os << std::setw(nameLength) << "Total" << std::setw(timeLength) << std::fixed << std::setprecision(2)
           << totalTimeMs << std::setw(avgLength) << std::fixed << std::setprecision(4) << totalTimeMs / mUpdatesCount
           << std::setw(percentageLength) << std::fixed << std::setprecision(1) << 100.0 << std::endl;
        // clang-format on
    }
    os << std::endl;
}

void Profiler::exportJSONProfile(const std::string& fileName) const noexcept
{
    std::ofstream os(fileName, std::ofstream::trunc);
    os << "[" << std::endl << "  { \"count\" : " << mUpdatesCount << " }" << std::endl;

    const auto totalTimeMs = getTotalTime();

    for (const auto& l : mLayers)
    {
        // clang-format off
        os << ", {" << " \"name\" : \""      << l.name << "\""
                       ", \"timeMs\" : "     << l.timeMs
           <<          ", \"averageMs\" : "  << l.timeMs / mUpdatesCount
           <<          ", \"percentage\" : " << l.timeMs / totalTimeMs * 100
           << " }"  << std::endl;
        // clang-format on
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

void exportJSONOutput(
    const nvinfer1::IExecutionContext& context, const Bindings& bindings, const std::string& fileName, int32_t batch)
{
    std::ofstream os(fileName, std::ofstream::trunc);
    std::string sep = "  ";
    const auto output = bindings.getOutputBindings();
    os << "[" << std::endl;
    for (const auto& binding : output)
    {
        // clang-format off
        os << sep << "{ \"name\" : \"" << binding.first << "\"" << std::endl;
        sep = ", ";
        os << "  " << sep << "\"dimensions\" : \"";
        bindings.dumpBindingDimensions(binding.second, context, os);
        os << "\"" << std::endl;
        os << "  " << sep << "\"values\" : [ ";
        bindings.dumpBindingValues(context, binding.second, os, sep, batch);
        os << " ]" << std::endl << "  }" << std::endl;
        // clang-format on
    }
    os << "]" << std::endl;
}

} // namespace sample
