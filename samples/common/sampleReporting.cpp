/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
float findPercentile(float percentile, std::vector<InferenceTime> const& timings, T const& toFloat)
{
    int32_t const all = static_cast<int32_t>(timings.size());
    int32_t const exclude = static_cast<int32_t>((1 - percentile / 100) * all);
    if (timings.empty())
    {
        return std::numeric_limits<float>::infinity();
    }
    if (percentile < 0.F || percentile > 100.F)
    {
        throw std::runtime_error("percentile is not in [0, 100]!");
    }
    return toFloat(timings[std::max(all - 1 - exclude, 0)]);
}

//!
//! \brief Find median in a sorted sequence of timings
//!
template <typename T>
float findMedian(std::vector<InferenceTime> const& timings, T const& toFloat)
{
    if (timings.empty())
    {
        return std::numeric_limits<float>::infinity();
    }

    int32_t const m = timings.size() / 2;
    if (timings.size() % 2)
    {
        return toFloat(timings[m]);
    }

    return (toFloat(timings[m - 1]) + toFloat(timings[m])) / 2;
}

//!
//! \brief Find coefficient of variance (which is std / mean) in a sorted sequence of timings given the mean
//!
template <typename T>
float findCoeffOfVariance(std::vector<InferenceTime> const& timings, T const& toFloat, float mean)
{
    if (timings.empty())
    {
        return 0;
    }

    if (mean == 0.F)
    {
        return std::numeric_limits<float>::infinity();
    }

    auto const metricAccumulator = [toFloat, mean](float acc, InferenceTime const& a) {
        float const diff = toFloat(a) - mean;
        return acc + diff * diff;
    };
    float const variance = std::accumulate(timings.begin(), timings.end(), 0.F, metricAccumulator) / timings.size();

    return std::sqrt(variance) / mean * 100.F;
}

inline InferenceTime traceToTiming(const InferenceTrace& a)
{
    return InferenceTime(
        (a.enqEnd - a.enqStart), (a.h2dEnd - a.h2dStart), (a.computeEnd - a.computeStart), (a.d2hEnd - a.d2hStart));
}

inline std::string dimsToString(Dims const& shape)
{
    std::stringstream ss;

    if (shape.nbDims == 0)
    {
        ss << "scalar";
    }
    else
    {
        for (int32_t i = 0; i < shape.nbDims; i++)
        {
            ss << shape.d[i] << (i != shape.nbDims - 1 ? "x" : "");
        }
    }
    return ss.str();
}

} // namespace

void printProlog(int32_t warmups, int32_t timings, float warmupMs, float benchTimeMs, std::ostream& os)
{
    os << "Warmup completed " << warmups << " queries over " << warmupMs << " ms" << std::endl;
    os << "Timing trace has " << timings << " queries over " << benchTimeMs / 1000 << " s" << std::endl;
}

void printTiming(std::vector<InferenceTime> const& timings, int32_t runsPerAvg, std::ostream& os)
{
    int64_t count = 0;
    InferenceTime sum;

    os << std::endl;
    os << "=== Trace details ===" << std::endl;
    os << "Trace averages of " << runsPerAvg << " runs:" << std::endl;

    // Show only the first N lines and the last N lines, where N = kTIMING_PRINT_THRESHOLD.
    constexpr int64_t kTIMING_PRINT_THRESHOLD{200};
    int64_t const maxNbTimings{kTIMING_PRINT_THRESHOLD * runsPerAvg};

    for (int64_t idx = 0, size = timings.size(); idx < size; ++idx)
    {
        // Omit some latency printing to avoid very long logs.
        if (size > 2 * maxNbTimings && idx == maxNbTimings)
        {
            os << "... Omitting " << (size - 2 * maxNbTimings) << " lines" << std::endl;
            idx = size - kTIMING_PRINT_THRESHOLD * runsPerAvg - 1;
        }

        sum += timings[idx];

        if (++count == runsPerAvg)
        {
            // clang-format off
            os << "Average on " << runsPerAvg << " runs - GPU latency: " << sum.compute / runsPerAvg
               << " ms - Host latency: " << sum.latency() / runsPerAvg << " ms (enqueue " << sum.enq / runsPerAvg
               << " ms)" << std::endl;
            // clang-format on
            count = 0;
            sum.enq = 0;
            sum.h2d = 0;
            sum.compute = 0;
            sum.d2h = 0;
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
}

PerformanceResult getPerformanceResult(std::vector<InferenceTime> const& timings,
    std::function<float(InferenceTime const&)> metricGetter, std::vector<float> const& percentiles)
{
    auto const metricComparator
        = [metricGetter](InferenceTime const& a, InferenceTime const& b) { return metricGetter(a) < metricGetter(b); };
    auto const metricAccumulator = [metricGetter](float acc, InferenceTime const& a) { return acc + metricGetter(a); };
    std::vector<InferenceTime> newTimings = timings;
    std::sort(newTimings.begin(), newTimings.end(), metricComparator);
    PerformanceResult result;
    result.min = metricGetter(newTimings.front());
    result.max = metricGetter(newTimings.back());
    result.mean = std::accumulate(newTimings.begin(), newTimings.end(), 0.0F, metricAccumulator) / newTimings.size();
    result.median = findMedian(newTimings, metricGetter);
    for (auto percentile : percentiles)
    {
        result.percentiles.emplace_back(findPercentile(percentile, newTimings, metricGetter));
    }
    result.coeffVar = findCoeffOfVariance(newTimings, metricGetter, result.mean);
    return result;
}

void printEpilog(std::vector<InferenceTime> const& timings, float walltimeMs, std::vector<float> const& percentiles,
    int32_t batchSize, int32_t infStreams, std::ostream& osInfo, std::ostream& osWarning, std::ostream& osVerbose)
{
    float const throughput = batchSize * timings.size() / walltimeMs * 1000;

    auto const getLatency = [](InferenceTime const& t) { return t.latency(); };
    auto const latencyResult = getPerformanceResult(timings, getLatency, percentiles);

    auto const getEnqueue = [](InferenceTime const& t) { return t.enq; };
    auto const enqueueResult = getPerformanceResult(timings, getEnqueue, percentiles);

    auto const getH2d = [](InferenceTime const& t) { return t.h2d; };
    auto const h2dResult = getPerformanceResult(timings, getH2d, percentiles);

    auto const getCompute = [](InferenceTime const& t) { return t.compute; };
    auto const gpuComputeResult = getPerformanceResult(timings, getCompute, percentiles);

    auto const getD2h = [](InferenceTime const& t) { return t.d2h; };
    auto const d2hResult = getPerformanceResult(timings, getD2h, percentiles);

    auto const toPerfString = [&](const PerformanceResult& r) {
        std::stringstream s;
        s << "min = " << r.min << " ms, max = " << r.max << " ms, mean = " << r.mean << " ms, "
          << "median = " << r.median << " ms";
        for (int32_t i = 0, n = percentiles.size(); i < n; ++i)
        {
            s << ", percentile(" << percentiles[i] << "%) = " << r.percentiles[i] << " ms";
        }
        return s.str();
    };

    osInfo << std::endl;
    osInfo << "=== Performance summary ===" << std::endl;
    osInfo << "Throughput: " << throughput << " qps" << std::endl;
    osInfo << "Latency: " << toPerfString(latencyResult) << std::endl;
    osInfo << "Enqueue Time: " << toPerfString(enqueueResult) << std::endl;
    osInfo << "H2D Latency: " << toPerfString(h2dResult) << std::endl;
    osInfo << "GPU Compute Time: " << toPerfString(gpuComputeResult) << std::endl;
    osInfo << "D2H Latency: " << toPerfString(d2hResult) << std::endl;
    osInfo << "Total Host Walltime: " << walltimeMs / 1000 << " s" << std::endl;
    osInfo << "Total GPU Compute Time: " << gpuComputeResult.mean * timings.size() / 1000 << " s" << std::endl;

    // Report warnings if the throughput is bound by other factors than GPU Compute Time.
    constexpr float kENQUEUE_BOUND_REPORTING_THRESHOLD{0.8F};
    if (enqueueResult.median > kENQUEUE_BOUND_REPORTING_THRESHOLD * gpuComputeResult.median)
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

    // Report warnings if the GPU Compute Time is unstable.
    constexpr float kUNSTABLE_PERF_REPORTING_THRESHOLD{1.0F};
    if (gpuComputeResult.coeffVar > kUNSTABLE_PERF_REPORTING_THRESHOLD)
    {
        osWarning << "* GPU compute time is unstable, with coefficient of variance = " << gpuComputeResult.coeffVar
                  << "%." << std::endl;
        osWarning << "  If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the "
                  << "stability." << std::endl;
    }

    // Report warnings if multiple inference streams are used.
    if (infStreams > 1)
    {
        osWarning << "* Multiple inference streams are used. Latencies may not be accurate since inferences may run in "
                  << "  parallel. Please use \"Throughput\" as the performance metric instead." << std::endl;
    }

    // Explain what the metrics mean.
    osInfo << "Explanations of the performance metrics are printed in the verbose logs." << std::endl;
    printMetricExplanations(osVerbose);

    osInfo << std::endl;
}

void printPerformanceReport(std::vector<InferenceTrace> const& trace, ReportingOptions const& reportingOpts,
    InferenceOptions const& infOpts, std::ostream& osInfo, std::ostream& osWarning, std::ostream& osVerbose)
{
    int32_t batchSize = infOpts.batch;
    float const warmupMs = infOpts.warmup;
    auto const isNotWarmup = [&warmupMs](const InferenceTrace& a) { return a.computeStart >= warmupMs; };
    auto const noWarmup = std::find_if(trace.begin(), trace.end(), isNotWarmup);
    int32_t const warmups = noWarmup - trace.begin();
    float const benchTime = trace.back().d2hEnd - noWarmup->h2dStart;
    // treat inference with explicit batch as a single query and report the throughput
    batchSize = batchSize ? batchSize : 1;
    printProlog(warmups * batchSize, (trace.size() - warmups) * batchSize, warmupMs, benchTime, osInfo);

    std::vector<InferenceTime> timings(trace.size() - warmups);
    std::transform(noWarmup, trace.end(), timings.begin(), traceToTiming);
    printTiming(timings, reportingOpts.avgs, osInfo);
    printEpilog(
        timings, benchTime, reportingOpts.percentiles, batchSize, infOpts.infStreams, osInfo, osWarning, osVerbose);

    if (!reportingOpts.exportTimes.empty())
    {
        exportJSONTrace(trace, reportingOpts.exportTimes, warmups);
    }
}

//! Printed format:
//! [ value, ...]
//! value ::= { "start enq : time, "end enq" : time, "start h2d" : time, "end h2d" : time, "start compute" : time,
//!             "end compute" : time, "start d2h" : time, "end d2h" : time, "h2d" : time, "compute" : time,
//!             "d2h" : time, "latency" : time }
//!
void exportJSONTrace(std::vector<InferenceTrace> const& trace, std::string const& fileName, int32_t const nbWarmups)
{
    std::ofstream os(fileName, std::ofstream::trunc);
    os << "[" << std::endl;
    char const* sep = "  ";
    for (auto iter = trace.begin() + nbWarmups; iter < trace.end(); ++iter)
    {
        auto const& t = *iter;
        InferenceTime const it(traceToTiming(t));
        os << sep << "{ ";
        sep = ", ";
        // clang-format off
        os << "\"startEnqMs\" : "     << t.enqStart     << sep << "\"endEnqMs\" : "     << t.enqEnd     << sep
           << "\"startH2dMs\" : "     << t.h2dStart     << sep << "\"endH2dMs\" : "     << t.h2dEnd     << sep
           << "\"startComputeMs\" : " << t.computeStart << sep << "\"endComputeMs\" : " << t.computeEnd << sep
           << "\"startD2hMs\" : "     << t.d2hStart     << sep << "\"endD2hMs\" : "     << t.d2hEnd     << sep
           << "\"h2dMs\" : "          << it.h2d         << sep << "\"computeMs\" : "    << it.compute   << sep
           << "\"d2hMs\" : "          << it.d2h         << sep << "\"latencyMs\" : "    << it.latency() << " }"
           << std::endl;
        // clang-format on
    }
    os << "]" << std::endl;
}

void Profiler::reportLayerTime(char const* layerName, float timeMs) noexcept
{
    if (mIterator == mLayers.end())
    {
        bool const first = !mLayers.empty() && mLayers.begin()->name == layerName;
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

    mIterator->timeMs.push_back(timeMs);
    ++mIterator;
}

void Profiler::print(std::ostream& os) const noexcept
{
    std::string const nameHdr("   Layer");
    std::string const timeHdr("   Time(ms)");
    std::string const avgHdr("     Avg.(ms)");
    std::string const medHdr("   Median(ms)");
    std::string const percentageHdr("   Time(%)");

    float const totalTimeMs = getTotalTime();

    auto const timeLength = timeHdr.size();
    auto const avgLength = avgHdr.size();
    auto const medLength = medHdr.size();
    auto const percentageLength = percentageHdr.size();

    os << std::endl
       << "=== Profile (" << mUpdatesCount << " iterations ) ===" << std::endl
       << timeHdr << avgHdr << medHdr << percentageHdr << nameHdr << std::endl;

    for (auto const& p : mLayers)
    {
        if (p.timeMs.empty() || getTotalTime(p) == 0.F)
        {
            // there is no point to print profiling for layer that didn't run at all
            continue;
        }
        // clang-format off
        os << std::setw(timeLength) << std::fixed << std::setprecision(2) << getTotalTime(p)
           << std::setw(avgLength) << std::fixed << std::setprecision(4) << getAvgTime(p)
           << std::setw(medLength) << std::fixed << std::setprecision(4) << getMedianTime(p)
           << std::setw(percentageLength) << std::fixed << std::setprecision(1) << getTotalTime(p) / totalTimeMs * 100
           << "   " << p.name << std::endl;
    }
    {
        os << std::setw(timeLength) << std::fixed << std::setprecision(2)
           << totalTimeMs << std::setw(avgLength) << std::fixed << std::setprecision(4) << totalTimeMs / mUpdatesCount
           << std::setw(medLength) << std::fixed << std::setprecision(4) << getMedianTime()
           << std::setw(percentageLength) << std::fixed << std::setprecision(1) << 100.0
           << "   Total" << std::endl;
        // clang-format on
    }
    os << std::endl;
}

void Profiler::exportJSONProfile(std::string const& fileName) const noexcept
{
    std::ofstream os(fileName, std::ofstream::trunc);
    os << "[" << std::endl << "  { \"count\" : " << mUpdatesCount << " }" << std::endl;

    auto const totalTimeMs = getTotalTime();

    for (auto const& l : mLayers)
    {
        // clang-format off
        os << ", {" << R"( "name" : ")"      << l.name << R"(")"
                       R"(, "timeMs" : )"     << getTotalTime(l)
           <<          R"(, "averageMs" : )"  << getAvgTime(l)
           <<          R"(, "medianMs" : )"  << getMedianTime(l)
           <<          R"(, "percentage" : )" << getTotalTime(l) / totalTimeMs * 100
           << " }"  << std::endl;
        // clang-format on
    }
    os << "]" << std::endl;
}

void dumpInputs(nvinfer1::IExecutionContext const& context, Bindings const& bindings, std::ostream& os)
{
    os << "Input Tensors:" << std::endl;
    bindings.dumpInputs(context, os);
}

template <typename ContextType>
void dumpOutputs(ContextType const& context, Bindings const& bindings, std::ostream& os)
{
    os << "Output Tensors:" << std::endl;
    bindings.dumpOutputs(context, os);
}

template
void dumpOutputs(nvinfer1::IExecutionContext const& context, Bindings const& bindings, std::ostream& os);
template
void dumpOutputs(nvinfer1::safe::IExecutionContext const& context, Bindings const& bindings, std::ostream& os);

template <typename ContextType>
void dumpRawBindingsToFiles(ContextType const& context, Bindings const& bindings, std::ostream& os)
{
    bindings.dumpRawBindingToFiles(context, os);
}

template
void dumpRawBindingsToFiles(nvinfer1::IExecutionContext const& context, Bindings const& bindings, std::ostream& os);

template
void dumpRawBindingsToFiles(nvinfer1::safe::IExecutionContext const& context, Bindings const& bindings, std::ostream& os);

template <typename ContextType>
void exportJSONOutput(
    ContextType const& context, Bindings const& bindings, std::string const& fileName, int32_t batch)
{
    std::ofstream os(fileName, std::ofstream::trunc);
    std::string sep = "  ";
    auto const output = bindings.getOutputBindings();
    os << "[" << std::endl;
    for (auto const& binding : output)
    {
        // clang-format off
        os << sep << R"({ "name" : ")" << binding.first << "\"" << std::endl;
        sep = ", ";
        os << "  " << sep << R"("dimensions" : ")";
        bindings.dumpBindingDimensions(binding.first, context, os);
        os << "\"" << std::endl;
        os << "  " << sep << "\"values\" : [ ";
        bindings.dumpBindingValues(context, binding.second, os, sep, batch);
        os << " ]" << std::endl << "  }" << std::endl;
        // clang-format on
    }
    os << "]" << std::endl;
}

template
void exportJSONOutput(nvinfer1::IExecutionContext const& context, Bindings const& bindings, std::string const& fileName, int32_t batch);

template void exportJSONOutput(nvinfer1::safe::IExecutionContext const& context, Bindings const& bindings,
    std::string const& fileName, int32_t batch);

void printLayerInfo(
    ReportingOptions const& reporting, nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* context)
{
    if (reporting.layerInfo)
    {
        sample::gLogInfo << "Layer Information:" << std::endl;
        sample::gLogInfo << getLayerInformation(engine, context, nvinfer1::LayerInformationFormat::kONELINE)
                         << std::flush;
    }
    if (!reporting.exportLayerInfo.empty())
    {
        std::ofstream os(reporting.exportLayerInfo, std::ofstream::trunc);
        os << getLayerInformation(engine, context, nvinfer1::LayerInformationFormat::kJSON) << std::flush;
    }
}

void printOptimizationProfileInfo(ReportingOptions const& reporting, nvinfer1::ICudaEngine const* engine)
{
    if (reporting.optProfileInfo)
    {
        sample::gLogInfo << "Optimization Profile Information:" << std::endl;
        for (int32_t i = 0; i < engine->getNbOptimizationProfiles(); i++)
        {
            for (int32_t j = 0, e = engine->getNbIOTensors(); j < e; j++)
            {
                auto const tensorName = engine->getIOTensorName(j);

                if (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT)
                {
                    auto tensorMinShape = engine->getProfileShape(tensorName, i, nvinfer1::OptProfileSelector::kMIN);
                    auto tensorOptShape = engine->getProfileShape(tensorName, i, nvinfer1::OptProfileSelector::kOPT);
                    auto tensorMaxShape = engine->getProfileShape(tensorName, i, nvinfer1::OptProfileSelector::kMAX);

                    sample::gLogInfo << "Model input " << tensorName << " (profile " << i << "): "
                                     << "min=" << dimsToString(tensorMinShape)
                                     << ", opt=" << dimsToString(tensorOptShape)
                                     << ", max=" << dimsToString(tensorMaxShape) << std::endl;
                }
            }
        }
    }
}

void printPerformanceProfile(ReportingOptions const& reporting, InferenceEnvironment& iEnv)
{
    if (reporting.profile)
    {
        iEnv.profiler->print(sample::gLogInfo);
    }
    if (!reporting.exportProfile.empty())
    {
        iEnv.profiler->exportJSONProfile(reporting.exportProfile);
    }

    // Print an warning about total per-layer latency when auxiliary streams are used.
    if (!iEnv.safe && (reporting.profile || !reporting.exportProfile.empty()))
    {
        int32_t const nbAuxStreams = iEnv.engine.get()->getNbAuxStreams();
        if (nbAuxStreams > 0)
        {
            sample::gLogWarning << "The engine uses " << nbAuxStreams << " auxiliary streams, so the \"Total\" latency "
                                << "may not be accurate because some layers may have run in parallel!" << std::endl;
        }
    }
}

namespace details
{
template <typename ContextType>
void dump(std::unique_ptr<ContextType> const& context, std::unique_ptr<Bindings> const& binding,
    ReportingOptions const& reporting, int32_t batch)
{
    if (!context)
    {
        sample::gLogError << "Empty context! Skip printing outputs." << std::endl;
        return;
    }
    if (reporting.output)
    {
        dumpOutputs(*context, *binding, sample::gLogInfo);
    }
    if (reporting.dumpRawBindings)
    {
        dumpRawBindingsToFiles(*context, *binding, sample::gLogInfo);
    }
    if (!reporting.exportOutput.empty())
    {
        exportJSONOutput(*context, *binding, reporting.exportOutput, batch);
    }
}
} // namespace details

void printOutput(ReportingOptions const& reporting, InferenceEnvironment const& iEnv, int32_t batch)
{
    auto const& binding = iEnv.bindings.at(0);
    if (!binding)
    {
        sample::gLogError << "Empty bindings! Skip printing outputs." << std::endl;
        return;
    }

    if (iEnv.safe)
    {
        auto const& context = iEnv.safeContexts.at(0);
        details::dump(context, binding, reporting, batch);
    }
    else
    {
        auto const& context = iEnv.contexts.at(0);
        details::dump(context, binding, reporting, batch);
    }
}

} // namespace sample
