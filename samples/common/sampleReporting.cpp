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
#include <utility>

#include "sampleOptions.h"
#include "sampleInference.h"
#include "sampleReporting.h"

using namespace nvinfer1;

namespace sample
{

namespace
{

template <typename T>
float percentile(float percentage, const std::vector<InferenceTime>& times, const T& toFloat)
{
    int all = static_cast<int>(times.size());
    int exclude = static_cast<int>((1 - percentage / 100) * all);
    if (0 <= exclude && exclude <= all)
    {
        return toFloat(times[std::max(all - 1 - exclude, 0)]);
    }
    return std::numeric_limits<float>::infinity();
}

template <typename T>
float median(const std::vector<InferenceTime>& times, const T& val)
{
    if (!times.size())
    {
        return std::numeric_limits<float>::infinity();
    }

    int m = times.size()/2;
    if (times.size() % 2)
    {
        return val(times[m]);
    }

    return (val(times[m-1]) + val(times[m])) / 2;
}

} // namespace

void printTimes(std::vector<InferenceTime>& times, const ReportingOptions& reporting, float queries, std::ostream& os)
{
    auto toGpuTime = [](const InferenceTime& t) { return t.gpuTime; };
    auto toLatency = [](const InferenceTime& t) { return t.latency; };

    auto cmpGpuTime = [](const InferenceTime& a, const InferenceTime& b) { return a.gpuTime < b.gpuTime; };
    auto cmpLatency = [](const InferenceTime& a, const InferenceTime& b) { return a.latency < b.latency; };

    int avgs{0};
    InferenceTime sum = times.back();
    times.pop_back();
    InferenceTime average;

    os << "Timing trace of "  << times.size() << " iterations over " << sum.latency << " s" << std::endl;
    os << "Averages of " << reporting.avgs << " iterations:" << std::endl;
    for (const auto& t : times)
    {
        average.gpuTime += t.gpuTime;
        average.latency += t.latency;

        if (++avgs == reporting.avgs)
        {
// clang off
            os << "GPU compute time average over " << reporting.avgs << " runs is " << average.gpuTime/reporting.avgs << " ms"
                  " (latency average " << average.latency/reporting.avgs << " ms)" << std::endl;
// clang on
            avgs = 0;
            average.gpuTime = 0;
            average.latency = 0;
        }
    }
    os << std::endl;

    std::sort(times.begin(), times.end(), cmpLatency);
    float latencyMin = times.front().latency;
    float latencyMax = times.back().latency;
    float latencyMedian = median(times, toLatency);
    float latencyPercentile = percentile(reporting.percentile, times, toLatency);
    float latencyThroughput = queries * times.size() * 1000 / sum.latency;

    std::sort(times.begin(), times.end(), cmpGpuTime);
    float gpuMin = times.front().gpuTime;
    float gpuMax = times.back().gpuTime;
    float gpuMedian = median(times, toGpuTime);
    float gpuPercentile = percentile(reporting.percentile, times, toGpuTime);
    float gpuThroughput = queries * times.size() * 1000 / sum.gpuTime;

// clang off
    os << "Host latency"                                         << std::endl <<
          "min: "        << latencyMin                << " ms"   << std::endl <<
          "max: "        << latencyMax                << " ms"   << std::endl <<
          "mean: "       << sum.latency/times.size()  << " ms"   << std::endl <<
          "median: "     << latencyMedian             << " ms"   << std::endl <<
          "percentile: " << latencyPercentile         << " ms (" << reporting.percentile << "%)" << std::endl <<
          "throughput: " << latencyThroughput         << " qps"  << std::endl <<
          "walltime: "   << sum.latency/1000          << " s"    << std::endl << std::endl <<
          "GPU Compute"                                          << std::endl <<
          "min: "        << gpuMin                    << " ms"   << std::endl <<
          "max: "        << gpuMax                    << " ms"   << std::endl <<
          "mean: "       << sum.gpuTime/times.size()  << " ms"   << std::endl <<
          "median: "     << gpuMedian                 << " ms"   << std::endl <<
          "percentile: " << gpuPercentile             << " ms (" << reporting.percentile << "%)" << std::endl <<
          "throughput: " << gpuThroughput             << " qps"  << std::endl <<
          "walltime: "   << sum.gpuTime/1000          << " s"    << std::endl << std::endl;
// clang on
}

void dumpOutputs(const nvinfer1::ICudaEngine& engine, const std::vector<Bindings>& bindings, std::ostream& os)
{
    os << std::endl << "Output tensors" << std::endl;
    for (int b = 0; b < engine.getNbBindings(); ++b)
    {
        if (!engine.bindingIsInput(b))
        {
            os << engine.getBindingName(b) << std::endl;
        }
    }
}

} // namespace sample
