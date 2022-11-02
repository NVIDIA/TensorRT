/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_SAMPLE_REPORTING_H
#define TRT_SAMPLE_REPORTING_H

#include <functional>
#include <iostream>
#include <numeric>

#include "NvInfer.h"

#include "sampleDevice.h"
#include "sampleOptions.h"
#include "sampleUtils.h"

namespace sample
{

class Bindings;

//!
//! \struct InferenceTime
//! \brief Measurement times in milliseconds
//!
struct InferenceTime
{
    InferenceTime(float q, float i, float c, float o)
        : enq(q)
        , h2d(i)
        , compute(c)
        , d2h(o)
    {
    }

    InferenceTime() = default;
    InferenceTime(InferenceTime const&) = default;
    InferenceTime(InferenceTime&&) = default;
    InferenceTime& operator=(InferenceTime const&) = default;
    InferenceTime& operator=(InferenceTime&&) = default;
    ~InferenceTime() = default;

    float enq{0};     // Enqueue
    float h2d{0};     // Host to Device
    float compute{0}; // Compute
    float d2h{0};     // Device to Host

    // ideal latency
    float latency() const
    {
        return h2d + compute + d2h;
    }
};

//!
//! \struct InferenceTrace
//! \brief Measurement points in milliseconds
//!
struct InferenceTrace
{
    InferenceTrace(int32_t s, float es, float ee, float is, float ie, float cs, float ce, float os, float oe)
        : stream(s)
        , enqStart(es)
        , enqEnd(ee)
        , h2dStart(is)
        , h2dEnd(ie)
        , computeStart(cs)
        , computeEnd(ce)
        , d2hStart(os)
        , d2hEnd(oe)
    {
    }

    InferenceTrace() = default;
    InferenceTrace(InferenceTrace const&) = default;
    InferenceTrace(InferenceTrace&&) = default;
    InferenceTrace& operator=(InferenceTrace const&) = default;
    InferenceTrace& operator=(InferenceTrace&&) = default;
    ~InferenceTrace() = default;

    int32_t stream{0};
    float enqStart{0};
    float enqEnd{0};
    float h2dStart{0};
    float h2dEnd{0};
    float computeStart{0};
    float computeEnd{0};
    float d2hStart{0};
    float d2hEnd{0};
};

inline InferenceTime operator+(InferenceTime const& a, InferenceTime const& b)
{
    return InferenceTime(a.enq + b.enq, a.h2d + b.h2d, a.compute + b.compute, a.d2h + b.d2h);
}

inline InferenceTime operator+=(InferenceTime& a, InferenceTime const& b)
{
    return a = a + b;
}

//!
//! \struct PerformanceResult
//! \brief Performance result of a performance metric
//!
struct PerformanceResult
{
    float min{0.F};
    float max{0.F};
    float mean{0.F};
    float median{0.F};
    std::vector<float> percentiles;
    float coeffVar{0.F}; // coefficient of variation
};

//!
//! \brief Print benchmarking time and number of traces collected
//!
void printProlog(int32_t warmups, int32_t timings, float warmupMs, float walltime, std::ostream& os);

//!
//! \brief Print a timing trace
//!
void printTiming(std::vector<InferenceTime> const& timings, int32_t runsPerAvg, std::ostream& os);

//!
//! \brief Print the performance summary of a trace
//!
void printEpilog(std::vector<InferenceTime> const& timings, std::vector<float> const& percentiles, int32_t batchSize,
    std::ostream& osInfo, std::ostream& osWarning, std::ostream& osVerbose);

//!
//! \brief Get the result of a specific performance metric from a trace
//!
PerformanceResult getPerformanceResult(std::vector<InferenceTime> const& timings,
    std::function<float(InferenceTime const&)> metricGetter, std::vector<float> const& percentiles);

//!
//! \brief Print the explanations of the performance metrics printed in printEpilog() function.
//!
void printMetricExplanations(std::ostream& os);

//!
//! \brief Print and summarize a timing trace
//!
void printPerformanceReport(std::vector<InferenceTrace> const& trace, ReportingOptions const& reporting, float warmupMs,
    int32_t batchSize, std::ostream& osInfo, std::ostream& osWarning, std::ostream& osVerbose);

//!
//! \brief Export a timing trace to JSON file
//!
void exportJSONTrace(
    std::vector<InferenceTrace> const& InferenceTime, std::string const& fileName, int32_t const nbWarmups);

//!
//! \brief Print input tensors to stream
//!
void dumpInputs(nvinfer1::IExecutionContext const& context, Bindings const& bindings, std::ostream& os);

//!
//! \brief Print output tensors to stream
//!
template <typename ContextType>
void dumpOutputs(ContextType const& context, Bindings const& bindings, std::ostream& os);

//!
//! \brief Export output tensors to JSON file
//!
template <typename ContextType>
void exportJSONOutput(
    ContextType const& context, Bindings const& bindings, std::string const& fileName, int32_t batch);


//!
//! \struct LayerProfile
//! \brief Layer profile information
//!
struct LayerProfile
{
    std::string name;
    std::vector<float> timeMs;
};

//!
//! \class Profiler
//! \brief Collect per-layer profile information, assuming times are reported in the same order
//!
class Profiler : public nvinfer1::IProfiler
{

public:
    void reportLayerTime(char const* layerName, float timeMs) noexcept override;

    void print(std::ostream& os) const noexcept;

    //!
    //! \brief Export a profile to JSON file
    //!
    void exportJSONProfile(std::string const& fileName) const noexcept;

private:
    float getTotalTime() const noexcept
    {
        auto const plusLayerTime = [](float accumulator, LayerProfile const& lp) {
            return accumulator + std::accumulate(lp.timeMs.begin(), lp.timeMs.end(), 0.F, std::plus<float>());
        };
        return std::accumulate(mLayers.begin(), mLayers.end(), 0.0F, plusLayerTime);
    }

    float getMedianTime() const noexcept
    {
        if (mLayers.empty())
        {
            return 0.F;
        }
        std::vector<float> totalTime;
        for (size_t run = 0; run < mLayers[0].timeMs.size(); ++run)
        {
            auto const layerTime
                = [&run](float accumulator, LayerProfile const& lp) { return accumulator + lp.timeMs[run]; };
            auto t = std::accumulate(mLayers.begin(), mLayers.end(), 0.F, layerTime);
            totalTime.push_back(t);
        }
        return median(totalTime);
    }

    float getMedianTime(LayerProfile const& p) const noexcept
    {
        return median(p.timeMs);
    }

    static float median(std::vector<float> vals)
    {
        if (vals.empty())
        {
            return 0.F;
        }
        std::sort(vals.begin(), vals.end());
        if (vals.size() % 2U == 1U)
        {
            return vals[vals.size() / 2U];
        }
        return (vals[vals.size() / 2U - 1U] + vals[vals.size() / 2U]) * 0.5F;
    }

    //! return the total runtime of given layer profile
    float getTotalTime(LayerProfile const& p) const noexcept
    {
        auto const& vals = p.timeMs;
        return std::accumulate(vals.begin(), vals.end(), 0.F, std::plus<float>());
    }

    float getAvgTime(LayerProfile const& p) const noexcept
    {
        return getTotalTime(p) / p.timeMs.size();
    }

    std::vector<LayerProfile> mLayers;
    std::vector<LayerProfile>::iterator mIterator{mLayers.begin()};
    int32_t mUpdatesCount{0};
};

} // namespace sample

#endif // TRT_SAMPLE_REPORTING_H
