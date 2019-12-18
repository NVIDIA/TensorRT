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

#ifndef TRT_SAMPLE_REPORTING_H
#define TRT_SAMPLE_REPORTING_H

#include <iostream>

#include "NvInfer.h"

#include "sampleOptions.h"
#include "sampleUtils.h"

namespace sample
{

//!
//! \struct InferenceTime
//! \brief Measurement times in milliseconds
//!
struct InferenceTime
{
    InferenceTime(float i, float c, float o, float e): in(i), compute(c), out(o), e2e(e) {}

    InferenceTime() = default;
    InferenceTime(const InferenceTime&) = default;
    InferenceTime(InferenceTime&&) = default;
    InferenceTime& operator=(const InferenceTime&) = default;
    InferenceTime& operator=(InferenceTime&&) = default;
    ~InferenceTime() = default;

    float in{0};      // Host to Device
    float compute{0}; // Compute
    float out{0};     // Device to Host
    float e2e{0};     // end to end

    // ideal latency
    float latency() const
    {
        return in + compute + out;
    }
};

//!
//! \struct InferenceTrace
//! \brief Measurement points in milliseconds
//!
struct InferenceTrace
{
    InferenceTrace(int s, float is, float ie, float cs, float ce, float os, float oe):
        stream(s), inStart(is), inEnd(ie), computeStart(cs), computeEnd(ce), outStart(os), outEnd(oe) {}

    InferenceTrace() = default;
    InferenceTrace(const InferenceTrace&) = default;
    InferenceTrace(InferenceTrace&&) = default;
    InferenceTrace& operator=(const InferenceTrace&) = default;
    InferenceTrace& operator=(InferenceTrace&&) = default;
    ~InferenceTrace() = default;

    int stream{0};
    float inStart{0};
    float inEnd{0};
    float computeStart{0};
    float computeEnd{0};
    float outStart{0};
    float outEnd{0};
};

inline InferenceTime operator+(const InferenceTime& a, const InferenceTime& b)
{
    return InferenceTime(a.in + b.in, a.compute + b.compute, a.out + b.out, a.e2e + b.e2e);
}

inline InferenceTime operator+=(InferenceTime& a, const InferenceTime& b)
{
    return a = a+b;
}

//!
//! \brief Print benchmarking time and number of traces collected
//!
void printProlog(int warmups, int timings, float warmupMs, float walltime, std::ostream& os);

//!
//! \brief Print a timing trace
//!
void printTiming(const std::vector<InferenceTime>& timings, int runsPerAvg, std::ostream& os);

//!
//! \brief Print the performance summary of a trace
//!
void printEpilog(std::vector<InferenceTime> timings, float percentile, int queries, std::ostream& os);

//!
//! \brief Print and summarize a timing trace
//!
void printPerformanceReport(const std::vector<InferenceTrace>& trace, const ReportingOptions& reporting, float warmupMs, int queries, std::ostream& os);

//!
//! \brief Export a timing trace to JSON file
//!
void exportJSONTrace(const std::vector<InferenceTrace>& trace, const std::string& fileName);

//!
//! \brief Print input tensors to stream
//!
void dumpInputs(const nvinfer1::IExecutionContext& context, const Bindings& bindings, std::ostream& os);

//!
//! \brief Print output tensors to stream
//!
void dumpOutputs(const nvinfer1::IExecutionContext& context, const Bindings& bindings, std::ostream& os);

//!
//! \brief Export output tensors to JSON file
//!
void exportJSONOutput(const nvinfer1::IExecutionContext& context, const Bindings& bindings, const std::string& fileName);

//!
//! \struct LayerProfile
//! \brief Layer profile information
//!
struct LayerProfile
{
    std::string name;
    float timeMs{0};
};

//!
//! \class Profiler
//! \brief Collect per-layer profile information, assuming times are reported in the same order
//!
class Profiler : public nvinfer1::IProfiler
{

public:

    void reportLayerTime(const char* layerName, float timeMs) override;

    void print(std::ostream& os) const;

    //!
    //! \brief Export a profile to JSON file
    //!
    void exportJSONProfile(const std::string& fileName) const;

private:

    float getTotalTime() const
    {
        const auto plusLayerTime = [](float accumulator, const LayerProfile& lp)
        {
            return accumulator + lp.timeMs;
        };
        return std::accumulate(mLayers.begin(), mLayers.end(), 0.0, plusLayerTime);
    }

    std::vector<LayerProfile> mLayers;
    std::vector<LayerProfile>::iterator mIterator{mLayers.begin()};
    int mUpdatesCount{0};
};

} // namespace sample

#endif // TRT_SAMPLE_REPORTING_H
