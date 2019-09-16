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

namespace sample
{

struct InferenceTime
{
    InferenceTime(float l, float g): latency(l), gpuTime(g) {}

    InferenceTime() = default;
    InferenceTime(const InferenceTime&) = default;
    InferenceTime(InferenceTime&&) = default;
    InferenceTime& operator=(const InferenceTime&) = default;
    InferenceTime& operator=(InferenceTime&&) = default;
    ~InferenceTime() = default;

    float latency{};
    float gpuTime{};
};

inline InferenceTime operator+(const InferenceTime& a, const InferenceTime& b)
{
    InferenceTime sum;
    sum.gpuTime = a.gpuTime + b.gpuTime;
    sum.latency = a.latency + b.latency;
    return sum;
}

//!
//! \brief Print and summarize a timing trace
//!
void printTimes(std::vector<InferenceTime>& times, const ReportingOptions& reporting, float queries, std::ostream& os);

//!
//! \brief Print output tensors to stream
//!
void dumpOutputs(const nvinfer1::ICudaEngine& engine, const std::vector<Bindings>& bindings, std::ostream& os);

} // namespace sample

#endif // TRT_SAMPLE_REPORTING_H
