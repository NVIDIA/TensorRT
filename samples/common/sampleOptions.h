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

#ifndef TRT_SAMPLE_OPTIONS_H
#define TRT_SAMPLE_OPTIONS_H

#include <utility>
#include <stdexcept>
#include <vector>
#include <array>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <iostream>

#include "NvInfer.h"

namespace sample
{

// Build default params
constexpr int defaultMaxBatch{1};
constexpr int defaultWorkspace{16};
constexpr int defaultMinTiming{1};
constexpr int defaultAvgTiming{8};

// System default params
constexpr int defaultDevice{0};

// Inference default params
constexpr int defaultBatch{1};
constexpr int defaultStreams{1};
constexpr int defaultIterations{10};
constexpr int defaultWarmUp{200};
constexpr int defaultDuration{3};
constexpr int defaultSleep{0};

// Reporting default params
constexpr int defaultAvgRuns{10};
constexpr float defaultPercentile{99};

enum class ModelFormat
{
    kANY,
    kCAFFE,
    kONNX,
    kUFF
};

using Arguments = std::unordered_multimap<std::string, std::string>;

using IOFormat = std::pair<nvinfer1::DataType, nvinfer1::TensorFormats>;

using ShapeRange = std::array<nvinfer1::Dims, nvinfer1::EnumMax<nvinfer1::OptProfileSelector>()>;

struct Options
{
    virtual void parse(Arguments& arguments) = 0;
};

struct BaseModelOptions : public Options
{
    ModelFormat format{ModelFormat::kANY};
    std::string model;

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

struct UffInput : public Options
{
    std::vector<std::pair<std::string, nvinfer1::Dims>> inputs;
    bool NHWC{false};

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

struct ModelOptions : public Options
{
    BaseModelOptions baseModel;
    std::string prototxt;
    std::vector<std::string> outputs;
    UffInput uffInputs;

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

struct BuildOptions : public Options
{
    int maxBatch{defaultMaxBatch}; // Parsing sets maxBatch to 0 if explicitBatch is true
    int workspace{defaultWorkspace};
    int minTiming{defaultMinTiming};
    int avgTiming{defaultAvgTiming};
    bool fp16{false};
    bool int8{false};
    bool safe{false};
    bool save{false};
    bool load{false};
    std::string engine;
    std::string calibration;
    std::unordered_map<std::string, ShapeRange> shapes;
    std::vector<IOFormat> inputFormats;
    std::vector<IOFormat> outputFormats;

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

struct SystemOptions : public Options
{
    int device{defaultDevice};
    int DLACore{-1};
    bool fallback{false};
    std::vector<std::string> plugins;

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

struct InferenceOptions : public Options
{
    int batch{defaultBatch}; // Parsing sets batch to 0 is shapes is not empty
    int iterations{defaultIterations};
    int warmup{defaultWarmUp};
    int duration{defaultDuration};
    int sleep{defaultSleep};
    int streams{defaultStreams};
    bool spin{false};
    bool threads{false};
    bool graph{false};
    bool skip{false};
    std::string inputs;
    std::unordered_map<std::string, nvinfer1::Dims> shapes;

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

struct ReportingOptions : public Options
{
    bool verbose{false};
    int avgs{defaultAvgRuns};
    float percentile{defaultPercentile};
    bool output{false};
    bool profile{false};
    std::string exportTimes;
    std::string exportOutput;
    std::string exportProfile;

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

struct AllOptions : public Options
{
    ModelOptions model;
    BuildOptions build;
    SystemOptions system;
    InferenceOptions inference;
    ReportingOptions reporting;
    bool helps{false};

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

Arguments argsToArgumentsMap(int argc, char* argv[]);

bool parseHelp(Arguments& arguments);

void helpHelp(std::ostream& out);

// Functions to print options

std::ostream& operator<<(std::ostream& os, const BaseModelOptions& options);

std::ostream& operator<<(std::ostream& os, const UffInput& input);

std::ostream& operator<<(std::ostream& os, const IOFormat& format);

std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& dims);

std::ostream& operator<<(std::ostream& os, const ShapeRange& dims);

std::ostream& operator<<(std::ostream& os, const ModelOptions& options);

std::ostream& operator<<(std::ostream& os, const BuildOptions& options);

std::ostream& operator<<(std::ostream& os, const SystemOptions& options);

std::ostream& operator<<(std::ostream& os, const InferenceOptions& options);

std::ostream& operator<<(std::ostream& os, const ReportingOptions& options);

std::ostream& operator<<(std::ostream& os, const AllOptions& options);


} // namespace sample

#endif // TRT_SAMPLES_OPTIONS_H
