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

#ifndef TRT_SAMPLE_OPTIONS_H
#define TRT_SAMPLE_OPTIONS_H

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "NvInfer.h"

namespace sample
{

// Build default params
constexpr int32_t maxBatchNotProvided{0};
constexpr int32_t defaultMinTiming{1};
constexpr int32_t defaultAvgTiming{8};

// System default params
constexpr int32_t defaultDevice{0};

// Inference default params
constexpr int32_t defaultBatch{1};
constexpr int32_t batchNotProvided{0};
constexpr int32_t defaultStreams{1};
constexpr int32_t defaultIterations{10};
constexpr float defaultWarmUp{200.F};
constexpr float defaultDuration{3.F};
constexpr float defaultSleep{};
constexpr float defaultIdle{};
constexpr float defaultPersistentCacheRatio{0};

// Reporting default params
constexpr int32_t defaultAvgRuns{10};
constexpr std::array<float, 3> defaultPercentiles{90, 95, 99};

enum class PrecisionConstraints
{
    kNONE,
    kOBEY,
    kPREFER
};

enum class ModelFormat
{
    kANY,
    kCAFFE,
    kONNX,
    kUFF
};

enum class SparsityFlag
{
    kDISABLE,
    kENABLE,
    kFORCE
};

enum class TimingCacheMode
{
    kDISABLE,
    kLOCAL,
    kGLOBAL
};

using Arguments = std::unordered_multimap<std::string, std::string>;

using IOFormat = std::pair<nvinfer1::DataType, nvinfer1::TensorFormats>;

using ShapeRange = std::array<std::vector<int32_t>, nvinfer1::EnumMax<nvinfer1::OptProfileSelector>()>;

using LayerPrecisions = std::unordered_map<std::string, nvinfer1::DataType>;
using LayerOutputTypes = std::unordered_map<std::string, std::vector<nvinfer1::DataType>>;

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
    int32_t maxBatch{maxBatchNotProvided};
    double workspace{-1.0};
    double dlaSRAM{-1.0};
    double dlaLocalDRAM{-1.0};
    double dlaGlobalDRAM{-1.0};
    int32_t minTiming{defaultMinTiming};
    int32_t avgTiming{defaultAvgTiming};
    bool tf32{true};
    bool fp16{false};
    bool int8{false};
    bool directIO{false};
    PrecisionConstraints precisionConstraints{PrecisionConstraints::kNONE};
    LayerPrecisions layerPrecisions;
    LayerOutputTypes layerOutputTypes;
    bool safe{false};
    bool consistency{false};
    bool restricted{false};
    bool buildOnly{false};
    bool save{false};
    bool load{false};
    bool refittable{false};
    bool heuristic{false};
    SparsityFlag sparsity{SparsityFlag::kDISABLE};
    nvinfer1::ProfilingVerbosity profilingVerbosity{nvinfer1::ProfilingVerbosity::kLAYER_NAMES_ONLY};
    std::string engine;
    std::string calibration;
    std::unordered_map<std::string, ShapeRange> shapes;
    std::unordered_map<std::string, ShapeRange> shapesCalib;
    std::vector<IOFormat> inputFormats;
    std::vector<IOFormat> outputFormats;
    nvinfer1::TacticSources enabledTactics{0};
    nvinfer1::TacticSources disabledTactics{0};
    TimingCacheMode timingCacheMode{TimingCacheMode::kLOCAL};
    std::string timingCacheFile{};
    // C++11 does not automatically generate hash function for enum class.
    // Use int32_t to support C++11 compilers.
    std::unordered_map<int32_t, bool> previewFeatures;
    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

struct SystemOptions : public Options
{
    int32_t device{defaultDevice};
    int32_t DLACore{-1};
    bool fallback{false};
    std::vector<std::string> plugins;

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

struct InferenceOptions : public Options
{
    int32_t batch{batchNotProvided};
    int32_t iterations{defaultIterations};
    int32_t streams{defaultStreams};
    float warmup{defaultWarmUp};
    float duration{defaultDuration};
    float sleep{defaultSleep};
    float idle{defaultIdle};
    float persistentCacheRatio{defaultPersistentCacheRatio};
    bool overlap{true};
    bool skipTransfers{false};
    bool useManaged{false};
    bool spin{false};
    bool threads{false};
    bool graph{false};
    bool rerun{false};
    bool timeDeserialize{false};
    bool timeRefit{false};
    std::unordered_map<std::string, std::string> inputs;
    std::unordered_map<std::string, std::vector<int32_t>> shapes;
    nvinfer1::ProfilingVerbosity nvtxVerbosity{nvinfer1::ProfilingVerbosity::kLAYER_NAMES_ONLY};

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

struct ReportingOptions : public Options
{
    bool verbose{false};
    int32_t avgs{defaultAvgRuns};
    std::vector<float> percentiles{defaultPercentiles.begin(), defaultPercentiles.end()};
    bool refit{false};
    bool output{false};
    bool profile{false};
    bool layerInfo{false};
    std::string exportTimes;
    std::string exportOutput;
    std::string exportProfile;
    std::string exportLayerInfo;

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

struct SafeBuilderOptions : public Options
{
    std::string serialized{};
    std::string onnxModelFile{};
    bool help{false};
    bool verbose{false};
    std::vector<IOFormat> inputFormats;
    std::vector<IOFormat> outputFormats;
    bool int8{false};
    std::string calibFile{};
    std::vector<std::string> plugins;
    bool consistency{false};
    bool standard{false};
    TimingCacheMode timingCacheMode{TimingCacheMode::kLOCAL};
    std::string timingCacheFile{};
    SparsityFlag sparsity{SparsityFlag::kDISABLE};
    int32_t minTiming{defaultMinTiming};
    int32_t avgTiming{defaultAvgTiming};

    void parse(Arguments& arguments) override;

    static void printHelp(std::ostream& out);
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

struct TaskInferenceOptions : public Options
{
    std::string engine;
    int32_t device{defaultDevice};
    int32_t DLACore{-1};
    int32_t batch{batchNotProvided};
    bool graph{false};
    float persistentCacheRatio{defaultPersistentCacheRatio};
    void parse(Arguments& arguments) override;
    static void help(std::ostream& out);
};

Arguments argsToArgumentsMap(int32_t argc, char* argv[]);

bool parseHelp(Arguments& arguments);

void helpHelp(std::ostream& out);

// Functions to print options

std::ostream& operator<<(std::ostream& os, const BaseModelOptions& options);

std::ostream& operator<<(std::ostream& os, const UffInput& input);

std::ostream& operator<<(std::ostream& os, const IOFormat& format);

std::ostream& operator<<(std::ostream& os, const ShapeRange& dims);

std::ostream& operator<<(std::ostream& os, const ModelOptions& options);

std::ostream& operator<<(std::ostream& os, const BuildOptions& options);

std::ostream& operator<<(std::ostream& os, const SystemOptions& options);

std::ostream& operator<<(std::ostream& os, const InferenceOptions& options);

std::ostream& operator<<(std::ostream& os, const ReportingOptions& options);

std::ostream& operator<<(std::ostream& os, const AllOptions& options);

std::ostream& operator<<(std::ostream& os, const SafeBuilderOptions& options);

inline std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& dims)
{
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        os << (i ? "x" : "") << dims.d[i];
    }
    return os;
}
inline std::ostream& operator<<(std::ostream& os, const nvinfer1::WeightsRole role)
{
    switch (role)
    {
    case nvinfer1::WeightsRole::kKERNEL:
    {
        os << "Kernel";
        break;
    }
    case nvinfer1::WeightsRole::kBIAS:
    {
        os << "Bias";
        break;
    }
    case nvinfer1::WeightsRole::kSHIFT:
    {
        os << "Shift";
        break;
    }
    case nvinfer1::WeightsRole::kSCALE:
    {
        os << "Scale";
        break;
    }
    case nvinfer1::WeightsRole::kCONSTANT:
    {
        os << "Constant";
        break;
    }
    case nvinfer1::WeightsRole::kANY:
    {
        os << "Any";
        break;
    }
    }

    return os;
}

inline std::ostream& operator<<(std::ostream& os, const std::vector<int32_t>& vec)
{
    for (int32_t i = 0, e = static_cast<int32_t>(vec.size()); i < e; ++i)
    {
        os << (i ? "x" : "") << vec[i];
    }
    return os;
}

} // namespace sample

#endif // TRT_SAMPLES_OPTIONS_H
