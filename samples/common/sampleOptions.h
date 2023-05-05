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
constexpr int32_t defaultMaxAuxStreams{-1};
constexpr int32_t defaultBuilderOptimizationLevel{-1};

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

//!
//! \enum RuntimeMode
//!
//! \brief Used to dictate which TensorRT runtime library to dynamically load.
//!
enum class RuntimeMode
{
    //! Maps to libnvinfer.so or nvinfer.dll
    kFULL,

    //! Maps to libnvinfer_dispatch.so or nvinfer_dispatch.dll
    kDISPATCH,

    //! Maps to libnvinfer_lean.so or nvinfer_lean.dll
    kLEAN,
};

inline std::ostream& operator<<(std::ostream& os, RuntimeMode const mode)
{
    switch (mode)
    {
    case RuntimeMode::kFULL:
    {
        os << "full";
        break;
    }
    case RuntimeMode::kDISPATCH:
    {
        os << "dispatch";
        break;
    }
    case RuntimeMode::kLEAN:
    {
        os << "lean";
        break;
    }
    }

    return os;
}

using Arguments = std::unordered_multimap<std::string, std::string>;

using IOFormat = std::pair<nvinfer1::DataType, nvinfer1::TensorFormats>;

using ShapeRange = std::array<std::vector<int32_t>, nvinfer1::EnumMax<nvinfer1::OptProfileSelector>()>;

using LayerPrecisions = std::unordered_map<std::string, nvinfer1::DataType>;
using LayerOutputTypes = std::unordered_map<std::string, std::vector<nvinfer1::DataType>>;
using LayerDeviceTypes = std::unordered_map<std::string, nvinfer1::DeviceType>;

class Options
{
public:
    virtual ~Options() = default;
    virtual void parse(Arguments& arguments) = 0;
};

class BaseModelOptions : public Options
{
public:
    ModelFormat format{ModelFormat::kANY};
    std::string model;

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

class UffInput : public Options
{
public:
    std::vector<std::pair<std::string, nvinfer1::Dims>> inputs;
    bool NHWC{false};

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

class ModelOptions : public Options
{
public:
    BaseModelOptions baseModel;
    std::string prototxt;
    std::vector<std::string> outputs;
    UffInput uffInputs;

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

constexpr nvinfer1::TempfileControlFlags getTempfileControlDefaults()
{
    using F = nvinfer1::TempfileControlFlag;
    return (1U << static_cast<uint32_t>(F::kALLOW_TEMPORARY_FILES))
        | (1U << static_cast<uint32_t>(F::kALLOW_IN_MEMORY_FILES));
}

class BuildOptions : public Options
{
public:
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
    bool fp8{false};
    bool directIO{false};
    PrecisionConstraints precisionConstraints{PrecisionConstraints::kNONE};
    LayerPrecisions layerPrecisions;
    LayerOutputTypes layerOutputTypes;
    LayerDeviceTypes layerDeviceTypes;
    bool safe{false};
    bool buildDLAStandalone{false};
    bool allowGPUFallback{false};
    bool consistency{false};
    bool restricted{false};
    bool skipInference{false};
    bool save{false};
    bool load{false};
    bool refittable{false};
    bool heuristic{false};
    bool versionCompatible{false};
    bool excludeLeanRuntime{false};
    int32_t builderOptimizationLevel{defaultBuilderOptimizationLevel};
    SparsityFlag sparsity{SparsityFlag::kDISABLE};
    nvinfer1::ProfilingVerbosity profilingVerbosity{nvinfer1::ProfilingVerbosity::kLAYER_NAMES_ONLY};
    std::string engine;
    std::string calibration;
    using ShapeProfile = std::unordered_map<std::string, ShapeRange>;
    ShapeProfile shapes;
    ShapeProfile shapesCalib;
    std::vector<IOFormat> inputFormats;
    std::vector<IOFormat> outputFormats;
    nvinfer1::TacticSources enabledTactics{0};
    nvinfer1::TacticSources disabledTactics{0};
    TimingCacheMode timingCacheMode{TimingCacheMode::kLOCAL};
    std::string timingCacheFile{};
    // C++11 does not automatically generate hash function for enum class.
    // Use int32_t to support C++11 compilers.
    std::unordered_map<int32_t, bool> previewFeatures;
    nvinfer1::HardwareCompatibilityLevel hardwareCompatibilityLevel{nvinfer1::HardwareCompatibilityLevel::kNONE};
    std::string tempdir{};
    nvinfer1::TempfileControlFlags tempfileControls{getTempfileControlDefaults()};
    RuntimeMode useRuntime{RuntimeMode::kFULL};
    std::string leanDLLPath{};
    int32_t maxAuxStreams{defaultMaxAuxStreams};

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

class SystemOptions : public Options
{
public:
    int32_t device{defaultDevice};
    int32_t DLACore{-1};
    bool ignoreParsedPluginLibs{false};
    std::vector<std::string> plugins;
    std::vector<std::string> setPluginsToSerialize;
    std::vector<std::string> dynamicPlugins;

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

class InferenceOptions : public Options
{
public:
    int32_t batch{batchNotProvided};
    int32_t iterations{defaultIterations};
    int32_t infStreams{defaultStreams};
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
    using ShapeProfile = std::unordered_map<std::string, std::vector<int32_t>>;
    ShapeProfile shapes;
    nvinfer1::ProfilingVerbosity nvtxVerbosity{nvinfer1::ProfilingVerbosity::kLAYER_NAMES_ONLY};

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

class ReportingOptions : public Options
{
public:
    bool verbose{false};
    int32_t avgs{defaultAvgRuns};
    std::vector<float> percentiles{defaultPercentiles.begin(), defaultPercentiles.end()};
    bool refit{false};
    bool output{false};
    bool dumpRawBindings{false};
    bool profile{false};
    bool layerInfo{false};
    std::string exportTimes;
    std::string exportOutput;
    std::string exportProfile;
    std::string exportLayerInfo;

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

class SafeBuilderOptions : public Options
{
public:
    std::string serialized{};
    std::string onnxModelFile{};
    bool help{false};
    bool verbose{false};
    std::vector<IOFormat> inputFormats;
    std::vector<IOFormat> outputFormats;
    bool int8{false};
    bool fp8{false};
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

class AllOptions : public Options
{
public:
    ModelOptions model;
    BuildOptions build;
    SystemOptions system;
    InferenceOptions inference;
    ReportingOptions reporting;
    bool helps{false};

    void parse(Arguments& arguments) override;

    static void help(std::ostream& out);
};

class TaskInferenceOptions : public Options
{
public:
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
