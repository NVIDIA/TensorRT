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

#include <cstring>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include <iostream>

#include "NvInfer.h"

#include "sampleOptions.h"

namespace sample
{

namespace
{

inline std::vector<std::string> splitToStringVec(const std::string& option, char separator)
{
    std::vector<std::string> options;

    for(size_t start = 0; start < option.length(); )
    {
        size_t separatorIndex = option.find(separator, start);
        if (separatorIndex == std::string::npos)
        {
            separatorIndex = option.length();
        }
        options.emplace_back(option.substr(start, separatorIndex - start));
        start = separatorIndex + 1;
    }

    return options;
}

template <typename T>
inline T stringToValue(const std::string& option)
{
    return T{option};
}

template <>
inline int stringToValue<int>(const std::string& option)
{
    return std::stoi(option);
}

template <>
inline float stringToValue<float>(const std::string& option)
{
    return std::stof(option);
}

template <>
inline bool stringToValue<bool>(const std::string& option)
{
    return true;
}

template <>
inline nvinfer1::Dims stringToValue<nvinfer1::Dims>(const std::string& option)
{
    nvinfer1::Dims dims;
    dims.nbDims = 0;
    std::vector<std::string> dimsStrings = splitToStringVec(option, 'x');
    for (const auto& d : dimsStrings)
    {
        if (d == "*")
        {
            break;
        }
        dims.d[dims.nbDims] = stringToValue<int>(d);
        ++dims.nbDims;
    }
    return dims;
}

template <>
inline nvinfer1::DataType stringToValue<nvinfer1::DataType>(const std::string& option)
{
    const std::unordered_map<std::string, nvinfer1::DataType> strToDT{{"fp32", nvinfer1::DataType::kFLOAT}, {"fp16", nvinfer1::DataType::kHALF},
                                                                      {"int8", nvinfer1::DataType::kINT8}, {"int32", nvinfer1::DataType::kINT32}};
    auto dt = strToDT.find(option);
    if (dt == strToDT.end())
    {
        throw std::invalid_argument("Invalid DataType " + option);
    }
    return dt->second;
}

template <>
inline nvinfer1::TensorFormats stringToValue<nvinfer1::TensorFormats>(const std::string& option)
{
    std::vector<std::string> optionStrings = splitToStringVec(option, '+');
    const std::unordered_map<std::string, nvinfer1::TensorFormat> strToFmt{{"chw", nvinfer1::TensorFormat::kLINEAR}, {"chw2", nvinfer1::TensorFormat::kCHW2},
                                                                           {"chw4", nvinfer1::TensorFormat::kCHW4}, {"hwc8", nvinfer1::TensorFormat::kHWC8},
                                                                           {"chw16", nvinfer1::TensorFormat::kCHW16}, {"chw32", nvinfer1::TensorFormat::kCHW32}};
    nvinfer1::TensorFormats formats{};
    for (auto f : optionStrings)
    {
        auto tf = strToFmt.find(f);
        if (tf == strToFmt.end())
        {
            throw std::invalid_argument(std::string("Invalid TensorFormat ") + f);
        }
        formats |= 1U << int(tf->second);
    }

    return formats;
}

template <>
inline IOFormat stringToValue<IOFormat>(const std::string& option)
{
    IOFormat ioFormat{};
    size_t colon = option.find(':');

    if (colon == std::string::npos)
    {
        throw std::invalid_argument(std::string("Invalid IOFormat ") + option);
    }
    ioFormat.first = stringToValue<nvinfer1::DataType>(option.substr(0, colon));
    ioFormat.second = stringToValue<nvinfer1::TensorFormats>(option.substr(colon+1));

    return ioFormat;
}

inline const char* boolToEnabled(bool enable)
{
    return enable ? "Enabled" : "Disabled";
}

template <typename T>
inline bool checkEraseOption(Arguments& arguments, const std::string& option, T& value)
{
    auto match = arguments.find(option);
    if (match != arguments.end())
    {
        value = stringToValue<T>(match->second);
        arguments.erase(match);
        return true;
    }

    return false;
}

template <typename T>
inline bool checkEraseRepeatedOption(Arguments& arguments, const std::string& option, std::vector<T>& values)
{
    auto match = arguments.equal_range(option);
    if (match.first == match.second)
    {
        return false;
    }
    auto addValue = [&values](Arguments::value_type& value) {values.emplace_back(stringToValue<T>(value.second));};
    std::for_each(match.first, match.second, addValue);
    arguments.erase(match.first, match.second);
    return true;
}

void insertShapes(std::unordered_map<std::string, ShapeRange>& shapes, const std::string& name, const nvinfer1::Dims& dims)
{
    std::pair<std::string, ShapeRange> profile;
    profile.first = name;
    profile.second[static_cast<size_t>(nvinfer1::OptProfileSelector::kMIN)] = dims;
    profile.second[static_cast<size_t>(nvinfer1::OptProfileSelector::kOPT)] = dims;
    profile.second[static_cast<size_t>(nvinfer1::OptProfileSelector::kMAX)] = dims;
    shapes.insert(profile);
}

template <typename T>
void printShapes(std::ostream& os, const char* phase, const T& shapes)
{
    if (shapes.empty())
    {
        os << "Input " << phase << " shapes: model" << std::endl;
    }
    else
    {
        for (const auto& s : shapes)
        {
            os << "Input " << phase << " shape: " << s.first << "=" << s.second << std::endl;
        }
    }
}

std::ostream& printBatch(std::ostream& os, int maxBatch)
{
    if (maxBatch)
    {
        os << maxBatch;
    }
    else
    {
        os << "explicit";
    }
    return os;
}

}

Arguments argsToArgumentsMap(int argc, char* argv[])
{
    Arguments arguments;
    for (int i = 1; i < argc; ++i)
    {
        auto valuePtr = strchr(argv[i], '=');
        if (valuePtr)
        {
            std::string value{valuePtr + 1};
            arguments.emplace(std::string(argv[i], valuePtr - argv[i]), value);
        }
        else
        {
            arguments.emplace(argv[i], "");
        }
    }
    return arguments;
}

void BaseModelOptions::parse(Arguments& arguments)
{
    if (checkEraseOption(arguments, "--onnx", model))
    {
        format = ModelFormat::kONNX;
    }
    else if (checkEraseOption(arguments, "--uff", model))
    {
        format = ModelFormat::kUFF;
    }
    else if (checkEraseOption(arguments, "--model", model))
    {
        format = ModelFormat::kCAFFE;
    }
}

void UffInput::parse(Arguments& arguments)
{
    checkEraseOption(arguments, "--uffNHWC", NHWC);
    std::vector<std::string> args;
    if (checkEraseRepeatedOption(arguments, "--uffInput", args))
    {
        for (const auto& i : args)
        {
            std::vector<std::string> values{splitToStringVec(i, ',')};
            if (values.size() == 4)
            {
                nvinfer1::Dims3 dims{std::stoi(values[1]), std::stoi(values[2]), std::stoi(values[3])};
                inputs.emplace_back(values[0], dims);
            }
            else
            {
                throw std::invalid_argument(std::string("Invalid uffInput ") + i);
            }
        }
    }
}

void ModelOptions::parse(Arguments& arguments)
{
    baseModel.parse(arguments);

    switch (baseModel.format)
    {
    case ModelFormat::kCAFFE:
    {
        checkEraseOption(arguments, "--deploy", prototxt);
        break;
    }
    case ModelFormat::kUFF:
    {
        uffInputs.parse(arguments);
        if (uffInputs.inputs.empty())
        {
            throw std::invalid_argument("Uff models require at least one input");
        }
        break;
    }
    case ModelFormat::kONNX:
        break;
    case ModelFormat::kANY:
    {
        if (checkEraseOption(arguments, "--deploy", prototxt))
        {
            baseModel.format = ModelFormat::kCAFFE;
        }
        break;
    }
    }
    if (baseModel.format == ModelFormat::kCAFFE || baseModel.format == ModelFormat::kUFF)
    {
        std::vector<std::string> outArgs;
        if (checkEraseRepeatedOption(arguments, "--output", outArgs))
        {
            for (const auto& o : outArgs)
            {
                for (auto& v : splitToStringVec(o, ','))
                {
                    outputs.emplace_back(std::move(v));
                }
            }
        }
        if (outputs.empty())
        {
            throw std::invalid_argument("Caffe and Uff models require at least one output");
        }
    }
}


void BuildOptions::parse(Arguments& arguments)
{
    auto getFormats = [&arguments](std::vector<IOFormat>& formatsVector, const char* argument)
    {
        std::string list;
        checkEraseOption(arguments, argument, list);
        std::vector<std::string> formats{splitToStringVec(list, ',')};
        for (const auto& f : formats)
        {
            formatsVector.push_back(stringToValue<IOFormat>(f));
        }
    };

    getFormats(inputFormats, "--inputIOFormats");
    getFormats(outputFormats, "--outputIOFormats");

    auto getShapes = [&arguments](std::unordered_map<std::string, ShapeRange>& shapes, const char* argument,
                         nvinfer1::OptProfileSelector selector)
    {
        std::string list;
        checkEraseOption(arguments, argument, list);
        std::vector<std::string> shapeList{splitToStringVec(list, ',')};
        for (const auto& s : shapeList)
        {
            std::vector<std::string> nameRange{splitToStringVec(s, ':')};
            if (shapes.find(nameRange[0]) == shapes.end())
            {
                auto dims = stringToValue<nvinfer1::Dims>(nameRange[1]);
                insertShapes(shapes, nameRange[0], dims);
            }
            else
            {
                shapes[nameRange[0]][static_cast<size_t>(selector)] = stringToValue<nvinfer1::Dims>(nameRange[1]);
            }
        }
    };

    bool explicitBatch{false};
    checkEraseOption(arguments, "--explicitBatch", explicitBatch);
    getShapes(shapes, "--minShapes", nvinfer1::OptProfileSelector::kMIN);
    getShapes(shapes, "--optShapes", nvinfer1::OptProfileSelector::kOPT);
    getShapes(shapes, "--maxShapes", nvinfer1::OptProfileSelector::kMAX);
    explicitBatch = explicitBatch || !shapes.empty();

    int batch{0};
    checkEraseOption(arguments, "--maxBatch", batch);
    if (explicitBatch && batch)
    {
        throw std::invalid_argument(
            "Explicit batch or dynamic shapes enabled with implicit maxBatch " + std::to_string(batch));
    }

    if (explicitBatch)
    {
        maxBatch = 0;
    }
    else
    {
        if (batch)
        {
            maxBatch = batch;
        }
    }

    checkEraseOption(arguments, "--workspace", workspace);
    checkEraseOption(arguments, "--minTiming", minTiming);
    checkEraseOption(arguments, "--avgTiming", avgTiming);
    checkEraseOption(arguments, "--fp16", fp16);
    checkEraseOption(arguments, "--int8", int8);
    checkEraseOption(arguments, "--safe", safe);
    checkEraseOption(arguments, "--calib", calibration);
    if (checkEraseOption(arguments, "--loadEngine", engine))
    {
        load = true;
    }
    if (checkEraseOption(arguments, "--saveEngine", engine))
    {
        save = true;
    }
    if (load && save)
    {
        throw std::invalid_argument("Incompatible load and save engine options selected");
    }
}

void SystemOptions::parse(Arguments& arguments)
{
    checkEraseOption(arguments, "--device", device);
    checkEraseOption(arguments, "--useDLACore", DLACore);
    checkEraseOption(arguments, "--allowGPUFallback", fallback);
    std::string pluginName;
    while (checkEraseOption(arguments, "--plugins", pluginName))
    {
        plugins.emplace_back(pluginName);
    }
}

void InferenceOptions::parse(Arguments& arguments)
{
    checkEraseOption(arguments, "--streams", streams);
    checkEraseOption(arguments, "--iterations", iterations);
    checkEraseOption(arguments, "--duration", duration);
    checkEraseOption(arguments, "--warmUp", warmup);
    checkEraseOption(arguments, "--sleepTime", sleep);
    checkEraseOption(arguments, "--useSpinWait", spin);
    checkEraseOption(arguments, "--threads", threads);
    checkEraseOption(arguments, "--loadInputs", inputs);
    checkEraseOption(arguments, "--useCudaGraph", graph);
    checkEraseOption(arguments, "--buildOnly", skip);

    std::string list;
    checkEraseOption(arguments, "--shapes", list);
    std::vector<std::string> shapeList{splitToStringVec(list, ',')};
    for (const auto& s : shapeList)
    {
        std::vector<std::string> shapeSpec{splitToStringVec(s, ':')};
        shapes.insert({shapeSpec[0], stringToValue<nvinfer1::Dims>(shapeSpec[1])});
    }

    int batchOpt{0};
    checkEraseOption(arguments, "--batch", batchOpt);
    if (!shapes.empty() && batchOpt)
    {
        throw std::invalid_argument(
            "Explicit batch or dynamic shapes enabled with implicit batch " + std::to_string(batchOpt));
    }
    if (batchOpt)
    {
        batch = batchOpt;
    }
    else
    {
        if (!shapes.empty())
        {
            batch = 0;
        }
    }
}

void ReportingOptions::parse(Arguments& arguments)
{
    checkEraseOption(arguments, "--percentile", percentile);
    checkEraseOption(arguments, "--avgRuns", avgs);
    checkEraseOption(arguments, "--verbose", verbose);
    checkEraseOption(arguments, "--dumpOutput", output);
    checkEraseOption(arguments, "--dumpProfile", profile);
    checkEraseOption(arguments, "--exportTimes", exportTimes);
    checkEraseOption(arguments, "--exportOutput", exportOutput);
    checkEraseOption(arguments, "--exportProfile", exportProfile);
    if (percentile < 0 || percentile > 100)
    {
        throw std::invalid_argument(std::string("Percentile ") + std::to_string(percentile) + "is not in [0,100]");
    }
}

bool parseHelp(Arguments& arguments)
{
    bool help{false};
    checkEraseOption(arguments, "--help", help);
    return help;
}

void AllOptions::parse(Arguments& arguments)
{
    model.parse(arguments);
    build.parse(arguments);
    system.parse(arguments);
    inference.parse(arguments);

    if ((!build.maxBatch && inference.batch && inference.batch != defaultBatch)
        || (build.maxBatch && build.maxBatch != defaultMaxBatch && !inference.batch))
    {
        // If either has selected implict batch and the other has selected explicit batch
        throw std::invalid_argument("Conflicting build and inference batch settings");
    }

    if (build.shapes.empty() && !inference.shapes.empty())
    {
        for (auto& s : inference.shapes)
        {
            insertShapes(build.shapes, s.first, s.second);
        }
        build.maxBatch = 0;
    }
    else
    {
        if (!build.shapes.empty() && inference.shapes.empty())
        {
            for (auto& s : build.shapes)
            {
                inference.shapes.insert({s.first, s.second[static_cast<size_t>(nvinfer1::OptProfileSelector::kOPT)]});
            }
        }
        if (!build.maxBatch)
        {
            inference.batch = 0;
        }
    }

    if (build.maxBatch && inference.batch)
    {
        // For implicit batch, check for compatibility and if --maxBatch is not given and inference batch is greater
        // than maxBatch, use inference batch also for maxBatch
        if (build.maxBatch != defaultMaxBatch && build.maxBatch < inference.batch)
        {
            throw std::invalid_argument("Build max batch " + std::to_string(build.maxBatch)
                + " is less than inference batch " + std::to_string(inference.batch));
        }
        else
        {
            if (build.maxBatch < inference.batch)
            {
                build.maxBatch = inference.batch;
            }
        }
    }

    reporting.parse(arguments);
    helps = parseHelp(arguments);

    if (!helps)
    {
        if (!build.load && model.baseModel.format == ModelFormat::kANY)
        {
            throw std::invalid_argument("Model missing or format not recognized");
        }
        if (!build.load && !build.maxBatch && model.baseModel.format != ModelFormat::kONNX)
        {
            throw std::invalid_argument("Explicit batch size not supported for Caffe and Uff models");
        }
        if (build.safe && system.DLACore >= 0)
        {
            auto checkSafeDLAFormats = [](const std::vector<IOFormat>& fmt)
            {
                return fmt.empty() ? false : std::all_of(fmt.begin(), fmt.end(), [](const IOFormat& pair)
                {
                    bool supported{false};
                    supported |= pair.first == nvinfer1::DataType::kINT8
                        && pair.second == 1U << static_cast<int>(nvinfer1::TensorFormat::kCHW32);
                    supported |= pair.first == nvinfer1::DataType::kHALF
                        && pair.second == 1U << static_cast<int>(nvinfer1::TensorFormat::kCHW16);
                    return supported;
                });
            };
            if (!checkSafeDLAFormats(build.inputFormats) || !checkSafeDLAFormats(build.inputFormats))
            {
                throw std::invalid_argument(
                    "I/O formats for safe DLA capability are restricted to fp16:chw16 or int8:chw32");
            }
            if (system.fallback)
            {
                throw std::invalid_argument("GPU fallback (--allowGPUFallback) not allowed for safe DLA capability");
            }
        }
    }
}

std::ostream& operator<<(std::ostream& os, const BaseModelOptions& options)
{
    os << "=== Model Options ===" << std::endl;

    os << "Format: ";
    switch (options.format)
    {
    case ModelFormat::kCAFFE:
    {
        os << "Caffe";
        break;
    }
    case ModelFormat::kONNX:
    {
        os << "ONNX";
        break;
    }
    case ModelFormat::kUFF:
    {
        os << "UFF";
        break;
    }
    case ModelFormat::kANY:
        os << "*";
        break;
    }
    os << std::endl << "Model: " << options.model << std::endl;

    return os;
}

std::ostream& operator<<(std::ostream& os, const UffInput& input)
{
    os << "Uff Inputs Layout: " << (input.NHWC ? "NHWC" : "NCHW") << std::endl;
    for (const auto& i : input.inputs)
    {
        os << "Input: " << i.first << "," << i.second.d[0] << "," << i.second.d[1] << "," << i.second.d[2] << std::endl;
    }

    return os;
}

std::ostream& operator<<(std::ostream& os, const ModelOptions& options)
{
    os << options.baseModel;
    switch (options.baseModel.format)
    {
    case ModelFormat::kCAFFE:
    {
        os << "Prototxt: " << options.prototxt << std::endl;
        break;
    }
    case ModelFormat::kUFF:
    {
        os << options.uffInputs;
        break;
    }
    case ModelFormat::kONNX: // Fallthrough: No options to report for ONNX or the generic case
    case ModelFormat::kANY:
        break;
    }

    os << "Output:";
    for (const auto& o : options.outputs)
    {
        os << " " << o;
    }
    os << std::endl;

    return os;
}

std::ostream& operator<<(std::ostream& os, const IOFormat& format)
{
    switch (format.first)
    {
    case nvinfer1::DataType::kFLOAT:
    {
        os << "fp32:";
        break;
    }
    case nvinfer1::DataType::kHALF:
    {
        os << "fp16:";
        break;
    }
    case nvinfer1::DataType::kINT8:
    {
        os << "int8:";
        break;
    }
    case nvinfer1::DataType::kINT32:
    {
        os << "int32:";
        break;
    }
    }

    for (int f = 0; f < nvinfer1::EnumMax<nvinfer1::TensorFormat>(); ++f)
    {
        if ((1U << f) & format.second)
        {
            if (f)
            {
                os << "+";
            }
            switch (nvinfer1::TensorFormat(f))
            {
            case nvinfer1::TensorFormat::kLINEAR:
            {
                os << "chw";
                break;
            }
            case nvinfer1::TensorFormat::kCHW2:
            {
                os << "chw2";
                break;
            }
            case nvinfer1::TensorFormat::kHWC8:
            {
                os << "hwc8";
                break;
            }
            case nvinfer1::TensorFormat::kCHW4:
            {
                os << "chw4";
                break;
            }
            case nvinfer1::TensorFormat::kCHW16:
            {
                os << "chw16";
                break;
            }
            case nvinfer1::TensorFormat::kCHW32:
            {
                os << "chw32";
                break;
            }
            }
        }
    }
    return os;
};

std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& dims)
{
    for (int i = 0; i < dims.nbDims; ++i)
    {
        os << (i ? "x" : "") << dims.d[i];
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const ShapeRange& dims)
{
    int i = 0;
    for (const auto& d : dims)
    {
        if (!d.nbDims)
        {
            break;
        }
        os << (i ? "+" : "") << d;
        ++i;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const BuildOptions& options)
{
// clang-format off
    os << "=== Build Options ==="                                                                                       << std::endl <<

          "Max batch: ";        printBatch(os, options.maxBatch)                                                        << std::endl <<
          "Workspace: "      << options.workspace << " MB"                                                              << std::endl <<
          "minTiming: "      << options.minTiming                                                                       << std::endl <<
          "avgTiming: "      << options.avgTiming                                                                       << std::endl <<
          "Precision: "      << (options.fp16 ? "FP16" : (options.int8 ? "INT8" : "FP32"))                              << std::endl <<
          "Calibration: "    << (options.int8 && options.calibration.empty() ? "Dynamic" : options.calibration.c_str()) << std::endl <<
          "Safe mode: "      << boolToEnabled(options.safe)                                                             << std::endl <<
          "Save engine: "    << (options.save ? options.engine : "")                                                    << std::endl <<
          "Load engine: "    << (options.load ? options.engine : "")                                                    << std::endl;
// clang-format on

    auto printIOFormats = [](std::ostream& os, const char* direction, const std::vector<IOFormat> formats)
    {
        if (formats.empty())
        {
            os << direction << "s format: fp32:CHW" << std::endl;
        }
        else
        {
            for(const auto& f : formats)
            {
                os << direction << ": " << f << std::endl;
            }
        }
    };

    printIOFormats(os, "Input", options.inputFormats);
    printIOFormats(os, "Output", options.outputFormats);
    printShapes(os, "build", options.shapes);

    return os;
}

std::ostream& operator<<(std::ostream& os, const SystemOptions& options)
{
// clang-format off
    os << "=== System Options ==="                                                                << std::endl <<

          "Device: "  << options.device                                                           << std::endl <<
          "DLACore: " << (options.DLACore != -1 ? std::to_string(options.DLACore) : "")           <<
                         (options.DLACore != -1 && options.fallback ? "(With GPU fallback)" : "") << std::endl;
// clang-format on
    os << "Plugins:";
    for (const auto p : options.plugins)
    {
        os << " " << p;
    }
    os << std::endl;

    return os;
}

std::ostream& operator<<(std::ostream& os, const InferenceOptions& options)
{
// clang-format off
    os << "=== Inference Options ==="                                        << std::endl <<

          "Batch: ";
    if (options.batch && options.shapes.empty())
    {
                          os << options.batch                                << std::endl;
    }
    else
    {
                          os << "Explicit"                                   << std::endl;
    }
    os << "Iterations: "     << options.iterations << " (" << options.warmup <<
                                                      " ms warm up)"         << std::endl <<
          "Inputs: "         << options.inputs                               << std::endl <<
          "Duration: "       << options.duration   << "s"                    << std::endl <<
          "Sleep time: "     << options.sleep      << "ms"                   << std::endl <<
          "Streams: "        << options.streams                              << std::endl <<
          "Spin-wait: "      << boolToEnabled(options.spin)                  << std::endl <<
          "Multithreading: " << boolToEnabled(options.threads)               << std::endl <<
          "CUDA Graph: "     << boolToEnabled(options.graph)                 << std::endl <<
          "Skip inference: " << boolToEnabled(options.skip)                  << std::endl;
// clang-format on
    if (options.batch)
    {
        printShapes(os, "inference", options.shapes);
    }

    return os;
}

std::ostream& operator<<(std::ostream& os, const ReportingOptions& options)
{
// clang-format off
    os << "=== Reporting Options ==="                                       << std::endl <<

          "Verbose: "                     << boolToEnabled(options.verbose) << std::endl <<
          "Averages: "                    << options.avgs << " inferences"  << std::endl <<
          "Percentile: "                  << options.percentile             << std::endl <<
          "Dump output: "                 << boolToEnabled(options.output)  << std::endl <<
          "Profile: "                     << boolToEnabled(options.profile) << std::endl <<
          "Export timing to JSON file: "  << options.exportTimes            << std::endl <<
          "Export output to JSON file: "  << options.exportOutput           << std::endl <<
          "Export profile to JSON file: " << options.exportProfile          << std::endl;
// clang-format on

    return os;
}

std::ostream& operator<<(std::ostream& os, const AllOptions& options)
{
    os << options.model << options.build << options.system << options.inference << options.reporting << std::endl;
    return os;
}

void BaseModelOptions::help(std::ostream& os)
{
// clang-format off
    os << "  --uff=<file>                UFF model"                                             << std::endl <<
          "  --onnx=<file>               ONNX model"                                            << std::endl <<
          "  --model=<file>              Caffe model (default = no model, random weights used)" << std::endl;
// clang-format on
}

void UffInput::help(std::ostream& os)
{
// clang-format off
    os << "  --uffInput=<name>,X,Y,Z     Input blob name and its dimensions (X,Y,Z=C,H,W), it can be specified "
                                                       "multiple times; at least one is required for UFF models" << std::endl <<
          "  --uffNHWC                   Set if inputs are in the NHWC layout instead of NCHW (use "             <<
                                                                    "X,Y,Z=H,W,C order in --uffInput)"           << std::endl;
// clang-format on
}

void ModelOptions::help(std::ostream& os)
{
// clang-format off
    os << "=== Model Options ==="                                                                                 << std::endl;
    BaseModelOptions::help(os);
    os << "  --deploy=<file>             Caffe prototxt file"                                                     << std::endl <<
          "  --output=<name>[,<name>]*   Output names (it can be specified multiple times); at least one output "
                                                                                  "is required for UFF and Caffe" << std::endl;
    UffInput::help(os);
// clang-format on
}

void BuildOptions::help(std::ostream& os)
{
// clang-format off
    os << "=== Build Options ==="                                                                                                     << std::endl <<

          "  --maxBatch                  Set max batch size and build an implicit batch engine (default = " << defaultMaxBatch << ")" << std::endl <<
          "  --explicitBatch             Use explicit batch sizes when building the engine (default = implicit)"                      << std::endl <<
          "  --minShapes=spec            Build with dynamic shapes using a profile with the min shapes provided"                      << std::endl <<
          "  --optShapes=spec            Build with dynamic shapes using a profile with the opt shapes provided"                      << std::endl <<
          "  --maxShapes=spec            Build with dynamic shapes using a profile with the max shapes provided"                      << std::endl <<
          "                              Note: if any of min/max/opt is missing, the profile will be completed using the shapes "     << std::endl <<
          "                                    provided and assuming that opt will be equal to max unless they are both specified;"   << std::endl <<           
          "                                    partially specified shapes are applied starting from the batch size;"                  << std::endl <<           
          "                                    dynamic shapes imply explicit batch"                                                   << std::endl <<           
          "                              Input shapes spec ::= Ishp[\",\"spec]"                                                       << std::endl <<
          "                                           Ishp ::= name\":\"shape"                                                        << std::endl <<
          "                                          shape ::= N[[\"x\"N]*\"*\"]"                                                     << std::endl <<
          "  --inputIOFormats=spec       Type and formats of the input tensors (default = all inputs in fp32:chw)"                    << std::endl <<
          "  --outputIOFormats=spec      Type and formats of the output tensors (default = all outputs in fp32:chw)"                  << std::endl <<
          "                              IO Formats: spec  ::= IOfmt[\",\"spec]"                                                      << std::endl <<
          "                                          IOfmt ::= type:fmt"                                                              << std::endl <<
          "                                          type  ::= \"fp32\"|\"fp16\"|\"int32\"|\"int8\""                                  << std::endl <<
          "                                          fmt   ::= (\"chw\"|\"chw2\"|\"chw4\"|\"hwc8\"|\"chw16\"|\"chw32\")[\"+\"fmt]"    << std::endl <<
          "  --workspace=N               Set workspace size in megabytes (default = "                      << defaultWorkspace << ")" << std::endl <<
          "  --minTiming=M               Set the minimum number of iterations used in kernel selection (default = "
                                                                                                           << defaultMinTiming << ")" << std::endl <<
          "  --avgTiming=M               Set the number of times averaged in each iteration for kernel selection (default = "
                                                                                                           << defaultAvgTiming << ")" << std::endl <<
          "  --fp16                      Enable fp16 mode (default = disabled)"                                                       << std::endl <<
          "  --int8                      Run in int8 mode (default = disabled)"                                                       << std::endl <<
          "  --calib=<file>              Read INT8 calibration cache file"                                                            << std::endl <<
          "  --safe                      Only test the functionality available in safety restricted flows"                            << std::endl <<
          "  --saveEngine=<file>         Save the serialized engine"                                                                  << std::endl <<
          "  --loadEngine=<file>         Load a serialized engine"                                                                    << std::endl;
// clang-format on
}

void SystemOptions::help(std::ostream& os)
{
// clang-format off
    os << "=== System Options ==="                                                                         << std::endl <<
          "  --device=N                  Select cuda device N (default = "         << defaultDevice << ")" << std::endl <<
          "  --useDLACore=N              Select DLA core N for layers that support DLA (default = none)"   << std::endl <<
          "  --allowGPUFallback          When DLA is enabled, allow GPU fallback for unsupported layers "
                                                                                    "(default = disabled)" << std::endl;
    os << "  --plugins                   Plugin library (.so) to load (can be specified multiple times)"   << std::endl;
// clang-format on
}

void InferenceOptions::help(std::ostream& os)
{
// clang-format off
    os << "=== Inference Options ==="                                                                                            << std::endl <<
          "  --batch=N                   Set batch size for implicit batch engines (default = "           << defaultBatch << ")" << std::endl <<
          "  --shapes=spec               Set input shapes for explicit batch and dynamic shapes inputs"                          << std::endl <<
          "  --loadInputs=<file>         Load input values from file (default = disabled)"                                       << std::endl <<
          "                              Input shapes spec ::= Ishp[\",\"spec]"                                                  << std::endl <<
          "                                           Ishp ::= name\":\"shape"                                                   << std::endl <<
          "                                          shape ::= N[[\"x\"N]*\"*\"]"                                                << std::endl <<
          "  --iterations=N              Run at least N inference iterations (default = "            << defaultIterations << ")" << std::endl <<
          "  --warmUp=N                  Run for N milliseconds to warmup before measuring performance (default = "
                                                                                                         << defaultWarmUp << ")" << std::endl <<
          "  --duration=N                Run performance measurements for at least N seconds wallclock time (default = "
                                                                                               << defaultDuration << ")"         << std::endl <<
          "  --sleepTime=N               Delay inference start with a gap of N milliseconds between launch and compute "
                                                                                            "(default = " << defaultSleep << ")" << std::endl <<
          "  --streams=N                 Instantiate N engines to use concurrently (default = "         << defaultStreams << ")" << std::endl <<
          "  --useSpinWait               Actively synchronize on GPU events. This option may decrease synchronization time but "
                                                                                "increase CPU usage and power (default = false)" << std::endl <<
          "  --threads                   Enable multithreading to drive engines with independent threads (default = disabled)"   << std::endl <<
          "  --useCudaGraph              Use cuda graph to capture engine execution and then launch inference (default = false)" << std::endl <<
          "  --buildOnly                 Skip inference perf measurement (default = disabled)"                                   << std::endl;
// clang-format on
}

void ReportingOptions::help(std::ostream& os)
{
// clang-format off
    os << "=== Reporting Options ==="                                                                    << std::endl <<
          "  --verbose                   Use verbose logging (default = false)"                          << std::endl <<
          "  --avgRuns=N                 Report performance measurements averaged over N consecutive "
                                                       "iterations (default = " << defaultAvgRuns << ")" << std::endl <<
          "  --percentile=P              Report performance for the P percentage (0<=P<=100, 0 "
                                        "representing max perf, and 100 representing min perf; (default"
                                                                      " = " << defaultPercentile << "%)" << std::endl <<
          "  --dumpOutput                Print the output tensor(s) of the last inference iteration "
                                                                                  "(default = disabled)" << std::endl <<
          "  --dumpProfile               Print profile information per layer (default = disabled)"       << std::endl <<
          "  --exportTimes=<file>        Write the timing results in a json file (default = disabled)"   << std::endl <<
          "  --exportOutput=<file>       Write the output tensors to a json file (default = disabled)"   << std::endl <<
          "  --exportProfile=<file>      Write the profile information per layer in a json file "
                                                                              "(default = disabled)"     << std::endl;
// clang-format on
}

void helpHelp(std::ostream& os)
{
// clang-format off
    os << "=== Help ==="                                     << std::endl <<
          "  --help                      Print this message" << std::endl;
// clang-format on
}

void AllOptions::help(std::ostream& os)
{
    ModelOptions::help(os);
    os << std::endl;
    BuildOptions::help(os);
    os << std::endl;
    InferenceOptions::help(os);
    os << std::endl;
// clang-format off
    os << "=== Build and Inference Batch Options ==="                                                                   << std::endl <<
          "                              When using implicit batch, the max batch size of the engine, if not given, "   << std::endl <<
          "                              is set to the inference batch size;"                                           << std::endl <<
          "                              when using explicit batch, if shapes are specified only for inference, they "  << std::endl <<
          "                              will be used also as min/opt/max in the build profile; if shapes are "         << std::endl <<
          "                              specified only for the build, the opt shapes will be used also for inference;" << std::endl <<
          "                              if both are specified, they must be compatible; and if explicit batch is "     << std::endl <<
          "                              enabled but neither is specified, the model must provide complete static"      << std::endl <<
          "                              dimensions, including batch size, for all inputs"                              << std::endl <<
    std::endl;
// clang-format on
    ReportingOptions::help(os);
    os << std::endl;
    SystemOptions::help(os);
    os << std::endl;
    helpHelp(os);
}

} // namespace sample
