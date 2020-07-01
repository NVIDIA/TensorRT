/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "NvInfer.h"

#include "sampleOptions.h"
#include "sampleUtils.h"

namespace sample
{

namespace
{

std::vector<std::string> splitToStringVec(const std::string& option, char separator)
{
    std::vector<std::string> options;

    for (size_t start = 0; start < option.length();)
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
T stringToValue(const std::string& option)
{
    return T{option};
}

template <>
int stringToValue<int>(const std::string& option)
{
    return std::stoi(option);
}

template <>
float stringToValue<float>(const std::string& option)
{
    return std::stof(option);
}

template <>
bool stringToValue<bool>(const std::string& option)
{
    return true;
}

template <>
std::vector<int> stringToValue<std::vector<int>>(const std::string& option)
{
    std::vector<int> shape;
    std::vector<std::string> dimsStrings = splitToStringVec(option, 'x');
    for (const auto& d : dimsStrings)
    {
        shape.push_back(stringToValue<int>(d));
    }
    return shape;
}

template <>
nvinfer1::DataType stringToValue<nvinfer1::DataType>(const std::string& option)
{
    const std::unordered_map<std::string, nvinfer1::DataType> strToDT{{"fp32", nvinfer1::DataType::kFLOAT},
        {"fp16", nvinfer1::DataType::kHALF}, {"int8", nvinfer1::DataType::kINT8},
        {"int32", nvinfer1::DataType::kINT32}};
    auto dt = strToDT.find(option);
    if (dt == strToDT.end())
    {
        throw std::invalid_argument("Invalid DataType " + option);
    }
    return dt->second;
}

template <>
nvinfer1::TensorFormats stringToValue<nvinfer1::TensorFormats>(const std::string& option)
{
    std::vector<std::string> optionStrings = splitToStringVec(option, '+');
    const std::unordered_map<std::string, nvinfer1::TensorFormat> strToFmt{{"chw", nvinfer1::TensorFormat::kLINEAR},
        {"chw2", nvinfer1::TensorFormat::kCHW2}, {"chw4", nvinfer1::TensorFormat::kCHW4},
        {"hwc8", nvinfer1::TensorFormat::kHWC8}, {"chw16", nvinfer1::TensorFormat::kCHW16},
        {"chw32", nvinfer1::TensorFormat::kCHW32}};
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
IOFormat stringToValue<IOFormat>(const std::string& option)
{
    IOFormat ioFormat{};
    size_t colon = option.find(':');

    if (colon == std::string::npos)
    {
        throw std::invalid_argument(std::string("Invalid IOFormat ") + option);
    }
    ioFormat.first = stringToValue<nvinfer1::DataType>(option.substr(0, colon));
    ioFormat.second = stringToValue<nvinfer1::TensorFormats>(option.substr(colon + 1));

    return ioFormat;
}

template <typename T>
std::pair<std::string, T> splitNameAndValue(const std::string& s)
{
    std::string tensorName;
    std::string valueString;
    // Split on the last :
    std::vector<std::string> nameRange{splitToStringVec(s, ':')};
    // Everything before the last : is the name
    tensorName = nameRange[0];
    for (size_t i = 1; i < nameRange.size() - 1; i++)
    {
        tensorName += ":" + nameRange[i];
    }
    // Value is the string element after the last :
    valueString = nameRange[nameRange.size() - 1];
    return std::pair<std::string, T>(tensorName, stringToValue<T>(valueString));
}

template <typename T>
void splitInsertKeyValue(const std::vector<std::string>& kvList, T& map)
{
    for (const auto& kv : kvList)
    {
        map.insert(splitNameAndValue<typename T::mapped_type>(kv));
    }
}

const char* boolToEnabled(bool enable)
{
    return enable ? "Enabled" : "Disabled";
}

template <typename T>
bool checkEraseOption(Arguments& arguments, const std::string& option, T& value)
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

// Like checkEraseOption, but sets value to false if arguments contain the option.
// This function should be used for options that default to true.
bool checkEraseNegativeOption(Arguments& arguments, const std::string& option, bool& value)
{
    bool dummy;
    if (checkEraseOption(arguments, option, dummy))
    {
        value = false;
        return true;
    }
    return false;
}

template <typename T>
bool checkEraseRepeatedOption(Arguments& arguments, const std::string& option, std::vector<T>& values)
{
    auto match = arguments.equal_range(option);
    if (match.first == match.second)
    {
        return false;
    }
    auto addValue = [&values](Arguments::value_type& value) { values.emplace_back(stringToValue<T>(value.second)); };
    std::for_each(match.first, match.second, addValue);
    arguments.erase(match.first, match.second);
    return true;
}

void insertShapesBuild(std::unordered_map<std::string, ShapeRange>& shapes, nvinfer1::OptProfileSelector selector,
    const std::string& name, const std::vector<int>& dims)
{
    shapes[name][static_cast<size_t>(selector)] = dims;
}

void insertShapesInference(
    std::unordered_map<std::string, std::vector<int>>& shapes, const std::string& name, const std::vector<int>& dims)
{
    shapes[name] = dims;
}

std::string removeSingleQuotationMarks(std::string& str)
{
    std::vector<std::string> strList{splitToStringVec(str, '\'')};
    // Remove all the escaped single quotation marks
    std::string retVal = "";
    // Do not really care about unterminated sequences
    for (size_t i = 0; i < strList.size(); i++)
    {
        retVal += strList[i];
    }
    return retVal;
}

bool getShapesBuild(Arguments& arguments, std::unordered_map<std::string, ShapeRange>& shapes, const char* argument,
    nvinfer1::OptProfileSelector selector)
{
    std::string list;
    bool retVal = checkEraseOption(arguments, argument, list);
    std::vector<std::string> shapeList{splitToStringVec(list, ',')};
    for (const auto& s : shapeList)
    {
        auto nameDimsPair = splitNameAndValue<std::vector<int>>(s);
        auto tensorName = removeSingleQuotationMarks(nameDimsPair.first);
        auto dims = nameDimsPair.second;
        insertShapesBuild(shapes, selector, tensorName, dims);
    }
    return retVal;
}

bool getShapesInference(
    Arguments& arguments, std::unordered_map<std::string, std::vector<int>>& shapes, const char* argument)
{
    std::string list;
    bool retVal = checkEraseOption(arguments, argument, list);
    std::vector<std::string> shapeList{splitToStringVec(list, ',')};
    for (const auto& s : shapeList)
    {
        auto nameDimsPair = splitNameAndValue<std::vector<int>>(s);
        auto tensorName = removeSingleQuotationMarks(nameDimsPair.first);
        auto dims = nameDimsPair.second;
        insertShapesInference(shapes, tensorName, dims);
    }
    return retVal;
}

void processShapes(
    std::unordered_map<std::string, ShapeRange>& shapes, bool minShapes, bool optShapes, bool maxShapes, bool calib)
{
    // Only accept optShapes only or all three of minShapes, optShapes, maxShapes
    if (((minShapes || maxShapes) && !optShapes)   // minShapes only, maxShapes only, both minShapes and maxShapes
        || (minShapes && !maxShapes && optShapes)  // both minShapes and optShapes
        || (!minShapes && maxShapes && optShapes)) // both maxShapes and optShapes
    {
        if (calib)
        {
            throw std::invalid_argument(
                "Must specify only --optShapesCalib or all of --minShapesCalib, --optShapesCalib, --maxShapesCalib");
        }
        else
        {
            throw std::invalid_argument(
                "Must specify only --optShapes or all of --minShapes, --optShapes, --maxShapes");
        }
    }

    // If optShapes only, expand optShapes to minShapes and maxShapes
    if (optShapes && !minShapes && !maxShapes)
    {
        std::unordered_map<std::string, ShapeRange> newShapes;
        for (auto& s : shapes)
        {
            insertShapesBuild(newShapes, nvinfer1::OptProfileSelector::kMIN, s.first,
                s.second[static_cast<size_t>(nvinfer1::OptProfileSelector::kOPT)]);
            insertShapesBuild(newShapes, nvinfer1::OptProfileSelector::kOPT, s.first,
                s.second[static_cast<size_t>(nvinfer1::OptProfileSelector::kOPT)]);
            insertShapesBuild(newShapes, nvinfer1::OptProfileSelector::kMAX, s.first,
                s.second[static_cast<size_t>(nvinfer1::OptProfileSelector::kOPT)]);
        }
        shapes = newShapes;
    }
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

std::ostream& printPrecision(std::ostream& os, const BuildOptions& options)
{
    os << "FP32";
    if (options.fp16)
    {
        os << "+FP16";
    }
    if (options.int8)
    {
        os << "+INT8";
    }
    return os;
}
} // namespace

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
    case ModelFormat::kONNX: break;
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
    auto getFormats = [&arguments](std::vector<IOFormat>& formatsVector, const char* argument) {
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

    bool explicitBatch{false};
    checkEraseOption(arguments, "--explicitBatch", explicitBatch);
    bool minShapes = getShapesBuild(arguments, shapes, "--minShapes", nvinfer1::OptProfileSelector::kMIN);
    bool optShapes = getShapesBuild(arguments, shapes, "--optShapes", nvinfer1::OptProfileSelector::kOPT);
    bool maxShapes = getShapesBuild(arguments, shapes, "--maxShapes", nvinfer1::OptProfileSelector::kMAX);
    processShapes(shapes, minShapes, optShapes, maxShapes, false);
    bool minShapesCalib
        = getShapesBuild(arguments, shapesCalib, "--minShapesCalib", nvinfer1::OptProfileSelector::kMIN);
    bool optShapesCalib
        = getShapesBuild(arguments, shapesCalib, "--optShapesCalib", nvinfer1::OptProfileSelector::kOPT);
    bool maxShapesCalib
        = getShapesBuild(arguments, shapesCalib, "--maxShapesCalib", nvinfer1::OptProfileSelector::kMAX);
    processShapes(shapesCalib, minShapesCalib, optShapesCalib, maxShapesCalib, true);
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

    bool best{false};
    checkEraseOption(arguments, "--best", best);
    if (best)
    {
        int8 = true;
        fp16 = true;
    }

    checkEraseNegativeOption(arguments, "--noTF32", tf32);
    checkEraseOption(arguments, "--fp16", fp16);
    checkEraseOption(arguments, "--int8", int8);
    checkEraseOption(arguments, "--safe", safe);
    bool calibCheck = checkEraseOption(arguments, "--calib", calibration);
    if (int8 && calibCheck && !shapes.empty() && shapesCalib.empty())
    {
        shapesCalib = shapes;
    }
    checkEraseNegativeOption(arguments, "--noBuilderCache", builderCache);

    std::string nvtxModeString;
    checkEraseOption(arguments, "--nvtxMode", nvtxModeString);
    if (nvtxModeString == "default")
    {
        nvtxMode = nvinfer1::ProfilingVerbosity::kDEFAULT;
    }
    else if (nvtxModeString == "none")
    {
        nvtxMode = nvinfer1::ProfilingVerbosity::kNONE;
    }
    else if (nvtxModeString == "verbose")
    {
        nvtxMode = nvinfer1::ProfilingVerbosity::kVERBOSE;
    }
    else if (!nvtxModeString.empty())
    {
        throw std::invalid_argument(std::string("Unknown nvtxMode: ") + nvtxModeString);
    }

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
    bool exposeDMA{false};
    if (checkEraseOption(arguments, "--exposeDMA", exposeDMA))
    {
        overlap = !exposeDMA;
    }
    checkEraseOption(arguments, "--useSpinWait", spin);
    checkEraseOption(arguments, "--threads", threads);
    checkEraseOption(arguments, "--useCudaGraph", graph);
    checkEraseOption(arguments, "--buildOnly", skip);

    std::string list;
    checkEraseOption(arguments, "--loadInputs", list);
    std::vector<std::string> inputsList{splitToStringVec(list, ',')};
    splitInsertKeyValue(inputsList, inputs);

    getShapesInference(arguments, shapes, "--shapes");

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
    bool helpLong{false};
    bool helpShort{false};
    checkEraseOption(arguments, "--help", helpLong);
    checkEraseOption(arguments, "-h", helpShort);
    return helpLong || helpShort;
}

void AllOptions::parse(Arguments& arguments)
{
    model.parse(arguments);
    build.parse(arguments);
    system.parse(arguments);
    inference.parse(arguments);

    if ((!build.maxBatch && inference.batch && inference.batch != defaultBatch && !build.shapes.empty())
        || (build.maxBatch && build.maxBatch != defaultMaxBatch && !inference.batch))
    {
        // If either has selected implict batch and the other has selected explicit batch
        throw std::invalid_argument("Conflicting build and inference batch settings");
    }

    if (build.shapes.empty() && !inference.shapes.empty())
    {
        for (auto& s : inference.shapes)
        {
            insertShapesBuild(build.shapes, nvinfer1::OptProfileSelector::kMIN, s.first, s.second);
            insertShapesBuild(build.shapes, nvinfer1::OptProfileSelector::kOPT, s.first, s.second);
            insertShapesBuild(build.shapes, nvinfer1::OptProfileSelector::kMAX, s.first, s.second);
        }
        build.maxBatch = 0;
    }
    else
    {
        if (!build.shapes.empty() && inference.shapes.empty())
        {
            for (auto& s : build.shapes)
            {
                insertShapesInference(
                    inference.shapes, s.first, s.second[static_cast<size_t>(nvinfer1::OptProfileSelector::kOPT)]);
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
            auto checkSafeDLAFormats = [](const std::vector<IOFormat>& fmt) {
                return fmt.empty() ? false : std::all_of(fmt.begin(), fmt.end(), [](const IOFormat& pair) {
                    bool supported{false};
                    const bool isCHW4{pair.second == 1U << static_cast<int>(nvinfer1::TensorFormat::kCHW4)};
                    const bool isCHW32{pair.second == 1U << static_cast<int>(nvinfer1::TensorFormat::kCHW32)};
                    const bool isCHW16{pair.second == 1U << static_cast<int>(nvinfer1::TensorFormat::kCHW16)};
                    supported |= pair.first == nvinfer1::DataType::kINT8 && (isCHW4 || isCHW32);
                    supported |= pair.first == nvinfer1::DataType::kHALF && (isCHW4 || isCHW16);
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
    case ModelFormat::kANY: os << "*"; break;
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
    case ModelFormat::kANY: break;
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
    case nvinfer1::DataType::kBOOL:
    {
        os << "Bool:";
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

std::ostream& operator<<(std::ostream& os, const ShapeRange& dims)
{
    int i = 0;
    for (const auto& d : dims)
    {
        if (!d.size())
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
          "Precision: ";        printPrecision(os, options)                                                             << std::endl <<
          "Calibration: "    << (options.int8 && options.calibration.empty() ? "Dynamic" : options.calibration.c_str()) << std::endl <<
          "Safe mode: "      << boolToEnabled(options.safe)                                                             << std::endl <<
          "Save engine: "    << (options.save ? options.engine : "")                                                    << std::endl <<
          "Load engine: "    << (options.load ? options.engine : "")                                                    << std::endl <<
          "Builder Cache: "  << boolToEnabled(options.builderCache)                                                     << std::endl <<
          "NVTX verbosity: " << static_cast<int>(options.nvtxMode)                                                      << std::endl;
    // clang-format on

    auto printIOFormats = [](std::ostream& os, const char* direction, const std::vector<IOFormat> formats) {
        if (formats.empty())
        {
            os << direction << "s format: fp32:CHW" << std::endl;
        }
        else
        {
            for (const auto& f : formats)
            {
                os << direction << ": " << f << std::endl;
            }
        }
    };

    printIOFormats(os, "Input", options.inputFormats);
    printIOFormats(os, "Output", options.outputFormats);
    printShapes(os, "build", options.shapes);
    printShapes(os, "calibration", options.shapesCalib);

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
    os << "=== Inference Options ==="                                << std::endl <<

          "Batch: ";
    if (options.batch && options.shapes.empty())
    {
                          os << options.batch                        << std::endl;
    }
    else
    {
                          os << "Explicit"                           << std::endl;
    }
    printShapes(os, "inference", options.shapes);
    os << "Iterations: "     << options.iterations                   << std::endl <<
          "Duration: "       << options.duration   << "s (+ "
                             << options.warmup     << "ms warm up)"  << std::endl <<
          "Sleep time: "     << options.sleep      << "ms"           << std::endl <<
          "Streams: "        << options.streams                      << std::endl <<
          "ExposeDMA: "      << boolToEnabled(!options.overlap)      << std::endl <<
          "Spin-wait: "      << boolToEnabled(options.spin)          << std::endl <<
          "Multithreading: " << boolToEnabled(options.threads)       << std::endl <<
          "CUDA Graph: "     << boolToEnabled(options.graph)         << std::endl <<
          "Skip inference: " << boolToEnabled(options.skip)          << std::endl;

    // clang-format on
    os << "Inputs:" << std::endl;
    for (const auto& input : options.inputs)
    {
        os << input.first << "<-" << input.second << std::endl;
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
          "  --minShapesCalib=spec       Calibrate with dynamic shapes using a profile with the min shapes provided"                  << std::endl <<
          "  --optShapesCalib=spec       Calibrate with dynamic shapes using a profile with the opt shapes provided"                  << std::endl <<
          "  --maxShapesCalib=spec       Calibrate with dynamic shapes using a profile with the max shapes provided"                  << std::endl <<
          "                              Note: All three of min, opt and max shapes must be supplied."                                << std::endl <<
          "                                    However, if only opt shapes is supplied then it will be expanded so"                   << std::endl <<
          "                                    that min shapes and max shapes are set to the same values as opt shapes."              << std::endl <<
          "                                    In addition, use of dynamic shapes implies explicit batch."                            << std::endl <<
          "                                    Input names can be wrapped with escaped single quotes (ex: \\\'Input:0\\\')."          << std::endl <<
          "                              Example input shapes spec: input0:1x3x256x256,input1:1x3x128x128"                            << std::endl <<
          "                              Each input shape is supplied as a key-value pair where key is the input name and"            << std::endl <<
          "                              value is the dimensions (including the batch dimension) to be used for that input."          << std::endl <<
          "                              Each key-value pair has the key and value separated using a colon (:)."                      << std::endl <<
          "                              Multiple input shapes can be provided via comma-separated key-value pairs."                  << std::endl <<
          "  --inputIOFormats=spec       Type and formats of the input tensors (default = all inputs in fp32:chw)"                    << std::endl <<
          "                              Note: If this option is specified, please make sure that all inputs are in the same order "  << std::endl <<
          "                                     as network inputs ID."                                                                << std::endl <<
          "  --outputIOFormats=spec      Type and formats of the output tensors (default = all outputs in fp32:chw)"                  << std::endl <<
          "                              Note: If this option is specified, please make sure that all outputs are in the same order " << std::endl <<
          "                                     as network outputs ID."                                                               << std::endl <<
          "                              IO Formats: spec  ::= IOfmt[\",\"spec]"                                                      << std::endl <<
          "                                          IOfmt ::= type:fmt"                                                              << std::endl <<
          "                                          type  ::= \"fp32\"|\"fp16\"|\"int32\"|\"int8\""                                  << std::endl <<
          "                                          fmt   ::= (\"chw\"|\"chw2\"|\"chw4\"|\"hwc8\"|\"chw16\"|\"chw32\")[\"+\"fmt]"    << std::endl <<
          "  --workspace=N               Set workspace size in megabytes (default = "                      << defaultWorkspace << ")" << std::endl <<
          "  --noBuilderCache            Disable timing cache in builder (default is to enable timing cache)"                         << std::endl <<
          "  --nvtxMode=[default|verbose|none] Specify NVTX annotation verbosity"                                                     << std::endl <<
          "  --minTiming=M               Set the minimum number of iterations used in kernel selection (default = "
                                                                                                           << defaultMinTiming << ")" << std::endl <<
          "  --avgTiming=M               Set the number of times averaged in each iteration for kernel selection (default = "
                                                                                                           << defaultAvgTiming << ")" << std::endl <<
          "  --noTF32                    Disable tf32 precision (default is to enable tf32, in addition to fp32)"                     << std::endl <<
          "  --fp16                      Enable fp16 precision, in addition to fp32 (default = disabled)"                             << std::endl <<
          "  --int8                      Enable int8 precision, in addition to fp32 (default = disabled)"                             << std::endl <<
          "  --best                      Enable all precisions to achieve the best performance (default = disabled)"                  << std::endl <<
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
    os << "=== Inference Options ==="                                                                                               << std::endl <<
          "  --batch=N                   Set batch size for implicit batch engines (default = "              << defaultBatch << ")" << std::endl <<
          "  --shapes=spec               Set input shapes for dynamic shapes inference inputs."                                     << std::endl <<
          "                              Note: Use of dynamic shapes implies explicit batch."                                       << std::endl <<
          "                                    Input names can be wrapped with escaped single quotes (ex: \\\'Input:0\\\')."        << std::endl <<
          "                              Example input shapes spec: input0:1x3x256x256, input1:1x3x128x128"                         << std::endl <<
          "                              Each input shape is supplied as a key-value pair where key is the input name and"          << std::endl <<
          "                              value is the dimensions (including the batch dimension) to be used for that input."        << std::endl <<
          "                              Each key-value pair has the key and value separated using a colon (:)."                    << std::endl <<
          "                              Multiple input shapes can be provided via comma-separated key-value pairs."                << std::endl <<
          "  --loadInputs=spec           Load input values from files (default = generate random inputs). Input names can be "
                                                                                       "wrapped with single quotes (ex: 'Input:0')" << std::endl <<
          "                              Input values spec ::= Ival[\",\"spec]"                                                     << std::endl <<
          "                                           Ival ::= name\":\"file"                                                       << std::endl <<
          "  --iterations=N              Run at least N inference iterations (default = "               << defaultIterations << ")" << std::endl <<
          "  --warmUp=N                  Run for N milliseconds to warmup before measuring performance (default = "
                                                                                                            << defaultWarmUp << ")" << std::endl <<
          "  --duration=N                Run performance measurements for at least N seconds wallclock time (default = "
                                                                                                          << defaultDuration << ")" << std::endl <<
          "  --sleepTime=N               Delay inference start with a gap of N milliseconds between launch and compute "
                                                                                               "(default = " << defaultSleep << ")" << std::endl <<
          "  --streams=N                 Instantiate N engines to use concurrently (default = "            << defaultStreams << ")" << std::endl <<
          "  --exposeDMA                 Serialize DMA transfers to and from device. (default = disabled)"                          << std::endl <<
          "  --useSpinWait               Actively synchronize on GPU events. This option may decrease synchronization time but "
                                                                             "increase CPU usage and power (default = disabled)"    << std::endl <<
          "  --threads                   Enable multithreading to drive engines with independent threads (default = disabled)"      << std::endl <<
          "  --useCudaGraph              Use cuda graph to capture engine execution and then launch inference (default = disabled)" << std::endl <<
          "  --buildOnly                 Skip inference perf measurement (default = disabled)"                                      << std::endl;
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
          "  --help, -h                  Print this message" << std::endl;
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
