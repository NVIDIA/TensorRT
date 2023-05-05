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
#include <cctype>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "NvInfer.h"

#include "logger.h"
#include "sampleOptions.h"
#include "sampleUtils.h"
using namespace nvinfer1;
namespace sample
{

namespace
{

template <typename T>
T stringToValue(const std::string& option)
{
    return T{option};
}

template <>
int32_t stringToValue<int32_t>(const std::string& option)
{
    return std::stoi(option);
}

template <>
float stringToValue<float>(const std::string& option)
{
    return std::stof(option);
}

template <>
double stringToValue<double>(const std::string& option)
{
    return std::stod(option);
}

template <>
bool stringToValue<bool>(const std::string& option)
{
    return true;
}

template <>
std::vector<int32_t> stringToValue<std::vector<int32_t>>(const std::string& option)
{
    std::vector<int32_t> shape;
    std::vector<std::string> dimsStrings = splitToStringVec(option, 'x');
    for (const auto& d : dimsStrings)
    {
        shape.push_back(stringToValue<int32_t>(d));
    }
    return shape;
}

template <>
nvinfer1::DataType stringToValue<nvinfer1::DataType>(const std::string& option)
{
    const std::unordered_map<std::string, nvinfer1::DataType> strToDT{{"fp32", nvinfer1::DataType::kFLOAT},
        {"fp16", nvinfer1::DataType::kHALF}, {"int8", nvinfer1::DataType::kINT8}, {"fp8", nvinfer1::DataType::kFP8},
        {"int32", nvinfer1::DataType::kINT32}};
    const auto& dt = strToDT.find(option);
    if (dt == strToDT.end())
    {
        throw std::invalid_argument("Invalid DataType " + option);
    }
    return dt->second;
}

template <>
nvinfer1::DeviceType stringToValue<nvinfer1::DeviceType>(std::string const& option)
{
    std::unordered_map<std::string, nvinfer1::DeviceType> const strToDevice = {
        {"GPU", nvinfer1::DeviceType::kGPU},
        {"DLA", nvinfer1::DeviceType::kDLA},
    };
    auto const& device = strToDevice.find(option);
    if (device == strToDevice.end())
    {
        throw std::invalid_argument("Invalid Device Type " + option);
    }
    return device->second;
}

template <>
nvinfer1::TensorFormats stringToValue<nvinfer1::TensorFormats>(const std::string& option)
{
    std::vector<std::string> optionStrings = splitToStringVec(option, '+');
    const std::unordered_map<std::string, nvinfer1::TensorFormat> strToFmt{{"chw", nvinfer1::TensorFormat::kLINEAR},
        {"chw2", nvinfer1::TensorFormat::kCHW2}, {"chw4", nvinfer1::TensorFormat::kCHW4},
        {"hwc8", nvinfer1::TensorFormat::kHWC8}, {"chw16", nvinfer1::TensorFormat::kCHW16},
        {"chw32", nvinfer1::TensorFormat::kCHW32}, {"dhwc8", nvinfer1::TensorFormat::kDHWC8},
        {"cdhw32", nvinfer1::TensorFormat::kCDHW32}, {"hwc", nvinfer1::TensorFormat::kHWC},
        {"dhwc", nvinfer1::TensorFormat::kDHWC}, {"dla_linear", nvinfer1::TensorFormat::kDLA_LINEAR},
        {"dla_hwc4", nvinfer1::TensorFormat::kDLA_HWC4}};
    nvinfer1::TensorFormats formats{};
    for (auto f : optionStrings)
    {
        const auto& tf = strToFmt.find(f);
        if (tf == strToFmt.end())
        {
            throw std::invalid_argument(std::string("Invalid TensorFormat ") + f);
        }
        formats |= 1U << static_cast<int32_t>(tf->second);
    }

    return formats;
}

template <>
IOFormat stringToValue<IOFormat>(const std::string& option)
{
    IOFormat ioFormat{};
    const size_t colon = option.find(':');

    if (colon == std::string::npos)
    {
        throw std::invalid_argument(std::string("Invalid IOFormat ") + option);
    }

    ioFormat.first = stringToValue<nvinfer1::DataType>(option.substr(0, colon));
    ioFormat.second = stringToValue<nvinfer1::TensorFormats>(option.substr(colon + 1));

    return ioFormat;
}

template <>
SparsityFlag stringToValue<SparsityFlag>(std::string const& option)
{
    std::unordered_map<std::string, SparsityFlag> const table{
        {"disable", SparsityFlag::kDISABLE}, {"enable", SparsityFlag::kENABLE}, {"force", SparsityFlag::kFORCE}};
    auto search = table.find(option);
    if (search == table.end())
    {
        throw std::invalid_argument(std::string("Unknown sparsity mode: ") + option);
    }
    return search->second;
}

template <typename T>
std::pair<std::string, T> splitNameAndValue(const std::string& s)
{
    std::string tensorName;
    std::string valueString;

    // Support 'inputName':Path format for --loadInputs flag when dealing with Windows paths.
    // i.e. 'inputName':c:\inputData
    std::vector<std::string> quoteNameRange{ splitToStringVec(s, '\'') };
    // splitToStringVec returns the entire string when delimiter is not found, so it's size is always at least 1
    if (quoteNameRange.size() != 1)
    {
        if (quoteNameRange.size() != 3)
        {
            throw std::invalid_argument(std::string("Found invalid number of \'s when parsing ") + s +
                std::string(". Expected: 2, received: ") + std::to_string(quoteNameRange.size() -1));
        }
        // Everything before the second "'" is the name.
        tensorName = quoteNameRange[0] + quoteNameRange[1];
        // Path is the last string - ignoring leading ":" so slice it with [1:]
        valueString = quoteNameRange[2].substr(1);
        return std::pair<std::string, T>(tensorName, stringToValue<T>(valueString));
    }

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

//! A helper function similar to sep.join(list) in Python.
template <typename T>
std::string joinValuesToString(std::vector<T> const& list, std::string const& sep)
{
    std::ostringstream os;
    for (int32_t i = 0, n = list.size(); i < n; ++i)
    {
        os << list[i];
        if (i != n - 1)
        {
            os << sep;
        }
    }
    return os.str();
}

template <typename T, size_t N>
std::string joinValuesToString(std::array<T, N> const& list, std::string const& sep)
{
    return joinValuesToString(std::vector<T>(list.begin(), list.end()), sep);
}

//! Check if input option exists in input arguments.
//! If it does: return its value, erase the argument and return true.
//! If it does not: return false.
template <typename T>
bool getAndDelOption(Arguments& arguments, const std::string& option, T& value)
{
    const auto match = arguments.find(option);
    if (match != arguments.end())
    {
        value = stringToValue<T>(match->second);
        arguments.erase(match);
        return true;
    }

    return false;
}

//! Check if input option exists in input arguments.
//! If it does: return false in value, erase the argument and return true.
//! If it does not: return false.
bool getAndDelNegOption(Arguments& arguments, const std::string& option, bool& value)
{
    bool dummy;
    if (getAndDelOption(arguments, option, dummy))
    {
        value = false;
        return true;
    }
    return false;
}

//! Check if input option exists in input arguments.
//! If it does: add all the matched arg values to values vector, erase the argument and return true.
//! If it does not: return false.
template <typename T>
bool getAndDelRepeatedOption(Arguments& arguments, const std::string& option, std::vector<T>& values)
{
    const auto match = arguments.equal_range(option);
    if (match.first == match.second)
    {
        return false;
    }

    auto addToValues
        = [&values](Arguments::value_type& argValue) { values.emplace_back(stringToValue<T>(argValue.second)); };
    std::for_each(match.first, match.second, addToValues);
    arguments.erase(match.first, match.second);

    return true;
}

void insertShapesBuild(BuildOptions::ShapeProfile& shapes, nvinfer1::OptProfileSelector selector,
    const std::string& name, const std::vector<int32_t>& dims)
{
    shapes[name][static_cast<size_t>(selector)] = dims;
}

void insertShapesInference(
    InferenceOptions::ShapeProfile& shapes, std::string const& name, std::vector<int32_t> const& dims)
{
    shapes[name] = dims;
}

std::string removeSingleQuotationMarks(std::string& str)
{
    std::vector<std::string> strList{splitToStringVec(str, '\'')};
    // Remove all the escaped single quotation marks
    std::string retVal;
    // Do not really care about unterminated sequences
    for (size_t i = 0; i < strList.size(); i++)
    {
        retVal += strList[i];
    }
    return retVal;
}

void getLayerPrecisions(Arguments& arguments, char const* argument, LayerPrecisions& layerPrecisions)
{
    std::string list;
    if (!getAndDelOption(arguments, argument, list))
    {
        return;
    }

    // The layerPrecisions flag contains comma-separated layerName:precision pairs.
    std::vector<std::string> precisionList{splitToStringVec(list, ',')};
    for (auto const& s : precisionList)
    {
        auto namePrecisionPair = splitNameAndValue<nvinfer1::DataType>(s);
        auto const layerName = removeSingleQuotationMarks(namePrecisionPair.first);
        layerPrecisions[layerName] = namePrecisionPair.second;
    }
}

void getLayerOutputTypes(Arguments& arguments, char const* argument, LayerOutputTypes& layerOutputTypes)
{
    std::string list;
    if (!getAndDelOption(arguments, argument, list))
    {
        return;
    }

    // The layerOutputTypes flag contains comma-separated layerName:types pairs.
    std::vector<std::string> precisionList{splitToStringVec(list, ',')};
    for (auto const& s : precisionList)
    {
        auto namePrecisionPair = splitNameAndValue<std::string>(s);
        auto const layerName = removeSingleQuotationMarks(namePrecisionPair.first);
        auto const typeStrings = splitToStringVec(namePrecisionPair.second, '+');
        std::vector<nvinfer1::DataType> typeVec(typeStrings.size(), nvinfer1::DataType::kFLOAT);
        std::transform(typeStrings.begin(), typeStrings.end(), typeVec.begin(), stringToValue<nvinfer1::DataType>);
        layerOutputTypes[layerName] = typeVec;
    }
}

void getLayerDeviceTypes(Arguments& arguments, char const* argument, LayerDeviceTypes& layerDeviceTypes)
{
    std::string list;
    if (!getAndDelOption(arguments, argument, list))
    {
        return;
    }

    // The layerDeviceTypes flag contains comma-separated layerName:deviceType pairs.
    std::vector<std::string> deviceList{splitToStringVec(list, ',')};
    for (auto const& s : deviceList)
    {
        auto nameDevicePair = splitNameAndValue<std::string>(s);
        auto const layerName = removeSingleQuotationMarks(nameDevicePair.first);
        layerDeviceTypes[layerName] = stringToValue<nvinfer1::DeviceType>(nameDevicePair.second);
    }
}

bool getShapesBuild(Arguments& arguments, BuildOptions::ShapeProfile& shapes, char const* argument,
    nvinfer1::OptProfileSelector selector)
{
    std::string list;
    bool retVal = getAndDelOption(arguments, argument, list);
    std::vector<std::string> shapeList{splitToStringVec(list, ',')};
    for (const auto& s : shapeList)
    {
        auto nameDimsPair = splitNameAndValue<std::vector<int32_t>>(s);
        auto tensorName = removeSingleQuotationMarks(nameDimsPair.first);
        auto dims = nameDimsPair.second;
        insertShapesBuild(shapes, selector, tensorName, dims);
    }
    return retVal;
}

bool getShapesInference(Arguments& arguments, InferenceOptions::ShapeProfile& shapes, const char* argument)
{
    std::string list;
    bool retVal = getAndDelOption(arguments, argument, list);
    std::vector<std::string> shapeList{splitToStringVec(list, ',')};
    for (const auto& s : shapeList)
    {
        auto nameDimsPair = splitNameAndValue<std::vector<int32_t>>(s);
        auto tensorName = removeSingleQuotationMarks(nameDimsPair.first);
        auto dims = nameDimsPair.second;
        insertShapesInference(shapes, tensorName, dims);
    }
    return retVal;
}

void fillShapes(BuildOptions::ShapeProfile& shapes, std::string const& name, ShapeRange const& sourceShapeRange,
    nvinfer1::OptProfileSelector minDimsSource, nvinfer1::OptProfileSelector optDimsSource,
    nvinfer1::OptProfileSelector maxDimsSource)
{
    insertShapesBuild(
        shapes, nvinfer1::OptProfileSelector::kMIN, name, sourceShapeRange[static_cast<size_t>(minDimsSource)]);
    insertShapesBuild(
        shapes, nvinfer1::OptProfileSelector::kOPT, name, sourceShapeRange[static_cast<size_t>(optDimsSource)]);
    insertShapesBuild(
        shapes, nvinfer1::OptProfileSelector::kMAX, name, sourceShapeRange[static_cast<size_t>(maxDimsSource)]);
}

void processShapes(BuildOptions::ShapeProfile& shapes, bool minShapes, bool optShapes, bool maxShapes, bool calib)
{
    // Only accept optShapes only or all three of minShapes, optShapes, maxShapes when calib is set
    if (((minShapes || maxShapes) && !optShapes)   // minShapes only, maxShapes only, both minShapes and maxShapes
        || (minShapes && !maxShapes && optShapes)  // both minShapes and optShapes
        || (!minShapes && maxShapes && optShapes)) // both maxShapes and optShapes
    {
        if (calib)
        {
            throw std::invalid_argument(
                "Must specify only --optShapesCalib or all of --minShapesCalib, --optShapesCalib, --maxShapesCalib");
        }
    }

    if (!minShapes && !optShapes && !maxShapes)
    {
        return;
    }

    BuildOptions::ShapeProfile newShapes;
    for (auto& s : shapes)
    {
        nvinfer1::OptProfileSelector minDimsSource, optDimsSource, maxDimsSource;
        minDimsSource = nvinfer1::OptProfileSelector::kMIN;
        optDimsSource = nvinfer1::OptProfileSelector::kOPT;
        maxDimsSource = nvinfer1::OptProfileSelector::kMAX;

        // Populate missing minShapes
        if (!minShapes)
        {
            if (optShapes)
            {
                minDimsSource = optDimsSource;
                sample::gLogWarning << "optShapes is being broadcasted to minShapes for tensor " << s.first
                                    << std::endl;
            }
            else
            {
                minDimsSource = maxDimsSource;
                sample::gLogWarning << "maxShapes is being broadcasted to minShapes for tensor " << s.first
                                    << std::endl;
            }
        }

        // Populate missing optShapes
        if (!optShapes)
        {
            if (maxShapes)
            {
                optDimsSource = maxDimsSource;
                sample::gLogWarning << "maxShapes is being broadcasted to optShapes for tensor " << s.first
                                    << std::endl;
            }
            else
            {
                optDimsSource = minDimsSource;
                sample::gLogWarning << "minShapes is being broadcasted to optShapes for tensor " << s.first
                                    << std::endl;
            }
        }

        // Populate missing maxShapes
        if (!maxShapes)
        {
            if (optShapes)
            {
                maxDimsSource = optDimsSource;
                sample::gLogWarning << "optShapes is being broadcasted to maxShapes for tensor " << s.first
                                    << std::endl;
            }
            else
            {
                maxDimsSource = minDimsSource;
                sample::gLogWarning << "minShapes is being broadcasted to maxShapes for tensor " << s.first
                                    << std::endl;
            }
        }

        fillShapes(newShapes, s.first, s.second, minDimsSource, optDimsSource, maxDimsSource);
    }
    shapes = newShapes;
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

std::ostream& printBatch(std::ostream& os, int32_t maxBatch)
{
    if (maxBatch != maxBatchNotProvided)
    {
        os << maxBatch;
    }
    else
    {
        os << "explicit batch";
    }
    return os;
}

std::ostream& printTacticSources(
    std::ostream& os, nvinfer1::TacticSources enabledSources, nvinfer1::TacticSources disabledSources)
{
    if (!enabledSources && !disabledSources)
    {
        os << "Using default tactic sources";
    }
    else
    {
        auto const addSource = [&](uint32_t source, std::string const& name) {
            if (enabledSources & source)
            {
                os << name << " [ON], ";
            }
            else if (disabledSources & source)
            {
                os << name << " [OFF], ";
            }
        };

        addSource(1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS), "cublas");
        addSource(1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS_LT), "cublasLt");
        addSource(1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUDNN), "cudnn");
        addSource(1U << static_cast<uint32_t>(nvinfer1::TacticSource::kEDGE_MASK_CONVOLUTIONS), "edge mask convolutions");
        addSource(1U << static_cast<uint32_t>(nvinfer1::TacticSource::kJIT_CONVOLUTIONS), "JIT convolutions");
    }
    return os;
}

std::ostream& printPrecision(std::ostream& os, BuildOptions const& options)
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
    if (options.fp8)
    {
        os << "+FP8";
    }
    if (options.precisionConstraints == PrecisionConstraints::kOBEY)
    {
        os << " (obey precision constraints)";
    }
    if (options.precisionConstraints == PrecisionConstraints::kPREFER)
    {
        os << " (prefer precision constraints)";
    }
    return os;
}

std::ostream& printTempfileControls(std::ostream& os, TempfileControlFlags const tempfileControls)
{
    auto getFlag = [&](TempfileControlFlag f) -> char const* {
        bool allowed = !!(tempfileControls & (1U << static_cast<int64_t>(f)));
        return allowed ? "allow" : "deny";
    };
    auto const inMemory = getFlag(TempfileControlFlag::kALLOW_IN_MEMORY_FILES);
    auto const temporary = getFlag(TempfileControlFlag::kALLOW_TEMPORARY_FILES);

    os << "{ in_memory: " << inMemory << ", temporary: " << temporary << " }";

    return os;
}

std::ostream& printTimingCache(std::ostream& os, TimingCacheMode const& timingCacheMode)
{
    switch (timingCacheMode)
    {
    case TimingCacheMode::kGLOBAL: os << "global"; break;
    case TimingCacheMode::kLOCAL: os << "local"; break;
    case TimingCacheMode::kDISABLE: os << "disable"; break;
    }
    return os;
}

std::ostream& printSparsity(std::ostream& os, BuildOptions const& options)
{
    switch (options.sparsity)
    {
    case SparsityFlag::kDISABLE: os << "Disabled"; break;
    case SparsityFlag::kENABLE: os << "Enabled"; break;
    case SparsityFlag::kFORCE: os << "Forced"; break;
    }

    return os;
}

std::ostream& printMemoryPools(std::ostream& os, BuildOptions const& options)
{
    auto const printValueOrDefault = [&os](double const val) {
        if (val >= 0)
        {
            os << val << " MiB";
        }
        else
        {
            os << "default";
        }
    };
    os << "workspace: ";
    printValueOrDefault(options.workspace);
    os << ", ";
    os << "dlaSRAM: ";
    printValueOrDefault(options.dlaSRAM);
    os << ", ";
    os << "dlaLocalDRAM: ";
    printValueOrDefault(options.dlaLocalDRAM);
    os << ", ";
    os << "dlaGlobalDRAM: ";
    printValueOrDefault(options.dlaGlobalDRAM);
    return os;
}

std::string previewFeatureToString(PreviewFeature feature)
{
    // clang-format off
    switch (feature)
    {
    case PreviewFeature::kFASTER_DYNAMIC_SHAPES_0805: return "kFASTER_DYNAMIC_SHAPES_0805";
    case PreviewFeature::kDISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805: return "kDISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805";
    case PreviewFeature::kPROFILE_SHARING_0806: return "kPROFILE_SHARING_0806";
    }
    return "Invalid Preview Feature";
    // clang-format on
}

std::ostream& printPreviewFlags(std::ostream& os, BuildOptions const& options)
{
    if (options.previewFeatures.empty())
    {
        os << "Use default preview flags.";
        return os;
    }

    auto const addFlag = [&](PreviewFeature feat) {
        int32_t featVal = static_cast<int32_t>(feat);
        if (options.previewFeatures.find(featVal) != options.previewFeatures.end())
        {
            os << previewFeatureToString(feat) << (options.previewFeatures.at(featVal) ? " [ON], " : " [OFF], ");
        }
    };

    addFlag(PreviewFeature::kFASTER_DYNAMIC_SHAPES_0805);
    addFlag(PreviewFeature::kDISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805);
    addFlag(PreviewFeature::kPROFILE_SHARING_0806);

    return os;
}

} // namespace

Arguments argsToArgumentsMap(int32_t argc, char* argv[])
{
    Arguments arguments;
    for (int32_t i = 1; i < argc; ++i)
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
    if (getAndDelOption(arguments, "--onnx", model))
    {
        format = ModelFormat::kONNX;
    }
    else if (getAndDelOption(arguments, "--uff", model))
    {
        format = ModelFormat::kUFF;
    }
    else if (getAndDelOption(arguments, "--model", model))
    {
        format = ModelFormat::kCAFFE;
    }
}

void UffInput::parse(Arguments& arguments)
{
    getAndDelOption(arguments, "--uffNHWC", NHWC);
    std::vector<std::string> args;
    if (getAndDelRepeatedOption(arguments, "--uffInput", args))
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
        getAndDelOption(arguments, "--deploy", prototxt);
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
        if (getAndDelOption(arguments, "--deploy", prototxt))
        {
            baseModel.format = ModelFormat::kCAFFE;
        }
        break;
    }
    }

    // The --output flag should only be used with Caffe and UFF. It has no effect on ONNX.
    std::vector<std::string> outArgs;
    if (getAndDelRepeatedOption(arguments, "--output", outArgs))
    {
        for (const auto& o : outArgs)
        {
            for (auto& v : splitToStringVec(o, ','))
            {
                outputs.emplace_back(std::move(v));
            }
        }
    }
    if (baseModel.format == ModelFormat::kCAFFE || baseModel.format == ModelFormat::kUFF)
    {
        if (outputs.empty())
        {
            throw std::invalid_argument("Caffe and Uff models require at least one output");
        }
    }
    else if (baseModel.format == ModelFormat::kONNX)
    {
        if (!outputs.empty())
        {
            throw std::invalid_argument("The --output flag should not be used with ONNX models.");
        }
    }
}

void getTempfileControls(Arguments& arguments, char const* argument, TempfileControlFlags& tempfileControls)
{
    std::string list;
    if (!getAndDelOption(arguments, argument, list))
    {
        return;
    }

    std::vector<std::string> controlList{splitToStringVec(list, ',')};
    for (auto const& s : controlList)
    {
        auto controlAllowPair = splitNameAndValue<std::string>(s);
        bool allowed{false};
        int32_t offset{-1};

        if (controlAllowPair.second.compare("allow") == 0)
        {
            allowed = true;
        }
        else if (controlAllowPair.second.compare("deny") != 0)
        {
            throw std::invalid_argument("--tempfileControls value should be `deny` or `allow`");
        }

        if (controlAllowPair.first.compare("in_memory") == 0)
        {
            offset = static_cast<int32_t>(TempfileControlFlag::kALLOW_IN_MEMORY_FILES);
        }
        else if (controlAllowPair.first.compare("temporary") == 0)
        {
            offset = static_cast<int32_t>(TempfileControlFlag::kALLOW_TEMPORARY_FILES);
        }
        else
        {
            throw std::invalid_argument(std::string{"Unknown --tempfileControls key "} + controlAllowPair.first);
        }

        if (allowed)
        {
            tempfileControls |= (1U << offset);
        }
        else
        {
            tempfileControls &= ~(1U << offset);
        }
    }
}

void BuildOptions::parse(Arguments& arguments)
{
    auto getFormats = [&arguments](std::vector<IOFormat>& formatsVector, const char* argument) {
        std::string list;
        getAndDelOption(arguments, argument, list);
        std::vector<std::string> formats{splitToStringVec(list, ',')};
        for (const auto& f : formats)
        {
            formatsVector.push_back(stringToValue<IOFormat>(f));
        }
    };

    getFormats(inputFormats, "--inputIOFormats");
    getFormats(outputFormats, "--outputIOFormats");

    bool addedExplicitBatchFlag{false};
    getAndDelOption(arguments, "--explicitBatch", addedExplicitBatchFlag);
    if (addedExplicitBatchFlag)
    {
        sample::gLogWarning << "--explicitBatch flag has been deprecated and has no effect!" << std::endl;
        sample::gLogWarning << "Explicit batch dim is automatically enabled if input model is ONNX or if dynamic "
                            << "shapes are provided when the engine is built." << std::endl;
    }

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

    bool addedExplicitPrecisionFlag{false};
    getAndDelOption(arguments, "--explicitPrecision", addedExplicitPrecisionFlag);
    if (addedExplicitPrecisionFlag)
    {
        sample::gLogWarning << "--explicitPrecision flag has been deprecated and has no effect!" << std::endl;
    }

    if (getAndDelOption(arguments, "--workspace", workspace))
    {
        sample::gLogWarning << "--workspace flag has been deprecated by --memPoolSize flag." << std::endl;
    }

    std::string memPoolSizes;
    getAndDelOption(arguments, "--memPoolSize", memPoolSizes);
    std::vector<std::string> memPoolSpecs{splitToStringVec(memPoolSizes, ',')};
    for (auto const& memPoolSpec : memPoolSpecs)
    {
        std::string memPoolName;
        double memPoolSize;
        std::tie(memPoolName, memPoolSize) = splitNameAndValue<double>(memPoolSpec);
        if (memPoolSize < 0)
        {
            throw std::invalid_argument(std::string("Negative memory pool size: ") + std::to_string(memPoolSize));
        }
        if (memPoolName == "workspace")
        {
            workspace = memPoolSize;
        }
        else if (memPoolName == "dlaSRAM")
        {
            dlaSRAM = memPoolSize;
        }
        else if (memPoolName == "dlaLocalDRAM")
        {
            dlaLocalDRAM = memPoolSize;
        }
        else if (memPoolName == "dlaGlobalDRAM")
        {
            dlaGlobalDRAM = memPoolSize;
        }
        else if (!memPoolName.empty())
        {
            throw std::invalid_argument(std::string("Unknown memory pool: ") + memPoolName);
        }
    }

    getAndDelOption(arguments, "--maxBatch", maxBatch);
    getAndDelOption(arguments, "--minTiming", minTiming);
    getAndDelOption(arguments, "--avgTiming", avgTiming);

    bool best{false};
    getAndDelOption(arguments, "--best", best);
    if (best)
    {
        int8 = true;
        fp16 = true;
    }

    getAndDelOption(arguments, "--refit", refittable);

    // --vc and --versionCompatible are synonyms
    getAndDelOption(arguments, "--vc", versionCompatible);
    if (!versionCompatible)
    {
        getAndDelOption(arguments, "--versionCompatible", versionCompatible);
    }

    getAndDelOption(arguments, "--excludeLeanRuntime", excludeLeanRuntime);

    getAndDelNegOption(arguments, "--noTF32", tf32);
    getAndDelOption(arguments, "--fp16", fp16);
    getAndDelOption(arguments, "--int8", int8);
    getAndDelOption(arguments, "--fp8", fp8);
    if (fp8 && int8)
    {
        throw std::invalid_argument("Invalid usage, fp8 and int8 aren't allowed to be enabled together.");
    }
    getAndDelOption(arguments, "--safe", safe);
    getAndDelOption(arguments, "--buildDLAStandalone", buildDLAStandalone);
    getAndDelOption(arguments, "--allowGPUFallback", allowGPUFallback);
    getAndDelOption(arguments, "--consistency", consistency);
    getAndDelOption(arguments, "--restricted", restricted);
    if (getAndDelOption(arguments, "--buildOnly", skipInference))
    {
        sample::gLogWarning << "--buildOnly flag has been deprecated by --skipInference flag." << std::endl;
    }
    getAndDelOption(arguments, "--skipInference", skipInference);
    getAndDelOption(arguments, "--directIO", directIO);

    std::string precisionConstraintsString;
    getAndDelOption(arguments, "--precisionConstraints", precisionConstraintsString);
    if (!precisionConstraintsString.empty())
    {
        const std::unordered_map<std::string, PrecisionConstraints> precisionConstraintsMap
            = {{"obey", PrecisionConstraints::kOBEY}, {"prefer", PrecisionConstraints::kPREFER},
                {"none", PrecisionConstraints::kNONE}};
        auto it = precisionConstraintsMap.find(precisionConstraintsString);
        if (it == precisionConstraintsMap.end())
        {
            throw std::invalid_argument(std::string("Unknown precision constraints: ") + precisionConstraintsString);
        }
        precisionConstraints = it->second;
    }
    else
    {
        precisionConstraints = PrecisionConstraints::kNONE;
    }

    getLayerPrecisions(arguments, "--layerPrecisions", layerPrecisions);
    getLayerOutputTypes(arguments, "--layerOutputTypes", layerOutputTypes);
    getLayerDeviceTypes(arguments, "--layerDeviceTypes", layerDeviceTypes);

    if (layerPrecisions.empty() && layerOutputTypes.empty() && precisionConstraints != PrecisionConstraints::kNONE)
    {
        sample::gLogWarning << R"(When --precisionConstraints flag is set to "obey" or "prefer", please add )"
                            << "--layerPrecision/--layerOutputTypes flags to set layer-wise precisions and output "
                            << "types." << std::endl;
    }
    else if ((!layerPrecisions.empty() || !layerOutputTypes.empty())
        && precisionConstraints == PrecisionConstraints::kNONE)
    {
        sample::gLogWarning << "--layerPrecision/--layerOutputTypes flags have no effect when --precisionConstraints "
                            << R"(flag is set to "none".)" << std::endl;
    }

    getAndDelOption(arguments, "--sparsity", sparsity);

    bool calibCheck = getAndDelOption(arguments, "--calib", calibration);
    if (int8 && calibCheck && !shapes.empty() && shapesCalib.empty())
    {
        shapesCalib = shapes;
    }

    std::string profilingVerbosityString;
    if (getAndDelOption(arguments, "--nvtxMode", profilingVerbosityString))
    {
        sample::gLogWarning << "--nvtxMode flag has been deprecated by --profilingVerbosity flag." << std::endl;
    }

    getAndDelOption(arguments, "--profilingVerbosity", profilingVerbosityString);
    if (profilingVerbosityString == "layer_names_only")
    {
        profilingVerbosity = nvinfer1::ProfilingVerbosity::kLAYER_NAMES_ONLY;
    }
    else if (profilingVerbosityString == "none")
    {
        profilingVerbosity = nvinfer1::ProfilingVerbosity::kNONE;
    }
    else if (profilingVerbosityString == "detailed")
    {
        profilingVerbosity = nvinfer1::ProfilingVerbosity::kDETAILED;
    }
    else if (profilingVerbosityString == "default")
    {
        sample::gLogWarning << "--profilingVerbosity=default has been deprecated by "
                               "--profilingVerbosity=layer_names_only."
                            << std::endl;
        profilingVerbosity = nvinfer1::ProfilingVerbosity::kLAYER_NAMES_ONLY;
    }
    else if (profilingVerbosityString == "verbose")
    {
        sample::gLogWarning << "--profilingVerbosity=verbose has been deprecated by --profilingVerbosity=detailed."
                            << std::endl;
        profilingVerbosity = nvinfer1::ProfilingVerbosity::kDETAILED;
    }
    else if (!profilingVerbosityString.empty())
    {
        throw std::invalid_argument(std::string("Unknown profilingVerbosity: ") + profilingVerbosityString);
    }

    if (getAndDelOption(arguments, "--loadEngine", engine))
    {
        load = true;
    }
    if (getAndDelOption(arguments, "--saveEngine", engine))
    {
        save = true;
    }
    if (load && save)
    {
        throw std::invalid_argument("Incompatible load and save engine options selected");
    }

    std::string tacticSourceArgs;
    if (getAndDelOption(arguments, "--tacticSources", tacticSourceArgs))
    {
        std::vector<std::string> tacticList = splitToStringVec(tacticSourceArgs, ',');
        for (auto& t : tacticList)
        {
            bool enable{false};
            if (t.front() == '+')
            {
                enable = true;
            }
            else if (t.front() != '-')
            {
                throw std::invalid_argument(
                    "Tactic source must be prefixed with + or -, indicating whether it should be enabled or disabled "
                    "respectively.");
            }
            t.erase(0, 1);

            const auto toUpper = [](std::string& sourceName) {
                std::transform(
                    sourceName.begin(), sourceName.end(), sourceName.begin(), [](char c) { return std::toupper(c); });
                return sourceName;
            };

            nvinfer1::TacticSource source{};
            t = toUpper(t);
            if (t == "CUBLAS")
            {
                source = nvinfer1::TacticSource::kCUBLAS;
            }
            else if (t == "CUBLASLT" || t == "CUBLAS_LT")
            {
                source = nvinfer1::TacticSource::kCUBLAS_LT;
            }
            else if (t == "CUDNN")
            {
                source = nvinfer1::TacticSource::kCUDNN;
            }
            else if (t == "EDGE_MASK_CONVOLUTIONS")
            {
                source = nvinfer1::TacticSource::kEDGE_MASK_CONVOLUTIONS;
            }
            else if (t == "JIT_CONVOLUTIONS")
            {
                source = nvinfer1::TacticSource::kJIT_CONVOLUTIONS;
            }
            else
            {
                throw std::invalid_argument(std::string("Unknown tactic source: ") + t);
            }

            uint32_t sourceBit = 1U << static_cast<uint32_t>(source);

            if (enable)
            {
                enabledTactics |= sourceBit;
            }
            else
            {
                disabledTactics |= sourceBit;
            }

            if (enabledTactics & disabledTactics)
            {
                throw std::invalid_argument(std::string("Cannot enable and disable ") + t);
            }
        }
    }

    bool noBuilderCache{false};
    getAndDelOption(arguments, "--noBuilderCache", noBuilderCache);
    getAndDelOption(arguments, "--timingCacheFile", timingCacheFile);
    if (noBuilderCache)
    {
        timingCacheMode = TimingCacheMode::kDISABLE;
    }
    else if (!timingCacheFile.empty())
    {
        timingCacheMode = TimingCacheMode::kGLOBAL;
    }
    else
    {
        timingCacheMode = TimingCacheMode::kLOCAL;
    }
    if (getAndDelOption(arguments, "--heuristic", heuristic))
    {
        sample::gLogWarning << "--heuristic flag has been deprecated, use --builderOptimizationLevel=<N> flag instead "
                               "(N <= 2 enables heuristic)."
                            << std::endl;
    }
    getAndDelOption(arguments, "--builderOptimizationLevel", builderOptimizationLevel);

    std::string hardwareCompatibleArgs;
    getAndDelOption(arguments, "--hardwareCompatibilityLevel", hardwareCompatibleArgs);
    if (hardwareCompatibleArgs == "none" || hardwareCompatibleArgs.empty())
    {
        hardwareCompatibilityLevel = HardwareCompatibilityLevel::kNONE;
    }
    else if (samplesCommon::toLower(hardwareCompatibleArgs) == "ampere+")
    {
        hardwareCompatibilityLevel = HardwareCompatibilityLevel::kAMPERE_PLUS;
    }
    else
    {
        throw std::invalid_argument(std::string("Unknown hardwareCompatibilityLevel: ") + hardwareCompatibleArgs
            + ". Valid options: none, ampere+.");
    }

    getAndDelOption(arguments, "--maxAuxStreams", maxAuxStreams);

    std::string previewFeaturesBuf;
    getAndDelOption(arguments, "--preview", previewFeaturesBuf);
    std::vector<std::string> previewFeaturesVec{splitToStringVec(previewFeaturesBuf, ',')};
    for (auto featureName : previewFeaturesVec)
    {
        bool enable{false};
        if (featureName.front() == '+')
        {
            enable = true;
        }
        else if (featureName.front() != '-')
        {
            throw std::invalid_argument(
                "Preview features must be prefixed with + or -, indicating whether it should be enabled or disabled "
                "respectively.");
        }
        featureName.erase(0, 1);

        PreviewFeature feat{};
        if (featureName == "profileSharing0806")
        {
            feat = PreviewFeature::kPROFILE_SHARING_0806;
        }
        else if (featureName == "fasterDynamicShapes0805")
        {
            feat = PreviewFeature::kFASTER_DYNAMIC_SHAPES_0805;
        }
        else if (featureName == "disableExternalTacticSourcesForCore0805")
        {
            feat = PreviewFeature::kDISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805;
        }
        else
        {
            throw std::invalid_argument(std::string("Unknown preview feature: ") + featureName);
        }
        previewFeatures[static_cast<int32_t>(feat)] = enable;
    }

    getAndDelOption(arguments, "--tempdir", tempdir);
    getTempfileControls(arguments, "--tempfileControls", tempfileControls);

    std::string runtimeMode;
    getAndDelOption(arguments, "--useRuntime", runtimeMode);
    if (runtimeMode == "full")
    {
        useRuntime = RuntimeMode::kFULL;
    }
    else if (runtimeMode == "dispatch")
    {
        useRuntime = RuntimeMode::kDISPATCH;
    }
    else if (runtimeMode == "lean")
    {
        useRuntime = RuntimeMode::kLEAN;
    }
    else if (!runtimeMode.empty())
    {
        throw std::invalid_argument(std::string("Unknown useRuntime: ") + runtimeMode);
    }

    if ((useRuntime == RuntimeMode::kDISPATCH || useRuntime == RuntimeMode::kLEAN) && !versionCompatible)
    {
        versionCompatible = true;
        sample::gLogWarning << "Implicitly enabling --versionCompatible since --useRuntime=" << runtimeMode
                            << " is set." << std::endl;
    }

    if (useRuntime != RuntimeMode::kFULL && !load)
    {
        throw std::invalid_argument(std::string("Building a TensorRT engine requires --useRuntime=full."));
    }

    getAndDelOption(arguments, "--leanDLLPath", leanDLLPath);
}

void SystemOptions::parse(Arguments& arguments)
{
    getAndDelOption(arguments, "--device", device);
    getAndDelOption(arguments, "--useDLACore", DLACore);
    std::string pluginName;
    while (getAndDelOption(arguments, "--plugins", pluginName))
    {
        sample::gLogWarning << "--plugins flag has been deprecated, use --staticPlugins flag instead." << std::endl;
        plugins.emplace_back(pluginName);
    }
    while (getAndDelOption(arguments, "--staticPlugins", pluginName))
    {
        plugins.emplace_back(pluginName);
    }
    while (getAndDelOption(arguments, "--setPluginsToSerialize", pluginName))
    {
        setPluginsToSerialize.emplace_back(pluginName);
    }
    while (getAndDelOption(arguments, "--dynamicPlugins", pluginName))
    {
        dynamicPlugins.emplace_back(pluginName);
    }
    getAndDelOption(arguments, "--ignoreParsedPluginLibs", ignoreParsedPluginLibs);
}

void InferenceOptions::parse(Arguments& arguments)
{

    if (getAndDelOption(arguments, "--streams", infStreams))
    {
        sample::gLogWarning << "--streams flag has been deprecated, use --infStreams flag instead." << std::endl;
    }
    getAndDelOption(arguments, "--infStreams", infStreams);

    getAndDelOption(arguments, "--iterations", iterations);
    getAndDelOption(arguments, "--duration", duration);
    getAndDelOption(arguments, "--warmUp", warmup);
    getAndDelOption(arguments, "--sleepTime", sleep);
    getAndDelOption(arguments, "--idleTime", idle);
    bool exposeDMA{false};
    if (getAndDelOption(arguments, "--exposeDMA", exposeDMA))
    {
        overlap = !exposeDMA;
    }
    getAndDelOption(arguments, "--noDataTransfers", skipTransfers);
    getAndDelOption(arguments, "--useManagedMemory", useManaged);
    getAndDelOption(arguments, "--useSpinWait", spin);
    getAndDelOption(arguments, "--threads", threads);
    getAndDelOption(arguments, "--useCudaGraph", graph);
    getAndDelOption(arguments, "--separateProfileRun", rerun);
    getAndDelOption(arguments, "--timeDeserialize", timeDeserialize);
    getAndDelOption(arguments, "--timeRefit", timeRefit);
    getAndDelOption(arguments, "--persistentCacheRatio", persistentCacheRatio);

    std::string list;
    getAndDelOption(arguments, "--loadInputs", list);
    std::vector<std::string> inputsList{splitToStringVec(list, ',')};
    splitInsertKeyValue(inputsList, inputs);

    getShapesInference(arguments, shapes, "--shapes");
    getAndDelOption(arguments, "--batch", batch);
}

void ReportingOptions::parse(Arguments& arguments)
{
    getAndDelOption(arguments, "--avgRuns", avgs);
    getAndDelOption(arguments, "--verbose", verbose);
    getAndDelOption(arguments, "--dumpRefit", refit);
    getAndDelOption(arguments, "--dumpOutput", output);
    getAndDelOption(arguments, "--dumpRawBindingsToFile", dumpRawBindings);
    getAndDelOption(arguments, "--dumpProfile", profile);
    getAndDelOption(arguments, "--dumpLayerInfo", layerInfo);
    getAndDelOption(arguments, "--exportTimes", exportTimes);
    getAndDelOption(arguments, "--exportOutput", exportOutput);
    getAndDelOption(arguments, "--exportProfile", exportProfile);
    getAndDelOption(arguments, "--exportLayerInfo", exportLayerInfo);

    std::string percentileString;
    getAndDelOption(arguments, "--percentile", percentileString);
    std::vector<std::string> percentileStrings = splitToStringVec(percentileString, ',');
    if (!percentileStrings.empty())
    {
        percentiles.clear();
    }
    for (const auto& p : percentileStrings)
    {
        percentiles.push_back(stringToValue<float>(p));
    }

    for (auto percentile : percentiles)
    {
        if (percentile < 0.F || percentile > 100.F)
        {
            throw std::invalid_argument(std::string("Percentile ") + std::to_string(percentile) + "is not in [0,100]");
        }
    }
}

bool parseHelp(Arguments& arguments)
{
    bool helpLong{false};
    bool helpShort{false};
    getAndDelOption(arguments, "--help", helpLong);
    getAndDelOption(arguments, "-h", helpShort);
    return helpLong || helpShort;
}

void AllOptions::parse(Arguments& arguments)
{
    model.parse(arguments);
    build.parse(arguments);
    system.parse(arguments);
    inference.parse(arguments);

    // Use explicitBatch when input model is ONNX or when dynamic shapes are used.
    const bool isOnnx{model.baseModel.format == ModelFormat::kONNX};
    const bool hasDynamicShapes{!build.shapes.empty() || !inference.shapes.empty()};
    const bool detectedExplicitBatch = isOnnx || hasDynamicShapes;

    // Throw an error if user tries to use --batch or --maxBatch when the engine has explicit batch dim.
    const bool maxBatchWasSet{build.maxBatch != maxBatchNotProvided};
    const bool batchWasSet{inference.batch != batchNotProvided};
    if (detectedExplicitBatch && (maxBatchWasSet || batchWasSet))
    {
        throw std::invalid_argument(
            "The --batch and --maxBatch flags should not be used when the input model is ONNX or when dynamic shapes "
            "are provided. Please use --optShapes and --shapes to set input shapes instead.");
    }

    if (build.useRuntime != RuntimeMode::kFULL && inference.timeRefit)
    {
        throw std::invalid_argument("--timeRefit requires --useRuntime=full.");
    }

    // If batch and/or maxBatch is not set and the engine has implicit batch dim, set them to default values.
    if (!detectedExplicitBatch)
    {
        // If batch is not set, set it to default value.
        if (!batchWasSet)
        {
            inference.batch = defaultBatch;
        }
        // If maxBatch is not set, set it to be equal to batch.
        if (!maxBatchWasSet)
        {
            build.maxBatch = inference.batch;
        }
        // MaxBatch should not be less than batch.
        if (build.maxBatch < inference.batch)
        {
            throw std::invalid_argument("Build max batch " + std::to_string(build.maxBatch)
                + " is less than inference batch " + std::to_string(inference.batch));
        }
    }

    // Propagate shape profile between builder and inference
    for (auto const& s : build.shapes)
    {
        if (inference.shapes.find(s.first) == inference.shapes.end())
        {
            insertShapesInference(
                inference.shapes, s.first, s.second[static_cast<size_t>(nvinfer1::OptProfileSelector::kOPT)]);
        }
    }
    for (auto const& s : inference.shapes)
    {
        if (build.shapes.find(s.first) == build.shapes.end())
        {
            // assume min/opt/max all the same
            insertShapesBuild(build.shapes, nvinfer1::OptProfileSelector::kMIN, s.first, s.second);
            insertShapesBuild(build.shapes, nvinfer1::OptProfileSelector::kOPT, s.first, s.second);
            insertShapesBuild(build.shapes, nvinfer1::OptProfileSelector::kMAX, s.first, s.second);
        }
    }

    // Set nvtxVerbosity to be the same as build-time profilingVerbosity.
    inference.nvtxVerbosity = build.profilingVerbosity;

    reporting.parse(arguments);
    helps = parseHelp(arguments);

    if (!helps)
    {
        if (!build.load && model.baseModel.format == ModelFormat::kANY)
        {
            throw std::invalid_argument("Model missing or format not recognized");
        }
        if (build.safe && system.DLACore >= 0)
        {
            build.buildDLAStandalone = true;
        }
        if (build.buildDLAStandalone)
        {
            build.skipInference = true;
            auto checkSafeDLAFormats = [](std::vector<IOFormat> const& fmt, bool isInput) {
                return fmt.empty() ? false : std::all_of(fmt.begin(), fmt.end(), [&](IOFormat const& pair) {
                    bool supported{false};
                    bool const isDLA_LINEAR{
                        pair.second == 1U << static_cast<int32_t>(nvinfer1::TensorFormat::kDLA_LINEAR)};
                    bool const isHWC4{pair.second == 1U << static_cast<int32_t>(nvinfer1::TensorFormat::kCHW4)
                        || pair.second == 1U << static_cast<int32_t>(nvinfer1::TensorFormat::kDLA_HWC4)};
                    bool const isCHW32{pair.second == 1U << static_cast<int32_t>(nvinfer1::TensorFormat::kCHW32)};
                    bool const isCHW16{pair.second == 1U << static_cast<int32_t>(nvinfer1::TensorFormat::kCHW16)};
                    supported |= pair.first == nvinfer1::DataType::kINT8
                        && (isDLA_LINEAR || (isInput ? isHWC4 : false) || isCHW32);
                    supported |= pair.first == nvinfer1::DataType::kHALF
                        && (isDLA_LINEAR || (isInput ? isHWC4 : false) || isCHW16);
                    return supported;
                });
            };
            if (!checkSafeDLAFormats(build.inputFormats, true) || !checkSafeDLAFormats(build.outputFormats, false))
            {
                throw std::invalid_argument(
                    "I/O formats for safe DLA capability are restricted to fp16/int8:dla_linear, fp16/int8:hwc4, "
                    "fp16:chw16 or "
                    "int8:chw32");
            }
            if (build.allowGPUFallback)
            {
                throw std::invalid_argument("GPU fallback (--allowGPUFallback) not allowed for DLA standalone mode");
            }
        }
    }
}

void TaskInferenceOptions::parse(Arguments& arguments)
{
    getAndDelOption(arguments, "engine", engine);
    getAndDelOption(arguments, "device", device);
    getAndDelOption(arguments, "batch", batch);
    getAndDelOption(arguments, "DLACore", DLACore);
    getAndDelOption(arguments, "graph", graph);
    getAndDelOption(arguments, "persistentCacheRatio", persistentCacheRatio);
}

void SafeBuilderOptions::parse(Arguments& arguments)
{
    auto getFormats = [&arguments](std::vector<IOFormat>& formatsVector, const char* argument) {
        std::string list;
        getAndDelOption(arguments, argument, list);
        std::vector<std::string> formats{splitToStringVec(list, ',')};
        for (const auto& f : formats)
        {
            formatsVector.push_back(stringToValue<IOFormat>(f));
        }
    };

    getAndDelOption(arguments, "--serialized", serialized);
    getAndDelOption(arguments, "--onnx", onnxModelFile);
    getAndDelOption(arguments, "--help", help);
    getAndDelOption(arguments, "-h", help);
    getAndDelOption(arguments, "--verbose", verbose);
    getAndDelOption(arguments, "-v", verbose);
    getFormats(inputFormats, "--inputIOFormats");
    getFormats(outputFormats, "--outputIOFormats");
    getAndDelOption(arguments, "--int8", int8);
    getAndDelOption(arguments, "--calib", calibFile);
    getAndDelOption(arguments, "--consistency", consistency);
    getAndDelOption(arguments, "--std", standard);
    std::string pluginName;
    while (getAndDelOption(arguments, "--plugins", pluginName))
    {
        sample::gLogWarning << "--plugins flag has been deprecated, use --staticPlugins flag instead." << std::endl;
        plugins.emplace_back(pluginName);
    }
    while (getAndDelOption(arguments, "--staticPlugins", pluginName))
    {
        plugins.emplace_back(pluginName);
    }
    bool noBuilderCache{false};
    getAndDelOption(arguments, "--noBuilderCache", noBuilderCache);
    getAndDelOption(arguments, "--timingCacheFile", timingCacheFile);
    getAndDelOption(arguments, "--minTiming", minTiming);
    getAndDelOption(arguments, "--avgTiming", avgTiming);
    if (noBuilderCache)
    {
        timingCacheMode = TimingCacheMode::kDISABLE;
    }
    else if (!timingCacheFile.empty())
    {
        timingCacheMode = TimingCacheMode::kGLOBAL;
    }
    else
    {
        timingCacheMode = TimingCacheMode::kLOCAL;
    }
    getAndDelOption(arguments, "--sparsity", sparsity);
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

std::ostream& operator<<(std::ostream& os, nvinfer1::DataType dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
    {
        os << "fp32";
        break;
    }
    case nvinfer1::DataType::kHALF:
    {
        os << "fp16";
        break;
    }
    case nvinfer1::DataType::kINT8:
    {
        os << "int8";
        break;
    }
    case nvinfer1::DataType::kINT32:
    {
        os << "int32";
        break;
    }
    case nvinfer1::DataType::kBOOL:
    {
        os << "bool";
        break;
    }
    case nvinfer1::DataType::kUINT8:
    {
        os << "uint8";
        break;
    }
    case nvinfer1::DataType::kFP8:
    {
        os << "fp8";
        break;
    }
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, IOFormat const& format)
{
    os << format.first << ":";

    for (int32_t f = 0; f < nvinfer1::EnumMax<nvinfer1::TensorFormat>(); ++f)
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
            case nvinfer1::TensorFormat::kHWC16:
            {
                os << "hwc16";
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
            case nvinfer1::TensorFormat::kDHWC8:
            {
                os << "dhwc8";
                break;
            }
            case nvinfer1::TensorFormat::kCDHW32:
            {
                os << "cdhw32";
                break;
            }
            case nvinfer1::TensorFormat::kHWC:
            {
                os << "hwc";
                break;
            }
            case nvinfer1::TensorFormat::kDHWC:
            {
                os << "dhwc";
                break;
            }
            case nvinfer1::TensorFormat::kDLA_LINEAR:
            {
                os << "dla_linear";
                break;
            }
            case nvinfer1::TensorFormat::kDLA_HWC4:
            {
                os << "dla_hwc4";
                break;
            }
            }
        }
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, nvinfer1::DeviceType devType)
{
    switch (devType)
    {
    case nvinfer1::DeviceType::kGPU:
    {
        os << "GPU";
        break;
    }
    case nvinfer1::DeviceType::kDLA:
    {
        os << "DLA";
        break;
    }
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const ShapeRange& dims)
{
    int32_t i = 0;
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

std::ostream& operator<<(std::ostream& os, LayerPrecisions const& layerPrecisions)
{
    int32_t i = 0;
    for (auto const& layerPrecision : layerPrecisions)
    {
        os << (i ? "," : "") << layerPrecision.first << ":" << layerPrecision.second;
        ++i;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, LayerDeviceTypes const& layerDeviceTypes)
{
    int32_t i = 0;
    for (auto const& layerDevicePair : layerDeviceTypes)
    {
        os << (i++ ? ", " : "") << layerDevicePair.first << ":" << layerDevicePair.second;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const BuildOptions& options)
{
    // clang-format off
    os << "=== Build Options ==="                                                                                       << std::endl <<
          "Max batch: ";        printBatch(os, options.maxBatch)                                                        << std::endl <<
          "Memory Pools: ";     printMemoryPools(os, options)                                                           << std::endl <<
          "minTiming: "      << options.minTiming                                                                       << std::endl <<
          "avgTiming: "      << options.avgTiming                                                                       << std::endl <<
          "Precision: ";        printPrecision(os, options)                                                             << std::endl <<
          "LayerPrecisions: " << options.layerPrecisions                                                                << std::endl <<
          "Layer Device Types: " << options.layerDeviceTypes                                                            << std::endl <<
          "Calibration: "    << (options.int8 && options.calibration.empty() ? "Dynamic" : options.calibration.c_str()) << std::endl <<
          "Refit: "          << boolToEnabled(options.refittable)                                                       << std::endl <<
          "Version Compatible: " << boolToEnabled(options.versionCompatible)                                            << std::endl <<
          "TensorRT runtime: " << options.useRuntime                                                                    << std::endl <<
          "Lean DLL Path: " << options.leanDLLPath                                                                      << std::endl <<
          "Tempfile Controls: "; printTempfileControls(os, options.tempfileControls)                                    << std::endl <<
          "Exclude Lean Runtime: " << boolToEnabled(options.excludeLeanRuntime)                                         << std::endl <<
          "Sparsity: ";         printSparsity(os, options)                                                              << std::endl <<
          "Safe mode: "      << boolToEnabled(options.safe)                                                             << std::endl <<
          "Build DLA standalone loadable: " << boolToEnabled(options.buildDLAStandalone)                                << std::endl <<
          "Allow GPU fallback for DLA: " << boolToEnabled(options.allowGPUFallback)                                     << std::endl <<
          "DirectIO mode: "  << boolToEnabled(options.directIO)                                                         << std::endl <<
          "Restricted mode: " << boolToEnabled(options.restricted)                                                      << std::endl <<
          "Skip inference: "     << boolToEnabled(options.skipInference)                                                << std::endl <<
          "Save engine: "    << (options.save ? options.engine : "")                                                    << std::endl <<
          "Load engine: "    << (options.load ? options.engine : "")                                                    << std::endl <<
          "Profiling verbosity: " << static_cast<int32_t>(options.profilingVerbosity)                                   << std::endl <<
          "Tactic sources: ";   printTacticSources(os, options.enabledTactics, options.disabledTactics)                 << std::endl <<
          "timingCacheMode: ";  printTimingCache(os, options.timingCacheMode)                                           << std::endl <<
          "timingCacheFile: " << options.timingCacheFile                                                                << std::endl <<
          "Heuristic: "       << boolToEnabled(options.heuristic)                                                       << std::endl <<
          "Preview Features: "; printPreviewFlags(os, options)                                                          << std::endl <<
          "MaxAuxStreams: "   << options.maxAuxStreams                                                                  << std::endl <<
          "BuilderOptimizationLevel: " << options.builderOptimizationLevel                                              << std::endl;
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

    printIOFormats(os, "Input(s)", options.inputFormats);
    printIOFormats(os, "Output(s)", options.outputFormats);
    printShapes(os, "build", options.shapes);
    printShapes(os, "calibration", options.shapesCalib);

    return os;
}

std::ostream& operator<<(std::ostream& os, const SystemOptions& options)
{
    // clang-format off
    os << "=== System Options ==="                                                                << std::endl <<

          "Device: "  << options.device                                                           << std::endl <<
          "DLACore: " << (options.DLACore != -1 ? std::to_string(options.DLACore) : "")           << std::endl;
    os << "Plugins:";

    for (const auto& p : options.plugins)
    {
        os << " " << p;
    }
    os << std::endl;

    os << "setPluginsToSerialize:";

    for (const auto& p : options.setPluginsToSerialize)
    {
        os << " " << p;
    }
    os << std::endl;

    os << "dynamicPlugins:";

    for (const auto& p : options.dynamicPlugins)
    {
        os << " " << p;
    }
    os << std::endl;

    os << "ignoreParsedPluginLibs: " << options.ignoreParsedPluginLibs << std::endl;
    os << std::endl;

    return os;
    // clang-format on
}

std::ostream& operator<<(std::ostream& os, const InferenceOptions& options)
{
    // clang-format off
    os << "=== Inference Options ==="                                     << std::endl <<

          "Batch: ";
    if (options.batch && options.shapes.empty())
    {
                          os << options.batch                             << std::endl;
    }
    else
    {
                          os << "Explicit"                                << std::endl;
    }
    printShapes(os, "inference", options.shapes);
    os << "Iterations: "                << options.iterations                                   << std::endl <<
          "Duration: "                  << options.duration   << "s (+ "
                                        << options.warmup     << "ms warm up)"                  << std::endl <<
          "Sleep time: "                << options.sleep      << "ms"                           << std::endl <<
          "Idle time: "                 << options.idle       << "ms"                           << std::endl <<
          "Inference Streams: "         << options.infStreams                                   << std::endl <<
          "ExposeDMA: "                 << boolToEnabled(!options.overlap)                      << std::endl <<
          "Data transfers: "            << boolToEnabled(!options.skipTransfers)                << std::endl <<
          "Spin-wait: "                 << boolToEnabled(options.spin)                          << std::endl <<
          "Multithreading: "            << boolToEnabled(options.threads)                       << std::endl <<
          "CUDA Graph: "                << boolToEnabled(options.graph)                         << std::endl <<
          "Separate profiling: "        << boolToEnabled(options.rerun)                         << std::endl <<
          "Time Deserialize: "          << boolToEnabled(options.timeDeserialize)               << std::endl <<
          "Time Refit: "                << boolToEnabled(options.timeRefit)                     << std::endl <<
          "NVTX verbosity: "            << static_cast<int32_t>(options.nvtxVerbosity)          << std::endl <<
          "Persistent Cache Ratio: "    << static_cast<float>(options.persistentCacheRatio)   << std::endl;
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
    os << "=== Reporting Options ==="                                                     << std::endl <<
          "Verbose: "                     << boolToEnabled(options.verbose)               << std::endl <<
          "Averages: "                    << options.avgs << " inferences"                << std::endl <<
          "Percentiles: "                 << joinValuesToString(options.percentiles, ",") << std::endl <<
          "Dump refittable layers:"       << boolToEnabled(options.refit)                 << std::endl <<
          "Dump output: "                 << boolToEnabled(options.output)                << std::endl <<
          "Profile: "                     << boolToEnabled(options.profile)               << std::endl <<
          "Export timing to JSON file: "  << options.exportTimes                          << std::endl <<
          "Export output to JSON file: "  << options.exportOutput                         << std::endl <<
          "Export profile to JSON file: " << options.exportProfile                        << std::endl;
    // clang-format on

    return os;
}

std::ostream& operator<<(std::ostream& os, const AllOptions& options)
{
    os << options.model << options.build << options.system << options.inference << options.reporting << std::endl;
    return os;
}

std::ostream& operator<<(std::ostream& os, const SafeBuilderOptions& options)
{
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

    os << "=== Build Options ===" << std::endl;
    os << "Model ONNX: " << options.onnxModelFile << std::endl;

    os << "Precision: FP16";
    if (options.int8)
    {
        os << " + INT8";
    }
    if (options.fp8)
    {
        os << " + FP8";
    }
    os << std::endl;
    os << "Calibration file: " << options.calibFile << std::endl;
    os << "Serialized Network: " << options.serialized << std::endl;

    printIOFormats(os, "Input(s)", options.inputFormats);
    printIOFormats(os, "Output(s)", options.outputFormats);

    os << "Plugins:";
    for (const auto& p : options.plugins)
    {
        os << " " << p;
    }

    os << "timingCacheMode: ";
    printTimingCache(os, options.timingCacheMode) << std::endl;
    os << "timingCacheFile: " << options.timingCacheFile << std::endl;
    os << std::endl;
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
    os << "=== Build Options ==="                                                                                                                   "\n"
          "  --maxBatch                         Set max batch size and build an implicit batch engine (default = same size as --batch)"             "\n"
          "                                     This option should not be used when the input model is ONNX or when dynamic shapes are provided."   "\n"
          "  --minShapes=spec                   Build with dynamic shapes using a profile with the min shapes provided"                             "\n"
          "  --optShapes=spec                   Build with dynamic shapes using a profile with the opt shapes provided"                             "\n"
          "  --maxShapes=spec                   Build with dynamic shapes using a profile with the max shapes provided"                             "\n"
          "  --minShapesCalib=spec              Calibrate with dynamic shapes using a profile with the min shapes provided"                         "\n"
          "  --optShapesCalib=spec              Calibrate with dynamic shapes using a profile with the opt shapes provided"                         "\n"
          "  --maxShapesCalib=spec              Calibrate with dynamic shapes using a profile with the max shapes provided"                         "\n"
          "                                     Note: All three of min, opt and max shapes must be supplied."                                       "\n"
          "                                           However, if only opt shapes is supplied then it will be expanded so"                          "\n"
          "                                           that min shapes and max shapes are set to the same values as opt shapes."                     "\n"
          "                                           Input names can be wrapped with escaped single quotes (ex: 'Input:0')."                       "\n"
          "                                     Example input shapes spec: input0:1x3x256x256,input1:1x3x128x128"                                   "\n"
          "                                     Each input shape is supplied as a key-value pair where key is the input name and"                   "\n"
          "                                     value is the dimensions (including the batch dimension) to be used for that input."                 "\n"
          "                                     Each key-value pair has the key and value separated using a colon (:)."                             "\n"
          "                                     Multiple input shapes can be provided via comma-separated key-value pairs."                         "\n"
          "  --inputIOFormats=spec              Type and format of each of the input tensors (default = all inputs in fp32:chw)"                    "\n"
          "                                     See --outputIOFormats help for the grammar of type and format list."                                "\n"
          "                                     Note: If this option is specified, please set comma-separated types and formats for all"            "\n"
          "                                           inputs following the same order as network inputs ID (even if only one input"                 "\n"
          "                                           needs specifying IO format) or set the type and format once for broadcasting."                "\n"
          "  --outputIOFormats=spec             Type and format of each of the output tensors (default = all outputs in fp32:chw)"                  "\n"
          "                                     Note: If this option is specified, please set comma-separated types and formats for all"            "\n"
          "                                           outputs following the same order as network outputs ID (even if only one output"              "\n"
          "                                           needs specifying IO format) or set the type and format once for broadcasting."                "\n"
          R"(                                     IO Formats: spec  ::= IOfmt[","spec])"                                                            "\n"
          "                                                 IOfmt ::= type:fmt"                                                                     "\n"
          R"(                                               type  ::= "fp32"|"fp16"|"int32"|"int8")"                                                "\n"
          R"(                                               fmt   ::= ("chw"|"chw2"|"chw4"|"hwc8"|"chw16"|"chw32"|"dhwc8"|)"                        "\n"
          R"(                                                          "cdhw32"|"hwc"|"dla_linear"|"dla_hwc4")["+"fmt])"                            "\n"
          "  --workspace=N                      Set workspace size in MiB."                                                                         "\n"
          "  --memPoolSize=poolspec             Specify the size constraints of the designated memory pool(s) in MiB."                              "\n"
          "                                     Note: Also accepts decimal sizes, e.g. 0.25MiB. Will be rounded down to the nearest integer bytes." "\n"
          "                                     In particular, for dlaSRAM the bytes will be rounded down to the nearest power of 2."               "\n"
          R"(                                   Pool constraint: poolspec ::= poolfmt[","poolspec])"                                                "\n"
          "                                                      poolfmt ::= pool:sizeInMiB"                                                        "\n"
          R"(                                                    pool ::= "workspace"|"dlaSRAM"|"dlaLocalDRAM"|"dlaGlobalDRAM")"                    "\n"
          "  --profilingVerbosity=mode          Specify profiling verbosity. mode ::= layer_names_only|detailed|none (default = layer_names_only)"  "\n"
          "  --minTiming=M                      Set the minimum number of iterations used in kernel selection (default = "
                                                                                                                  << defaultMinTiming << ")"        "\n"
          "  --avgTiming=M                      Set the number of times averaged in each iteration for kernel selection (default = "
                                                                                                                  << defaultAvgTiming << ")"        "\n"
          "  --refit                            Mark the engine as refittable. This will allow the inspection of refittable layers "                "\n"
          "                                     and weights within the engine."                                                                     "\n"
          "  --versionCompatible, --vc          Mark the engine as version compatible. This allows the engine to be used with newer versions"       "\n"
          "                                     of TensorRT on the same host OS, as well as TensorRT's dispatch and lean runtimes."                 "\n"
          "                                     Only supported with explicit batch."                                                                "\n"
          R"(  --useRuntime=runtime               TensorRT runtime to execute engine. "lean" and "dispatch" require loading VC engine and do)"      "\n"
          "                                     not support building an engine."                                                                    "\n"
          R"(                                           runtime::= "full"|"lean"|"dispatch")"                                                       "\n"
          "  --leanDLLPath=<file>               External lean runtime DLL to use in version compatiable mode."                                      "\n"
          "  --excludeLeanRuntime               When --versionCompatible is enabled, this flag indicates that the generated engine should"          "\n"
          "                                     not include an embedded lean runtime. If this is set, the user must explicitly specify a"           "\n"
          "                                     valid lean runtime to use when loading the engine.  Only supported with explicit batch"             "\n"
          "                                     and weights within the engine."                                                                     "\n"
          "  --sparsity=spec                    Control sparsity (default = disabled). "                                                            "\n"
          R"(                                   Sparsity: spec ::= "disable", "enable", "force")"                                                   "\n"
          "                                     Note: Description about each of these options is as below"                                          "\n"
          "                                           disable = do not enable sparse tactics in the builder (this is the default)"                  "\n"
          "                                           enable  = enable sparse tactics in the builder (but these tactics will only be"               "\n"
          "                                                     considered if the weights have the right sparsity pattern)"                         "\n"
          "                                           force   = enable sparse tactics in the builder and force-overwrite the weights to have"       "\n"
          "                                                     a sparsity pattern (even if you loaded a model yourself)"                           "\n"
          "  --noTF32                           Disable tf32 precision (default is to enable tf32, in addition to fp32)"                            "\n"
          "  --fp16                             Enable fp16 precision, in addition to fp32 (default = disabled)"                                    "\n"
          "  --int8                             Enable int8 precision, in addition to fp32 (default = disabled)"                                    "\n"
          "  --fp8                              Enable fp8 precision, in addition to fp32 (default = disabled)"                                     "\n"
          "  --best                             Enable all precisions to achieve the best performance (default = disabled)"                         "\n"
          "  --directIO                         Avoid reformatting at network boundaries. (default = disabled)"                                     "\n"
          "  --precisionConstraints=spec        Control precision constraint setting. (default = none)"                                             "\n"
          R"(                                       Precision Constraints: spec ::= "none" | "obey" | "prefer")"                                    "\n"
          "                                         none = no constraints"                                                                          "\n"
          "                                         prefer = meet precision constraints set by --layerPrecisions/--layerOutputTypes if possible"    "\n"
          "                                         obey = meet precision constraints set by --layerPrecisions/--layerOutputTypes or fail"          "\n"
          "                                                otherwise"                                                                               "\n"
          "  --layerPrecisions=spec             Control per-layer precision constraints. Effective only when precisionConstraints is set to"        "\n"
          R"(                                   "obey" or "prefer". (default = none))"                                                              "\n"
          R"(                                   The specs are read left-to-right, and later ones override earlier ones. "*" can be used as a)"      "\n"
          "                                     layerName to specify the default precision for all the unspecified layers."                         "\n"
          R"(                                   Per-layer precision spec ::= layerPrecision[","spec])"                                              "\n"
          R"(                                                       layerPrecision ::= layerName":"precision)"                                      "\n"
          R"(                                                       precision ::= "fp32"|"fp16"|"int32"|"int8")"                                    "\n"
          "  --layerOutputTypes=spec            Control per-layer output type constraints. Effective only when precisionConstraints is set to"      "\n"
          R"(                                   "obey" or "prefer". (default = none)"                                                               "\n"
          R"(                                   The specs are read left-to-right, and later ones override earlier ones. "*" can be used as a)"      "\n"
          "                                     layerName to specify the default precision for all the unspecified layers. If a layer has more than""\n"
          R"(                                   one output, then multiple types separated by "+" can be provided for this layer.)"                  "\n"
          R"(                                   Per-layer output type spec ::= layerOutputTypes[","spec])"                                          "\n"
          R"(                                                         layerOutputTypes ::= layerName":"type)"                                       "\n"
          R"(                                                         type ::= "fp32"|"fp16"|"int32"|"int8"["+"type])"                              "\n"
          "  --layerDeviceTypes=spec            Specify layer-specific device type."                                                                "\n"
          "                                     The specs are read left-to-right, and later ones override earlier ones. If a layer does not have"   "\n"
          "                                     a device type specified, the layer will opt for the default device type."                           "\n"
          R"(                                   Per-layer device type spec ::= layerDeviceTypePair[","spec])"                                       "\n"
          R"(                                                         layerDeviceTypePair ::= layerName":"deviceType)"                              "\n"
          R"(                                                           deviceType ::= "GPU"|"DLA")"                                                "\n"
          "  --calib=<file>                     Read INT8 calibration cache file"                                                                   "\n"
          "  --safe                             Enable build safety certified engine, if DLA is enable, --buildDLAStandalone will be specified"     "\n"
          "                                     automatically (default = disabled)"                                                                 "\n"
          "  --buildDLAStandalone               Enable build DLA standalone loadable which can be loaded by cuDLA, when this option is enabled, "   "\n"
          "                                     --allowGPUFallback is disallowed and --skipInference is enabled by default. Additionally, "         "\n"
          "                                     specifying --inputIOFormats and --outputIOFormats restricts I/O data type and memory layout"        "\n"
          "                                     (default = disabled)"        "\n"
          "  --allowGPUFallback                 When DLA is enabled, allow GPU fallback for unsupported layers (default = disabled)"                "\n"
          "  --consistency                      Perform consistency checking on safety certified engine"                                            "\n"
          "  --restricted                       Enable safety scope checking with kSAFETY_SCOPE build flag"                                         "\n"
          "  --saveEngine=<file>                Save the serialized engine"                                                                         "\n"
          "  --loadEngine=<file>                Load a serialized engine"                                                                           "\n"
          "  --tacticSources=tactics            Specify the tactics to be used by adding (+) or removing (-) tactics from the default "             "\n"
          "                                     tactic sources (default = all available tactics)."                                                  "\n"
          "                                     Note: Currently only cuDNN, cuBLAS, cuBLAS-LT, and edge mask convolutions are listed as optional"   "\n"
          "                                           tactics."                                                                                     "\n"
          R"(                                   Tactic Sources: tactics ::= [","tactic])"                                                           "\n"
          "                                                     tactic  ::= (+|-)lib"                                                               "\n"
          R"(                                                   lib     ::= "CUBLAS"|"CUBLAS_LT"|"CUDNN"|"EDGE_MASK_CONVOLUTIONS")"                 "\n"
          R"(                                                               |"JIT_CONVOLUTIONS")"                                                   "\n"
          "                                     For example, to disable cudnn and enable cublas: --tacticSources=-CUDNN,+CUBLAS"                    "\n"
          "  --noBuilderCache                   Disable timing cache in builder (default is to enable timing cache)"                                "\n"
          "  --heuristic                        Enable tactic selection heuristic in builder (default is to disable the heuristic)"                 "\n"
          "  --timingCacheFile=<file>           Save/load the serialized global timing cache"                                                       "\n"
          "  --preview=features                 Specify preview feature to be used by adding (+) or removing (-) preview features from the default" "\n"
          R"(                                   Preview Features: features ::= [","feature])"                                                       "\n"
          "                                                       feature  ::= (+|-)flag"                                                           "\n"
          R"(                                                     flag     ::= "fasterDynamicShapes0805")"                                          "\n"
          R"(                                                                  |"disableExternalTacticSourcesForCore0805")"                         "\n"
          R"(                                                                  |"profileSharing0806")"                                              "\n"
          "  --builderOptimizationLevel         Set the builder optimization level. (default is 3)"                                                 "\n"
          "                                     Higher level allows TensorRT to spend more building time for more optimization options."            "\n"
          "                                     Valid values include integers from 0 to the maximum optimization level, which is currently 5."      "\n"
          "  --hardwareCompatibilityLevel=mode  Make the engine file compatible with other GPU architectures. (default = none)"                     "\n"
          R"(                                   Hardware Compatibility Level: mode ::= "none" | "ampere+")"                                         "\n"
          "                                         none = no compatibility"                                                                        "\n"
          "                                         ampere+ = compatible with Ampere and newer GPUs"                                                "\n"
          "  --tempdir=<dir>                    Overrides the default temporary directory TensorRT will use when creating temporary files."         "\n"
          "                                     See IRuntime::setTemporaryDirectory API documentation for more information."                        "\n"
          "  --tempfileControls=controls        Controls what TensorRT is allowed to use when creating temporary executable files."                 "\n"
          "                                     Should be a comma-separated list with entries in the format (in_memory|temporary):(allow|deny)."    "\n"
          "                                     in_memory: Controls whether TensorRT is allowed to create temporary in-memory executable files."    "\n"
          "                                     temporary: Controls whether TensorRT is allowed to create temporary executable files in the"        "\n"
          "                                                filesystem (in the directory given by --tempdir)."                                       "\n"
          "                                     For example, to allow in-memory files and disallow temporary files:"                                "\n"
          "                                         --tempfileControls=in_memory:allow,temporary:deny"                                              "\n"
          R"(                                   If a flag is unspecified, the default behavior is "allow".)"                                        "\n"
          "  --maxAuxStreams=N                  Set maximum number of auxiliary streams per inference stream that TRT is allowed to use to run "    "\n"
          "                                     kernels in parallel if the network contains ops that can run in parallel, with the cost of more "   "\n"
          "                                     memory usage. Set this to 0 for optimal memory usage. (default = using heuristics)"                 "\n"
          ;
    // clang-format on
    os << std::flush;
}

void SystemOptions::help(std::ostream& os)
{
    // clang-format off
    os << "=== System Options ==="                                                                         << std::endl <<
          "  --device=N                  Select cuda device N (default = "         << defaultDevice << ")" << std::endl <<
          "  --useDLACore=N              Select DLA core N for layers that support DLA (default = none)"   << std::endl <<
          "  --staticPlugins             Plugin library (.so) to load statically (can be specified multiple times)" << std::endl <<
          "  --dynamicPlugins            Plugin library (.so) to load dynamically and may be serialized with the engine if they are included in --setPluginsToSerialize (can be specified multiple times)" << std::endl <<
          "  --setPluginsToSerialize     Plugin library (.so) to be serialized with the engine (can be specified multiple times)" << std::endl <<
          "  --ignoreParsedPluginLibs    By default, when building a version-compatible engine, plugin libraries specified by the ONNX parser " << std::endl <<
          "                              are implicitly serialized with the engine (unless --excludeLeanRuntime is specified) and loaded dynamically. " << std::endl <<
          "                              Enable this flag to ignore these plugin libraries instead." << std::endl;
    // clang-format on
}

void InferenceOptions::help(std::ostream& os)
{
    // clang-format off
    os << "=== Inference Options ==="                                                                                                << std::endl <<
          "  --batch=N                   Set batch size for implicit batch engines (default = "              << defaultBatch << ")"  << std::endl <<
          "                              This option should not be used when the engine is built from an ONNX model or when dynamic" << std::endl <<
          "                              shapes are provided when the engine is built."                                              << std::endl <<
          "  --shapes=spec               Set input shapes for dynamic shapes inference inputs."                                      << std::endl <<
          R"(                              Note: Input names can be wrapped with escaped single quotes (ex: 'Input:0').)"            << std::endl <<
          "                              Example input shapes spec: input0:1x3x256x256, input1:1x3x128x128"                          << std::endl <<
          "                              Each input shape is supplied as a key-value pair where key is the input name and"           << std::endl <<
          "                              value is the dimensions (including the batch dimension) to be used for that input."         << std::endl <<
          "                              Each key-value pair has the key and value separated using a colon (:)."                     << std::endl <<
          "                              Multiple input shapes can be provided via comma-separated key-value pairs."                 << std::endl <<
          "  --loadInputs=spec           Load input values from files (default = generate random inputs). Input names can be "
                                                                                       "wrapped with single quotes (ex: 'Input:0')"  << std::endl <<
          R"(                            Input values spec ::= Ival[","spec])"                                                       << std::endl <<
          R"(                                         Ival ::= name":"file)"                                                         << std::endl <<
          "  --iterations=N              Run at least N inference iterations (default = "               << defaultIterations << ")"  << std::endl <<
          "  --warmUp=N                  Run for N milliseconds to warmup before measuring performance (default = "
                                                                                                            << defaultWarmUp << ")"  << std::endl <<
          "  --duration=N                Run performance measurements for at least N seconds wallclock time (default = "
                                                                                                          << defaultDuration << ")"  << std::endl <<
          "                              If -1 is specified, inference will keep running unless stopped manually"                    << std::endl <<
          "  --sleepTime=N               Delay inference start with a gap of N milliseconds between launch and compute "
                                                                                               "(default = " << defaultSleep << ")"  << std::endl <<
          "  --idleTime=N                Sleep N milliseconds between two continuous iterations"
                                                                                               "(default = " << defaultIdle << ")"   << std::endl <<
          "  --infStreams=N              Instantiate N engines to run inference concurrently (default = "  << defaultStreams << ")"  << std::endl <<
          "  --exposeDMA                 Serialize DMA transfers to and from device (default = disabled)."                           << std::endl <<
          "  --noDataTransfers           Disable DMA transfers to and from device (default = enabled)."                              << std::endl <<
          "  --useManagedMemory          Use managed memory instead of separate host and device allocations (default = disabled)."   << std::endl <<
          "  --useSpinWait               Actively synchronize on GPU events. This option may decrease synchronization time but "
                                                                             "increase CPU usage and power (default = disabled)"     << std::endl <<
          "  --threads                   Enable multithreading to drive engines with independent threads"
                                                                                " or speed up refitting (default = disabled) "       << std::endl <<
          "  --useCudaGraph              Use CUDA graph to capture engine execution and then launch inference (default = disabled)." << std::endl <<
          "                              This flag may be ignored if the graph capture fails."                                       << std::endl <<
          "  --timeDeserialize           Time the amount of time it takes to deserialize the network and exit."                      << std::endl <<
          "  --timeRefit                 Time the amount of time it takes to refit the engine before inference."                     << std::endl <<
          "  --separateProfileRun        Do not attach the profiler in the benchmark run; if profiling is enabled, a second "
                                                                                "profile run will be executed (default = disabled)"  << std::endl <<
          "  --skipInference             Exit after the engine has been built and skip inference perf measurement "
                                                                                                             "(default = disabled)"  << std::endl <<
          "  --persistentCacheRatio      Set the persistentCacheLimit in ratio, 0.5 represent half of max persistent L2 size "
                                                                                                                    "(default = 0)"  << std::endl;
    // clang-format on
}

void ReportingOptions::help(std::ostream& os)
{
    // clang-format off
    os << "=== Reporting Options ==="                                                                    << std::endl <<
          "  --verbose                   Use verbose logging (default = false)"                          << std::endl <<
          "  --avgRuns=N                 Report performance measurements averaged over N consecutive "
                                                       "iterations (default = " << defaultAvgRuns << ")" << std::endl <<
          "  --percentile=P1,P2,P3,...   Report performance for the P1,P2,P3,... percentages (0<=P_i<=100, 0 "
                                        "representing max perf, and 100 representing min perf; (default"
                                            " = " << joinValuesToString(defaultPercentiles, ",") << "%)" << std::endl <<
          "  --dumpRefit                 Print the refittable layers and weights from a refittable "
                                        "engine"                                                         << std::endl <<
          "  --dumpOutput                Print the output tensor(s) of the last inference iteration "
                                                                                  "(default = disabled)" << std::endl <<
          "  --dumpRawBindingsToFile     Print the input/output tensor(s) of the last inference iteration to file"
                                                                                  "(default = disabled)" << std::endl <<
          "  --dumpProfile               Print profile information per layer (default = disabled)"       << std::endl <<
          "  --dumpLayerInfo             Print layer information of the engine to console "
                                                                                "(default = disabled)"   << std::endl <<
          "  --exportTimes=<file>        Write the timing results in a json file (default = disabled)"   << std::endl <<
          "  --exportOutput=<file>       Write the output tensors to a json file (default = disabled)"   << std::endl <<
          "  --exportProfile=<file>      Write the profile information per layer in a json file "
                                                                              "(default = disabled)"     << std::endl <<
          "  --exportLayerInfo=<file>    Write the layer information of the engine in a json file "
                                                                              "(default = disabled)"     << std::endl;
    // clang-format on
}

void TaskInferenceOptions::help(std::ostream& os)
{
    // clang-format off
    os << "=== Task Inference Options ==="                                                                                           << std::endl <<
          "  engine=<file>               Specify a serialized engine for this task"                                                  << std::endl <<
          "  device=N                    Specify a GPU device for this task"                                                         << std::endl <<
          "  DLACore=N                   Specify a DLACore for this task"                                                            << std::endl <<
          "  batch=N                     Set batch size for implicit batch engines (default = "              << defaultBatch << ")"  << std::endl <<
          "                              This option should not be used for explicit batch engines"                                  << std::endl <<
          "  graph=1                     Use cuda graph for this task"                                                               << std::endl <<
          "  persistentCacheRatio=[0-1]  Set the persistentCacheLimit ratio for this task                            (default = 0)"  << std::endl;
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
          "                              Using ONNX models automatically forces explicit batch."                        << std::endl <<
    std::endl;
    // clang-format on
    ReportingOptions::help(os);
    os << std::endl;
    SystemOptions::help(os);
    os << std::endl;
    helpHelp(os);
}

void SafeBuilderOptions::printHelp(std::ostream& os)
{
    // clang-format off
    os << "=== Mandatory ==="                                                                                                                << std::endl <<
          "  --onnx=<file>               ONNX model"                                                                                         << std::endl <<
          " "                                                                                                                                << std::endl <<
          "=== Optional ==="                                                                                                                 << std::endl <<
          "  --inputIOFormats=spec       Type and format of each of the input tensors (default = all inputs in fp32:chw)"                    << std::endl <<
          "                              See --outputIOFormats help for the grammar of type and format list."                                << std::endl <<
          "                              Note: If this option is specified, please set comma-separated types and formats for all"            << std::endl <<
          "                                    inputs following the same order as network inputs ID (even if only one input"                 << std::endl <<
          "                                    needs specifying IO format) or set the type and format once for broadcasting."                << std::endl <<
          "  --outputIOFormats=spec      Type and format of each of the output tensors (default = all outputs in fp32:chw)"                  << std::endl <<
          "                              Note: If this option is specified, please set comma-separated types and formats for all"            << std::endl <<
          "                                    outputs following the same order as network outputs ID (even if only one output"              << std::endl <<
          "                                    needs specifying IO format) or set the type and format once for broadcasting."                << std::endl <<
          R"(                            IO Formats: spec  ::= IOfmt[","spec])"                                                              << std::endl <<
          "                                          IOfmt ::= type:fmt"                                                                     << std::endl <<
          R"(                                          type  ::= "fp32"|"fp16"|"int32"|"int8")"                                              << std::endl <<
          R"(                                          fmt   ::= ("chw"|"chw2"|"chw4"|"hwc8"|"chw16"|"chw32"|"dhwc8"|)"                      << std::endl <<
          R"(                                                     "cdhw32"|"hwc"|"dla_linear"|"dla_hwc4")["+"fmt])"                          << std::endl <<
          "  --int8                      Enable int8 precision, in addition to fp16 (default = disabled)"                                    << std::endl <<
          "  --consistency               Enable consistency check for serialized engine, (default = disabled)"                               << std::endl <<
          "  --std                       Build standard serialized engine, (default = disabled)"                                             << std::endl <<
          "  --calib=<file>              Read INT8 calibration cache file"                                                                   << std::endl <<
          "  --serialized=<file>         Save the serialized network"                                                                        << std::endl <<
          "  --staticPlugins             Plugin library (.so) to load statically (can be specified multiple times)"                          << std::endl <<
          "  --verbose or -v             Use verbose logging (default = false)"                                                              << std::endl <<
          "  --help or -h                Print this message"                                                                                 << std::endl <<
          "  --noBuilderCache            Disable timing cache in builder (default is to enable timing cache)"                                << std::endl <<
          "  --timingCacheFile=<file>    Save/load the serialized global timing cache"                                                       << std::endl <<
          "  --sparsity=spec             Control sparsity (default = disabled). "                                                            << std::endl <<
          R"(                              Sparsity: spec ::= "disable", "enable", "force")"                                                 << std::endl <<
          "                              Note: Description about each of these options is as below"                                          << std::endl <<
          "                                    disable = do not enable sparse tactics in the builder (this is the default)"                  << std::endl <<
          "                                    enable  = enable sparse tactics in the builder (but these tactics will only be"               << std::endl <<
          "                                              considered if the weights have the right sparsity pattern)"                         << std::endl <<
          "                                    force   = enable sparse tactics in the builder and force-overwrite the weights to have"       << std::endl <<
          "                                              a sparsity pattern"                                                                 << std::endl <<
          "  --minTiming=M               Set the minimum number of iterations used in kernel selection (default = "                          << std::endl <<
          ""                                                                                               << defaultMinTiming << ")"        << std::endl <<
          "  --avgTiming=M               Set the number of times averaged in each iteration for kernel selection (default = "                << std::endl <<
          ""                                                                                               << defaultAvgTiming << ")"        << std::endl <<
          ""                                                                                                                                 << std::endl;
    // clang-format on
}

} // namespace sample
