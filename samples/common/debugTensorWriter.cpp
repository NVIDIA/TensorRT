/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "debugTensorWriter.h"
#include "common.h"
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if CUDA_VERSION >= 11060
#include <cuda_fp8.h>
#endif
#if CUDA_VERSION >= 12070
#include <cuda_fp4.h>
#endif
#include <cuda_runtime_api.h>
#include <numeric>
namespace sample
{

namespace
{

class Int4
{
public:
    Int4() = default;
    explicit Int4(int8_t val)
        : mValue(val)
    {
    }

    operator int64_t() const
    {
        return static_cast<int64_t>(mValue);
    }

private:
    int8_t mValue{};
};

class Int4x2
{
public:
    using StorageType = uint8_t;

    Int4x2() = default;
    explicit Int4x2(StorageType val)
        : mRep(val)
    {
    }

    // Get a single element
    inline Int4 element(int32_t index) const
    {
        ASSERT(index == 0 || index == 1);
        return Int4(index == 0 ? static_cast<int8_t>(mRep << 4) >> 4 : static_cast<int8_t>(mRep) >> 4);
    }

private:
    StorageType mRep{};
};

#if CUDA_VERSION >= 12070
using Fp4 = __nv_fp4_e2m1;

class Fp4x2
{
public:
    using StorageType = uint8_t;

    Fp4x2() = default;
    explicit Fp4x2(StorageType val)
        : mRep(val)
    {
    }

    // Get a single element
    inline Fp4 element(int32_t index) const
    {
        ASSERT(index == 0 || index == 1);
        int8_t bits = index == 0 ? static_cast<int8_t>(mRep << 4) >> 4 : static_cast<int8_t>(mRep) >> 4;
        Fp4 fp4_el = *reinterpret_cast<Fp4*>(&bits);
        return fp4_el;
    }

private:
    StorageType mRep{};
};
#endif

// Iterator that can handle packed format data (int4 and fp4)
template <typename T>
class DataIterator
{
public:
#if CUDA_VERSION >= 12070
    using value_type
        = std::conditional_t<std::is_same_v<T, Int4x2>, Int4, std::conditional_t<std::is_same_v<T, Fp4x2>, Fp4, T>>;
#else
    using value_type = std::conditional_t<std::is_same_v<T, Int4x2>, Int4, T>;
#endif

    DataIterator(void const* data, int64_t volume, int64_t index = 0)
        : mData(static_cast<uint8_t const*>(data))
        , mVolume(volume)
        , mIndex(index)
    {
    }

    value_type operator*() const
    {
        if constexpr (std::is_same_v<T, Int4x2>)
        {
            // For Int4x2, each byte contains two 4-bit integers
            Int4x2 packed(mData[mIndex / 2]);
            return packed.element(mIndex % 2);
        }
#if CUDA_VERSION >= 12070
        else if constexpr (std::is_same_v<T, Fp4x2>)
        {
            // For Fp4x2, each byte contains two 4-bit floating point numbers
            Fp4x2 packed(mData[mIndex / 2]);
            return packed.element(mIndex % 2);
        }
#endif
        else
        {
            return reinterpret_cast<T const*>(mData)[mIndex];
        }
    }

    DataIterator& operator++()
    {
        ++mIndex;
        return *this;
    }

    DataIterator operator++(int)
    {
        DataIterator tmp = *this;
        ++mIndex;
        return tmp;
    }

    bool operator==(DataIterator const& other) const
    {
        return mIndex == other.mIndex;
    }

    bool operator!=(DataIterator const& other) const
    {
        return mIndex != other.mIndex;
    }

    DataIterator operator+(int64_t n) const
    {
        DataIterator tmp = *this;
        tmp.mIndex += n;
        return tmp;
    }

private:
    uint8_t const* mData;
    int64_t mVolume;
    int64_t mIndex;
};

template <typename T>
class DataRange
{
public:
    using iterator = DataIterator<T>;
    using value_type = typename iterator::value_type;

    DataRange(void const* data, int64_t volume)
        : mData(data)
        , mVolume(volume)
    {
    }

    iterator begin() const
    {
        return iterator(mData, mVolume, 0);
    }
    iterator end() const
    {
        return iterator(mData, mVolume, mVolume);
    }

private:
    void const* mData;
    int64_t mVolume;
};

template <typename T>
static constexpr bool isFloatingPoint
    = std::is_floating_point_v<T> || std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>
#if CUDA_VERSION >= 11060
    || std::is_same_v<T, __nv_fp8_e4m3>
#endif
#if CUDA_VERSION >= 12070
    || std::is_same_v<T, Fp4> || std::is_same_v<T, Fp4x2>
#endif
    ;

constexpr int32_t kFLOATING_POINT_PRECISION = 6;
constexpr int32_t kFLOATING_POINT_WIDTH = 13;

std::string_view getDataTypeString(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kBOOL: return "BOOL";
    case nvinfer1::DataType::kINT4: return "INT4";
    case nvinfer1::DataType::kINT8: return "INT8";
    case nvinfer1::DataType::kINT32: return "INT32";
    case nvinfer1::DataType::kINT64: return "INT64";
    case nvinfer1::DataType::kUINT8: return "UINT8";
    case nvinfer1::DataType::kFP4: return "FP4";
    case nvinfer1::DataType::kFP8: return "FP8";
    case nvinfer1::DataType::kE8M0: return "E8M0";
    case nvinfer1::DataType::kHALF: return "HALF";
    case nvinfer1::DataType::kBF16: return "BF16";
    case nvinfer1::DataType::kFLOAT: return "FLOAT";
    }
    return "UNKNOWN";
}

template <typename T>
void printTensorElements(T const* data, int64_t volume, std::ofstream& f)
{
    f << "        \"elements\": \"";
    constexpr int32_t kPRINT_ELEMENTS_COUNT = 10;
    int64_t firstHalf = std::min(static_cast<int64_t>(kPRINT_ELEMENTS_COUNT / 2), volume);
    int64_t secondHalf = (volume > kPRINT_ELEMENTS_COUNT)
        ? kPRINT_ELEMENTS_COUNT / 2
        : std::max(static_cast<int64_t>(0), volume - kPRINT_ELEMENTS_COUNT / 2);

    auto printElement = [&f](auto value) {
        if constexpr (isFloatingPoint<T>)
        {
            f << static_cast<float>(value);
        }
        else
        {
            f << static_cast<int64_t>(value);
        }
    };

    DataRange<T> range(data, volume);
    auto it = range.begin();

    // Print first half elements
    std::string delimiter = "";
    for (int64_t i = 0; i < firstHalf; ++i)
    {
        f << delimiter;
        printElement(*it++);
        delimiter = ", ";
    }

    // Add ellipsis if needed
    f << (volume > kPRINT_ELEMENTS_COUNT ? ", ..." : "");

    // Print last elements
    it = range.begin() + (volume - secondHalf);
    for (int64_t i = volume - secondHalf; i < volume; ++i)
    {
        f << ", ";
        printElement(*it++);
    }

    f << "\"" << std::endl;
}

template <typename T>
void processTensorSummary(void const* addr_host, int64_t volume, std::ofstream& f)
{
    DataRange<T> range(addr_host, volume);

    if constexpr (isFloatingPoint<T>)
    {
        float minVal = std::numeric_limits<float>::max();
        float maxVal = std::numeric_limits<float>::lowest();
        double sum = 0.0;

        for (auto value : range)
        {
            float val = static_cast<float>(value);
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
            sum += val;
        }
        float avgVal = sum / volume;

        // nan and inf turn into string in json
        auto valueToStr = [](float val) -> std::string {
            std::stringstream ss;
            if (!std::isfinite(val))
            {
                ss << "\"" << val << "\"";
            }
            else
            {
                ss << val;
            }
            return ss.str();
        };
        f << "        \"min\": " << valueToStr(minVal) << "," << std::endl;
        f << "        \"max\": " << valueToStr(maxVal) << "," << std::endl;
        f << "        \"avg\": " << valueToStr(avgVal) << "," << std::endl;
    }
    else
    {
        // For integer types, use int64_t for min/max calculation
        int64_t minVal = std::numeric_limits<int64_t>::max();
        int64_t maxVal = std::numeric_limits<int64_t>::lowest();
        int64_t sum = 0;

        for (auto value : range)
        {
            int64_t val = static_cast<int64_t>(value);
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
            sum += val;
        }
        double avgVal = static_cast<double>(sum) / volume;

        f << "        \"min\": " << minVal << "," << std::endl;
        f << "        \"max\": " << maxVal << "," << std::endl;
        f << "        \"avg\": " << avgVal << "," << std::endl;
    }

    printTensorElements<T>(static_cast<T const*>(addr_host), volume, f);
}

std::string getCurrentTimeString()
{
    auto now = std::chrono::system_clock::now();
    auto nowC = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&nowC), "%Y-%m-%dT%H:%M:%S%z");
    return ss.str();
}

template <typename T>
void writeTensorStringRecursive(T const* data, nvinfer1::Dims const& shape, int32_t currentDim, int64_t offset,
    int64_t stride, std::ofstream& f, bool isFirstElement = true, int32_t indent = 0, int32_t maxWidth = 0)
{
    bool isLastDim = currentDim == shape.nbDims - 1;
    if (isLastDim)
    {
        // Last dimension - print elements in a row
        f << std::string(indent, ' ') << "[";
        DataRange<T> range(data + offset, shape.d[currentDim]);
        auto it = range.begin();
        for (int32_t i = 0; i < shape.d[currentDim]; ++i)
        {
            if (i > 0)
            {
                f << " ";
            }
            if constexpr (isFloatingPoint<T>)
            {
                f << std::scientific << std::setprecision(kFLOATING_POINT_PRECISION) << std::setw(kFLOATING_POINT_WIDTH)
                  << std::right << static_cast<float>(*it++);
            }
            else
            {
                f << std::setw(maxWidth) << static_cast<int64_t>(*it++);
            }
        }
        f << "]" << std::endl;
    }
    else
    {
        // For higher dimensions, print each slice
        f << std::string(indent, ' ') << "[" << std::endl;
        for (int32_t i = 0; i < shape.d[currentDim]; ++i)
        {
            writeTensorStringRecursive(data, shape, currentDim + 1, offset + i * stride,
                stride / shape.d[currentDim + 1], f, i == 0, indent + 1, maxWidth);
        }
        f << std::string(indent, ' ') << "]" << std::endl;
    }
}

template <typename T>
int32_t getMaxWidthInDimension(
    T const* data, nvinfer1::Dims const& shape, int32_t currentDim, int64_t offset, int64_t stride)
{
    int32_t maxWidth = 0;
    if (currentDim == shape.nbDims - 1)
    {
        // Last dimension - check each element
        DataRange<T> range(data + offset, shape.d[currentDim]);
        for (auto value : range)
        {
            std::stringstream ss;
            ss << static_cast<int64_t>(value);
            maxWidth = std::max(maxWidth, static_cast<int32_t>(ss.str().length()));
        }
    }
    else
    {
        // For higher dimensions, check each slice
        for (int64_t i = 0; i < shape.d[currentDim]; ++i)
        {
            maxWidth = std::max(maxWidth,
                getMaxWidthInDimension(
                    data, shape, currentDim + 1, offset + i * stride, stride / shape.d[currentDim + 1]));
        }
    }
    return maxWidth;
}

template <typename T>
void writeTensorString(
    T const* data, nvinfer1::Dims const& shape, std::string_view tensorName, std::string const& fileName)
{
    sample::gLogVerbose << "Writing debug tensor '" << tensorName << "' to file '" << fileName << "'" << std::endl;

    std::ofstream f(fileName, std::ios::out);
    if (!f)
    {
        sample::gLogError << "Cannot open file for write: " << fileName << std::endl;
        return;
    }

    if (shape.nbDims == 0)
    {
        f << "[]";
        return;
    }

    int64_t totalElements = 1;
    for (int32_t i = 0; i < shape.nbDims; ++i)
    {
        totalElements *= shape.d[i];
    }

    if (totalElements == 0)
    {
        f << "[]";
        return;
    }

    // Calculate stride for the first dimension
    int64_t stride = totalElements / shape.d[0];

    // Calculate max width for proper alignment only for non-floating point types
    int32_t maxWidth = 0;
    if constexpr (!isFloatingPoint<T>)
    {
        maxWidth = getMaxWidthInDimension(data, shape, 0, 0, stride);
    }

    writeTensorStringRecursive(data, shape, 0, 0, stride, f, true, 0, maxWidth);
    f << std::endl;
}

std::string writeStringFile(void const* addr_host, nvinfer1::DataType type, nvinfer1::Dims const& shape,
    std::string const& tensorName, std::string const& prefix)
{
    std::string fileName = genFilenameSafeString(prefix + tensorName + ".str");

    switch (type)
    {
    case nvinfer1::DataType::kBOOL:
        writeTensorString(static_cast<bool const*>(addr_host), shape, tensorName, fileName);
        break;
    case nvinfer1::DataType::kINT4:
        writeTensorString(reinterpret_cast<Int4x2 const*>(addr_host), shape, tensorName, fileName);
        break;
    case nvinfer1::DataType::kINT8:
        writeTensorString(static_cast<int8_t const*>(addr_host), shape, tensorName, fileName);
        break;
    case nvinfer1::DataType::kINT32:
        writeTensorString(static_cast<int32_t const*>(addr_host), shape, tensorName, fileName);
        break;
    case nvinfer1::DataType::kINT64:
        writeTensorString(static_cast<int64_t const*>(addr_host), shape, tensorName, fileName);
        break;
    case nvinfer1::DataType::kUINT8:
        writeTensorString(static_cast<uint8_t const*>(addr_host), shape, tensorName, fileName);
        break;
    case nvinfer1::DataType::kFP4:
#if CUDA_VERSION >= 12070
        writeTensorString(static_cast<Fp4x2 const*>(addr_host), shape, tensorName, fileName);
        break;
#else
        sample::gLogWarning << "Unsupported data type kFP4 for tensor string dump in this CUDA version." << std::endl;
        return "";
#endif
    case nvinfer1::DataType::kFP8:
#if CUDA_VERSION >= 11060
        writeTensorString(static_cast<__nv_fp8_e4m3 const*>(addr_host), shape, tensorName, fileName);
        break;
#else
        sample::gLogWarning << "Unsupported data type kFP8 for tensor string dump in this CUDA version." << std::endl;
        return "";
#endif
    case nvinfer1::DataType::kE8M0:
        sample::gLogWarning << "Unsupported data type kE8M0 for tensor string dump." << std::endl;
        return "";
    case nvinfer1::DataType::kHALF:
        writeTensorString(static_cast<half const*>(addr_host), shape, tensorName, fileName);
        break;
    case nvinfer1::DataType::kBF16:
        writeTensorString(static_cast<nv_bfloat16 const*>(addr_host), shape, tensorName, fileName);
        break;
    case nvinfer1::DataType::kFLOAT:
        writeTensorString(static_cast<float const*>(addr_host), shape, tensorName, fileName);
        break;
    }
    return fileName;
}

std::string escapeJsonString(std::string_view str)
{
    std::string result;
    result.reserve(str.length());
    for (char c : str)
    {
        switch (c)
        {
        case '\\': result += "\\\\"; break;
        case '\"': result += "\\\""; break;
        case '\b': result += "\\b"; break;
        case '\f': result += "\\f"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default: result += c;
        }
    }
    return result;
}

template <typename U, typename T>
std::vector<U> convertBufferTo(T const* data, int64_t volume)
{
    std::vector<U> buffer(volume);
    DataRange<T> range(data, volume);
    int64_t i = 0;
    for (auto value : range)
    {
        buffer[i++] = static_cast<U>(value);
    }
    return buffer;
}

} // namespace

DebugTensorWriter::DebugTensorWriter(std::unordered_map<std::string, std::string> const& debugTensorFileNames,
    std::vector<std::string> const& debugTensorFormats, std::string const& engineName, std::string const& cmdline)
    : mDebugTensorFileNames(debugTensorFileNames)
    , mDebugTensorFormats(debugTensorFormats)
    , mEngineName(engineName)
    , mCmdline(cmdline)
{
    // Create a summary file if "summary" format is requested
    if (std::find(mDebugTensorFormats.begin(), mDebugTensorFormats.end(), "summary") != mDebugTensorFormats.end())
    {
        mSummaryFileName = "tensor_summary.json";
        mSummaryFile.open(mSummaryFileName, std::ios::out);
        if (mSummaryFile.is_open())
        {
            sample::gLogInfo << "Writing tensor summary to file: " << mSummaryFileName << std::endl;
            writeSummaryHeader();
        }
        else
        {
            sample::gLogError << "Failed to open tensor summary file: " << mSummaryFileName << std::endl;
        }
    }
}

DebugTensorWriter::~DebugTensorWriter()
{
    // Close the summary file
    if (mSummaryFile.is_open())
    {
        writeSummaryFooter();
        mSummaryFile.close();
    }
}

void DebugTensorWriter::writeSummaryHeader()
{
    mSummaryFile << "{" << std::endl;
    mSummaryFile << "  \"metadata\": {" << std::endl;
    mSummaryFile << "    \"title\": \"Tensor Summary Report\"," << std::endl;
    mSummaryFile << "    \"time_generated\": \"" << getCurrentTimeString() << "\"," << std::endl;
    mSummaryFile << "    \"engine_name\": \"" << mEngineName << "\"," << std::endl;
    mSummaryFile << "    \"command_line\": \"" << escapeJsonString(mCmdline) << "\"" << std::endl;
    mSummaryFile << "  }," << std::endl;
    mSummaryFile << "  \"tensors\": [" << std::endl;
}

void DebugTensorWriter::writeSummaryFooter()
{
    mSummaryFile << std::endl << "  ]" << std::endl;
    mSummaryFile << "}" << std::endl;
}

void DebugTensorWriter::writeSummary(std::string_view name, nvinfer1::Dims const& shape, nvinfer1::DataType type,
    int64_t volume, void const* addr_host, std::string_view assignedFileName, std::string_view numpyFileName,
    std::string_view stringFileName, std::string_view rawFileName)
{
    // Add comma separator if not the first tensor
    if (!mFirstTensor)
    {
        mSummaryFile << "," << std::endl;
    }
    mFirstTensor = false;

    // Write tensor information
    mSummaryFile << "  {\n"
                 << "    \"name\": \"" << name << "\",\n"
                 << "    \"shape\": [";

    for (int32_t i = 0; i < shape.nbDims; ++i)
    {
        if (i > 0)
        {
            mSummaryFile << ", ";
        }
        mSummaryFile << shape.d[i];
    }

    mSummaryFile << "],\n"
                 << "    \"type\": \"" << getDataTypeString(type) << "\",\n";

    // Write statistics
    mSummaryFile << "    \"statistics\": {\n";

    switch (type)
    {
    case nvinfer1::DataType::kBOOL: processTensorSummary<bool>(addr_host, volume, mSummaryFile); break;
    case nvinfer1::DataType::kINT4: processTensorSummary<Int4x2>(addr_host, volume, mSummaryFile); break;
    case nvinfer1::DataType::kINT8: processTensorSummary<int8_t>(addr_host, volume, mSummaryFile); break;
    case nvinfer1::DataType::kINT32: processTensorSummary<int32_t>(addr_host, volume, mSummaryFile); break;
    case nvinfer1::DataType::kINT64: processTensorSummary<int64_t>(addr_host, volume, mSummaryFile); break;
    case nvinfer1::DataType::kUINT8: processTensorSummary<uint8_t>(addr_host, volume, mSummaryFile); break;
    case nvinfer1::DataType::kFP4:
#if CUDA_VERSION >= 12070
        processTensorSummary<Fp4x2>(addr_host, volume, mSummaryFile);
#else
        sample::gLogWarning << "Unsupported data type kFP4 for tensor '" << name
                            << "' summary dump in this CUDA version." << std::endl;
#endif
        break;
    case nvinfer1::DataType::kFP8:
#if CUDA_VERSION >= 11060
        processTensorSummary<__nv_fp8_e4m3>(addr_host, volume, mSummaryFile);
        break;
#else
        sample::gLogWarning << "Unsupported data type kFP8 for tensor '" << name
                            << "' summary dump in this CUDA version." << std::endl;
#endif
        break;
    case nvinfer1::DataType::kE8M0:
        sample::gLogWarning << "Unsupported data type kE8M0 for tensor '" << name << "' summary dump." << std::endl;
        break;
    case nvinfer1::DataType::kHALF: processTensorSummary<half>(addr_host, volume, mSummaryFile); break;
    case nvinfer1::DataType::kBF16: processTensorSummary<nv_bfloat16>(addr_host, volume, mSummaryFile); break;
    case nvinfer1::DataType::kFLOAT: processTensorSummary<float>(addr_host, volume, mSummaryFile); break;
    }

    mSummaryFile << "    }";

    // Write file information only if at least one file exists
    if (!assignedFileName.empty() || !numpyFileName.empty() || !stringFileName.empty() || !rawFileName.empty())
    {
        mSummaryFile << ",\n    \"files\": {\n";
        std::string delimiter = "";

        if (!assignedFileName.empty())
        {
            mSummaryFile << delimiter << "      \"assigned\": \"" << escapeJsonString(assignedFileName) << "\"";
            delimiter = ",\n";
        }

        if (!numpyFileName.empty())
        {
            mSummaryFile << delimiter << "      \"numpy\": \"" << escapeJsonString(numpyFileName) << "\"";
            delimiter = ",\n";
        }

        if (!stringFileName.empty())
        {
            mSummaryFile << delimiter << "      \"string\": \"" << escapeJsonString(stringFileName) << "\"";
            delimiter = ",\n";
        }

        if (!rawFileName.empty())
        {
            mSummaryFile << delimiter << "      \"raw\": \"" << escapeJsonString(rawFileName) << "\"";
        }

        mSummaryFile << "\n    }";
    }

    mSummaryFile << "\n  }";
}

bool writeNumpyFile(void const* addr_host, std::string_view dtype, nvinfer1::Dims const& shape, int64_t size,
    std::string_view tensorName, std::string const& fileName)
{
    sample::gLogVerbose << "Writing debug tensor '" << tensorName << "' to numpy file '" << fileName << "'"
                        << std::endl;

    std::ofstream f(fileName, std::ios::out | std::ios::binary);
    if (!f)
    {
        sample::gLogError << "Cannot open file for write: " << fileName << std::endl;
        return false;
    }

    // Write numpy magic string and version
    char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    char version[] = {'\x01', '\x00'};
    f.write(magic, sizeof(magic));
    f.write(version, sizeof(version));

    // Construct header
    std::stringstream header;
    header << "{'descr': '" << dtype << "', 'fortran_order': False, 'shape': (";

    for (int32_t i = 0; i < shape.nbDims; i++)
    {
        header << shape.d[i];
        header << ", ";
    }
    header << "), }";

    // Pad header to 16 bytes alignment
    std::string headerStr = header.str();
    int32_t headerLen = 10 + headerStr.length();
    int32_t padding = 16 - ((headerLen + 1) % 16);
    headerStr.append(padding, ' ');
    headerStr += '\n';

    // Write header length and header
    uint16_t headerSize = headerStr.length();
    f.write(reinterpret_cast<char*>(&headerSize), sizeof(uint16_t));
    f.write(headerStr.c_str(), headerSize);

    // Write data
    f.write(static_cast<char const*>(addr_host), size);
    f.close();

    return true;
}

std::string writeNumpy(nvinfer1::DataType type, void const* addr_host, int64_t volume, nvinfer1::Dims const& shape,
    std::string const& name, std::string const& prefix)
{
    std::string fileName = prefix + name;
    std::string_view dtype = "";
    void const* data = addr_host;
    int64_t size = samplesCommon::getNbBytes(type, volume);
    std::vector<float> floatBuffer;
    std::vector<int8_t> int8Buffer;

    auto convertToFloat = [&](std::vector<float> const& buffer) {
        sample::gLogWarning << "Converting " << getDataTypeString(type) << " to float for numpy dump of tensor '"
                            << name << "'." << std::endl;
        dtype = "<f4";
        data = buffer.data();
        size = volume * sizeof(float);
        fileName += "_to_float";
    };

    auto convertToInt8 = [&](std::vector<int8_t> const& buffer) {
        sample::gLogWarning << "Converting " << getDataTypeString(type) << " to int8 for numpy dump of tensor '" << name
                            << "'." << std::endl;
        dtype = "<i1";
        data = buffer.data();
        size = volume * sizeof(int8_t);
        fileName += "_to_int8";
    };

    switch (type)
    {
    case nvinfer1::DataType::kBOOL: dtype = "|b1"; break;
    case nvinfer1::DataType::kINT4:
        int8Buffer = convertBufferTo<int8_t>(reinterpret_cast<Int4x2 const*>(addr_host), volume);
        convertToInt8(int8Buffer);
        break;
    case nvinfer1::DataType::kINT8: dtype = "<i1"; break;
    case nvinfer1::DataType::kINT32: dtype = "<i4"; break;
    case nvinfer1::DataType::kINT64: dtype = "<i8"; break;
    case nvinfer1::DataType::kUINT8: dtype = "|u1"; break;
    case nvinfer1::DataType::kFP4:
#if CUDA_VERSION >= 12070
        floatBuffer = convertBufferTo<float>(static_cast<Fp4x2 const*>(addr_host), volume);
        convertToFloat(floatBuffer);
#else
        sample::gLogWarning << "Unsupported data type kFP4 for tensor '" << name << "' numpy dump in this CUDA version."
                            << std::endl;
        return "";
#endif
        break;
    case nvinfer1::DataType::kFP8:
#if CUDA_VERSION >= 11060
        floatBuffer = convertBufferTo<float>(static_cast<__nv_fp8_e4m3 const*>(addr_host), volume);
        convertToFloat(floatBuffer);
#else
        sample::gLogWarning << "Unsupported data type kFP8 for tensor '" << name << "' numpy dump in this CUDA version."
                            << std::endl;
        return "";
#endif
        break;
    case nvinfer1::DataType::kE8M0:
        sample::gLogWarning << "Unsupported data type kE8M0 for tensor '" << name << "' numpy dump." << std::endl;
        return "";
    case nvinfer1::DataType::kHALF: dtype = "<f2"; break;
    case nvinfer1::DataType::kBF16:
        floatBuffer = convertBufferTo<float>(static_cast<nv_bfloat16 const*>(addr_host), volume);
        convertToFloat(floatBuffer);
        break;
    case nvinfer1::DataType::kFLOAT: dtype = "<f4"; break;
    }

    if (!dtype.empty())
    {

        fileName += ".npy";
        fileName = genFilenameSafeString(fileName);
        writeNumpyFile(data, dtype, shape, size, name, fileName);
        return fileName;
    }
    return "";
}

bool DebugTensorWriter::processDebugTensor(void const* addr, nvinfer1::TensorLocation location, nvinfer1::DataType type,
    nvinfer1::Dims const& shape, char const* name, cudaStream_t stream)
{
    CHECK(cudaStreamSynchronize(stream));
    // Store data from callback.
    auto volume = std::accumulate(shape.d, shape.d + shape.nbDims, 1LL, std::multiplies<int64_t>{});
    int64_t size = samplesCommon::getNbBytes(type, volume);
    std::vector<char> hostDataOut;
    void const* addrHost = nullptr;
    if (location == nvinfer1::TensorLocation::kDEVICE)
    {
        hostDataOut.resize(size);
        CHECK(cudaMemcpy(hostDataOut.data(), addr, size, cudaMemcpyDeviceToHost));
        addrHost = hostDataOut.data();
    }
    else
    {
        addrHost = addr;
    }

    std::string assignedFileName;
    std::string numpyFileName;
    std::string rawFileName;
    std::string stringFileName;
    auto it = mDebugTensorFileNames.find(name);
    if (it != mDebugTensorFileNames.end())
    {
        assignedFileName = it->second;
        std::ofstream f(assignedFileName, std::ios::out | std::ios::binary);
        ASSERT(f && "Cannot open file for write");
        sample::gLogVerbose << "Writing debug tensor '" << name << "' to file '" << assignedFileName << "'"
                            << std::endl;
        f.write(static_cast<char const*>(addrHost), size);
        f.close();
    }

    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << mTensorIndex << "_";
    std::string prefix = ss.str();

    if (std::find(mDebugTensorFormats.begin(), mDebugTensorFormats.end(), "raw") != mDebugTensorFormats.end())
    {
        rawFileName = genFilenameSafeString(prefix + name + ".raw");
        sample::gLogVerbose << "Writing debug tensor '" << name << "' to raw file '" << rawFileName << "'" << std::endl;
        std::ofstream f(rawFileName, std::ios::out | std::ios::binary);
        ASSERT(f && "Cannot open file for write");
        f.write(static_cast<char const*>(addrHost), size);
        f.close();
    }

    if (std::find(mDebugTensorFormats.begin(), mDebugTensorFormats.end(), "numpy") != mDebugTensorFormats.end())
    {
        numpyFileName = writeNumpy(type, addrHost, volume, shape, name, prefix);
    }

    if (std::find(mDebugTensorFormats.begin(), mDebugTensorFormats.end(), "string") != mDebugTensorFormats.end())
    {
        stringFileName = writeStringFile(addrHost, type, shape, name, prefix);
    }

    if (std::find(mDebugTensorFormats.begin(), mDebugTensorFormats.end(), "summary") != mDebugTensorFormats.end()
        && mSummaryFile.is_open())
    {
        writeSummary(name, shape, type, volume, addrHost, assignedFileName, numpyFileName, stringFileName, rawFileName);
        mSummaryFile.flush();
    }

    mTensorIndex++;
    return true;
}

} // namespace sample
