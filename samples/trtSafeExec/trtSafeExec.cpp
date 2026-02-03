/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "NvInferSafeRuntime.h" // TRTS-10206: NvInferSafeRuntime.h may be refactored
#include "cuda_runtime.h"
#include "delayStreamKernel.h"
#include "safeCommon.h"
#include "safeCudaAllocator.h"
#include "safeErrorRecorder.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace nvinfer1;
using namespace samplesSafeCommon;
using SafetyPluginArguments = std::vector<SafetyPluginLibraryArgument>;
__attribute__((weak)) std::shared_ptr<sample::SampleSafeRecorder> gSafeRecorder
    = std::make_shared<sample::SampleSafeRecorder>(nvinfer2::safe::Severity::kINFO);

//!
//! \brief The TimingMetric struct stores the timing metric of a performance metric
//!
//! \param[in] gpuTime: GPU time in milliseconds.
//! \param[in] hostTime: Host time in milliseconds.
//! \param[in] enqueueTime: Enqueue time in milliseconds.
//!
using TimingMetric = std::array<float, 3>;
using TimingMetrics = std::vector<TimingMetric>;
//!
//! \brief The SafeExecArgs struct stores the arguments required by the sample
//!
class SafeExecArgs
{
public:
    std::string engineFile{"sample.engine"};
    int32_t iterations{10};
    int32_t avgRuns{10};
    int32_t warmUp{1};
    int32_t device{0};
    int32_t streams{1};
    float idle{0.F};
    float duration{3.F};
    float sleep{2.F};
    float percentile{99.F};
    bool spin{false};
    bool verbose{false};
    bool debug{false};
    bool help{false};
    bool useCudaGraph{false};
    bool useScratchMemory{false};
    bool separateProfileRun{false};
    int32_t threads{1};
    int64_t ioProfile{0};
    SafetyPluginArguments pluginLibraries;
    std::unordered_map<std::string, std::string>
        loadInputs; //!< Map of tensor names to file paths for loading custom input data
};

//!
//! \brief The PerformanceResult struct stores the performance result of a performance metric
//!
class SafePerformanceResult
{
public:
    float min{0.F};
    float max{0.F};
    float mean{0.F};
    float median{0.F};
    float percentile{0.F};
    float coeffVar{0.F};
};

namespace
{
//! Default alignment for memory allocations
constexpr uint64_t kDEFAULT_ALIGNMENT{256U};

//!
//! \brief RAII wrapper for SafeMemAllocator to ensure automatic cleanup.
//!
class ScopedSafeMemory
{
public:
    ScopedSafeMemory(uint64_t size, uint64_t alignment, nvinfer2::safe::MemoryPlacement placement,
        nvinfer2::safe::MemoryUsage usage, nvinfer2::safe::ISafeRecorder& recorder)
        : mPtr(nullptr)
        , mPlacement(placement)
        , mRecorder(recorder)
    {
        auto& allocator = nvinfer2::safe::getSafeMemAllocator();
        mPtr = allocator.allocate(size, alignment, placement, usage, recorder);
    }

    ~ScopedSafeMemory()
    {
        if (mPtr)
        {
            auto& allocator = nvinfer2::safe::getSafeMemAllocator();
            allocator.deallocate(mPtr, mPlacement, mRecorder);
        }
    }

    ScopedSafeMemory(ScopedSafeMemory const&) = delete;
    ScopedSafeMemory& operator=(ScopedSafeMemory const&) = delete;

    ScopedSafeMemory(ScopedSafeMemory&& other) noexcept
        : mPtr(other.mPtr)
        , mPlacement(other.mPlacement)
        , mRecorder(other.mRecorder)
    {
        other.mPtr = nullptr;
    }

    ScopedSafeMemory& operator=(ScopedSafeMemory&& other) noexcept
    {
        if (this != &other)
        {
            // Clean up existing resource
            auto& allocator = nvinfer2::safe::getSafeMemAllocator();
            allocator.deallocate(mPtr, mPlacement, mRecorder);

            // Transfer ownership
            mPtr = other.mPtr;
            mPlacement = other.mPlacement;
            other.mPtr = nullptr;
        }
        return *this;
    }

    void* get() const noexcept
    {
        return mPtr;
    }

    explicit operator bool() const noexcept
    {
        return mPtr != nullptr;
    }

    bool operator==(ScopedSafeMemory const& other) const noexcept
    {
        return mPtr == other.mPtr;
    }

    bool operator!=(ScopedSafeMemory const& other) const noexcept
    {
        return mPtr != other.mPtr;
    }

private:
    void* mPtr;
    nvinfer2::safe::MemoryPlacement mPlacement;
    nvinfer2::safe::ISafeRecorder& mRecorder;
};

//! Similar to C++20 template function std::ssize.
template <class C>
constexpr auto signedSize(C const& c) ->
    typename std::common_type_t<std::ptrdiff_t, std::make_signed_t<decltype(c.size())>>
{
    /* polyspace +2 RTE:OVFL [Justified:Low] */
    return static_cast<typename std::common_type_t<std::ptrdiff_t, std::make_signed_t<decltype(c.size())>>>(c.size());
}

bool parseString(std::string const& arg, std::string const& name, std::string& value)
{
    std::string const pattern = "--" + name + "=";
    bool const matched = !arg.compare(0ULL, pattern.size(), pattern);
    if (matched)
    {
        value = arg.size() > pattern.size() ? arg.substr(pattern.size()) : "";
        safeLogInfo(*gSafeRecorder, name + " : " + value);
    }
    return matched;
}

bool parseBool(std::string const& arg, std::string const& name, bool isSingleDash = false)
{
    std::string const pattern = (isSingleDash ? "-" : "--") + name;
    bool const matched = (arg == pattern);
    if (matched)
    {
        safeLogInfo(*gSafeRecorder, name + " : True");
    }
    return matched;
}

//!
//! \brief Get the percentile of a performance metric
//!
//! \param[in] percentage: Percentile to get
//! \param[in] times: Measurement times in milliseconds.
//! \param[in] metricIndex: Index of performance measurement metrics
//!
//! \return The percentile of a performance metric
float percentile(float percentage, TimingMetrics const& times, int32_t metricIndex)
{
    int32_t const all = static_cast<int32_t>(times.size());
    int32_t const exclude = static_cast<int32_t>((1 - percentage / 100.F) * all);
    if (times.empty())
    {
        return std::numeric_limits<float>::infinity();
    }
    if (percentage < 0.F || percentage > 100.F)
    {
        throw std::runtime_error("percentile is not in [0, 100]!");
    }
    return times[std::max(all - 1 - exclude, 0)][metricIndex];
}

//!
//! \brief Find coefficient of variance (which is std / mean) in a sorted sequence of timings
//!
//! \param[in] times: Measurement times in milliseconds.
//! \param[in] metricIndex: Index of performance measurement metrics
//! \param[in] mean: Mean of the performance measurement metrics
//!
//! \return The coefficient of variance
float findCoeffOfVariance(TimingMetrics const& times, int32_t metricIndex, float mean)
{
    if (times.empty())
    {
        return 0.F;
    }

    if (mean == 0.F)
    {
        return std::numeric_limits<float>::infinity();
    }

    auto const metricAccumulator = [metricIndex, mean](float acc, TimingMetric const& a) {
        float const diff = a[metricIndex] - mean;
        return acc + diff * diff;
    };

    float const variance = std::accumulate(times.begin(), times.end(), 0.F, metricAccumulator) / times.size();

    return std::sqrt(variance) / mean * 100.F;
}

//!
//! \brief Get the performance result of a performance metric
//!
//! \param[in] times: Measurement times in milliseconds.
//! \param[in] metricIndex: Index of performance measurement metrics
//! \param[in] percent: Percentile to get
//!
//! \return The performance result of a performance metric
SafePerformanceResult getSafePerformanceResult(TimingMetrics const& times, int32_t metricIndex, float percent)
{
    auto const ascendingSorter
        = [metricIndex](TimingMetric& a, TimingMetric& b) { return a[metricIndex] < b[metricIndex]; };
    // make a copy w/o const qualifier
    TimingMetrics newTimes = times;
    std::sort(newTimes.begin(), newTimes.end(), ascendingSorter);
    SafePerformanceResult result;
    result.min = newTimes[0][metricIndex];
    result.max = newTimes[newTimes.size() - 1][metricIndex];
    result.mean = std::accumulate(newTimes.begin(), newTimes.end(), 0.F,
                      [metricIndex](float acc, TimingMetric& a) { return acc + a[metricIndex]; })
        / newTimes.size();
    size_t const medianIndex = newTimes.size() / 2ULL;
    result.median = newTimes.size() % 2ULL
        ? newTimes[medianIndex][metricIndex]
        : (newTimes[medianIndex][metricIndex] + newTimes[medianIndex + 1ULL][metricIndex]) / 2.0f;
    result.percentile = percentile(percent, newTimes, metricIndex);
    result.coeffVar = findCoeffOfVariance(newTimes, metricIndex, result.mean);
    return result;
}

nvinfer2::safe::TypedArray createTypedArray(
    void* const ptr, DataType type, uint64_t bufferSize, nvinfer2::safe::ISafeRecorder& recorder)
{
    switch (type)
    {
    case DataType::kFLOAT: return nvinfer2::safe::TypedArray(static_cast<float*>(ptr), bufferSize);
    case DataType::kHALF: return nvinfer2::safe::TypedArray(static_cast<nvinfer2::safe::half_t*>(ptr), bufferSize);
    case DataType::kINT64: return nvinfer2::safe::TypedArray(static_cast<int64_t*>(ptr), bufferSize);
    case DataType::kINT32: return nvinfer2::safe::TypedArray(static_cast<int32_t*>(ptr), bufferSize);
    case DataType::kINT8: return nvinfer2::safe::TypedArray(static_cast<int8_t*>(ptr), bufferSize);
    case DataType::kBOOL: return nvinfer2::safe::TypedArray(static_cast<bool*>(ptr), bufferSize);
    default:
    {
        safeLogError(recorder, "Invalid tensor DataType encountered.");
        return nvinfer2::safe::TypedArray{};
    }
    }
}

//!
//! \brief Allocate memory and memset it to zero using safe CUDA-compatible APIs.
//!
//! \param[in] sizeInBytes The size of memory to allocate in bytes
//! \param[in] recorder The safe recorder for error logging and API calls
//!
//! \return ScopedSafeMemory object containing the allocated zeroed memory
//!
ScopedSafeMemory allocateAndMemset(uint64_t sizeInBytes, nvinfer2::safe::ISafeRecorder& recorder)
{
    ScopedSafeMemory deviceBuf(sizeInBytes, kDEFAULT_ALIGNMENT, nvinfer2::safe::MemoryPlacement::kGPU,
        nvinfer2::safe::MemoryUsage::kIOTENSOR, recorder);
    if (!deviceBuf)
    {
        return deviceBuf;
    }

    // Use async memset and synchronize (required for QNX safety builds where cudaMemset is not available)
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), recorder);
    CUDA_CALL(cudaMemsetAsync(deviceBuf.get(), 0, sizeInBytes, stream), recorder);
    CUDA_CALL(cudaStreamSynchronize(stream), recorder);
    CUDA_CALL(cudaStreamDestroy(stream), recorder);
    return deviceBuf;
}

//!
//! \brief Load data from a binary file into a pre-allocated buffer.
//!
//! This function reads data from a binary file and validates the file size matches
//! the expected buffer size. It provides detailed error reporting for file operations.
//!
//! \param[in] fileName The path to the binary file to load
//! \param[out] buffer Pointer to the buffer to load data into
//! \param[in] sizeInBytes The expected size of the file and buffer in bytes
//! \param[in] recorder The safe recorder for error logging
//!
//! \return True if file was loaded successfully, false otherwise
//!
bool loadDataFromFile(
    std::string const& fileName, void* buffer, uint64_t sizeInBytes, nvinfer2::safe::ISafeRecorder& recorder)
{
    std::ifstream file(fileName, std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
        safeLogError(recorder, "Cannot open input file: " + fileName);
        return false;
    }

    file.seekg(0, std::ios::end);
    int64_t fileSize = static_cast<int64_t>(file.tellg());
    if (fileSize != static_cast<int64_t>(sizeInBytes))
    {
        file.close();
        std::ostringstream msg;
        msg << "File size mismatch for " << fileName << ". Expected: " << sizeInBytes << " bytes, got: " << fileSize
            << " bytes";
        safeLogError(recorder, msg.str());
        return false;
    }

    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(buffer), sizeInBytes);
    size_t const nbBytesRead = file.gcount();
    file.close();

    if (nbBytesRead != sizeInBytes)
    {
        std::ostringstream msg;
        msg << "Failed to read complete file " << fileName << ". Expected: " << sizeInBytes
            << " bytes, read: " << nbBytesRead << " bytes";
        safeLogError(recorder, msg.str());
        return false;
    }

    return true;
}

//!
//! \brief Allocate memory and load data from file using safe CUDA-compatible APIs.
//!
//! This function allocates GPU memory and loads data from a binary file into it.
//! It performs file size validation and uses RAII for automatic memory cleanup.
//!
//! \param[in] sizeInBytes The size of memory to allocate in bytes
//! \param[in] fileName The path to the binary file to load
//! \param[in] recorder The safe recorder for error logging and API calls
//!
//! \return ScopedSafeMemory object containing the loaded data, or an invalid object on failure
//!
ScopedSafeMemory allocateAndLoadFromFile(
    uint64_t sizeInBytes, std::string const& fileName, nvinfer2::safe::ISafeRecorder& recorder)
{
    // Allocate pinned host memory for temporary storage with RAII
    ScopedSafeMemory hostBuf(sizeInBytes, kDEFAULT_ALIGNMENT, nvinfer2::safe::MemoryPlacement::kCPU_PINNED,
        nvinfer2::safe::MemoryUsage::kIOTENSOR, recorder);
    if (!hostBuf)
    {
        safeLogError(recorder, "Failed to allocate host memory for input file: " + fileName);
        return ScopedSafeMemory(0, kDEFAULT_ALIGNMENT, nvinfer2::safe::MemoryPlacement::kGPU,
            nvinfer2::safe::MemoryUsage::kIOTENSOR, recorder);
    }

    // Load data from file into host buffer
    if (!loadDataFromFile(fileName, hostBuf.get(), sizeInBytes, recorder))
    {
        return ScopedSafeMemory(0, kDEFAULT_ALIGNMENT, nvinfer2::safe::MemoryPlacement::kGPU,
            nvinfer2::safe::MemoryUsage::kIOTENSOR, recorder);
    }

    // Allocate device memory with RAII and copy data
    ScopedSafeMemory deviceBuf(sizeInBytes, kDEFAULT_ALIGNMENT, nvinfer2::safe::MemoryPlacement::kGPU,
        nvinfer2::safe::MemoryUsage::kIOTENSOR, recorder);
    if (!deviceBuf)
    {
        safeLogError(recorder, "Failed to allocate device memory for input file: " + fileName);
        return ScopedSafeMemory(0, kDEFAULT_ALIGNMENT, nvinfer2::safe::MemoryPlacement::kGPU,
            nvinfer2::safe::MemoryUsage::kIOTENSOR, recorder);
    }

    // Use async copy and synchronize (required for QNX safety builds where cudaMemcpy may not be available)
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), recorder);
    CUDA_CALL(cudaMemcpyAsync(deviceBuf.get(), hostBuf.get(), sizeInBytes, cudaMemcpyHostToDevice, stream), recorder);
    CUDA_CALL(cudaStreamSynchronize(stream), recorder);
    CUDA_CALL(cudaStreamDestroy(stream), recorder);

    return deviceBuf;
}

//!
//! \brief Parse loadInputs string in the format "name1:file1,name2:file2,..."
//!
//! This function parses a comma-separated string of input tensor name to file path mappings.
//! It supports quoted tensor names (e.g., 'input_name':file.bin) to handle names with special characters.
//! The format follows the same convention as standard trtexec --loadInputs parameter.
//!
//! \param[in] loadInputsStr The input string to parse in format "name1:file1,name2:file2,..."
//!
//! \return A map containing tensor names as keys and file paths as values. Returns empty map if input is invalid.
//!
std::unordered_map<std::string, std::string> parseLoadInputs(std::string const& loadInputsStr)
{
    std::unordered_map<std::string, std::string> result;
    if (loadInputsStr.empty())
    {
        return result;
    }

    // Split by comma
    std::stringstream ss(loadInputsStr);
    std::string pair;
    while (std::getline(ss, pair, ','))
    {
        // Handle quoted names (e.g., 'input_name':file.bin)
        std::string tensorName;
        std::string fileName;

        size_t colonPos = pair.find_last_of(':');
        if (colonPos == std::string::npos)
        {
            safeLogDebug(*gSafeRecorder,
                "Invalid input pair skipped: \"" + pair
                    + "\" (reason: no ':' separator found - expected format 'tensorName:fileName')");
            continue; // Skip invalid pairs
        }

        tensorName = pair.substr(0, colonPos);
        fileName = pair.substr(colonPos + 1);

        // Remove quotes if present
        if (tensorName.size() >= 2 && tensorName.front() == '\'' && tensorName.back() == '\'')
        {
            tensorName = tensorName.substr(1, tensorName.size() - 2);
        }

        result[tensorName] = fileName;
    }

    return result;
}

bool parseSafetyPluginLibrary(
    std::string const& arg, std::string const& name, SafetyPluginLibraryArgument& pluginLibArgs)
{
    std::string const pattern = "--" + name + "=";
    bool const matched = !arg.compare(0ULL, pattern.size(), pattern);
    bool status{false};
    if (matched)
    {
        std::string const optionStr = arg.substr(pattern.size());
        status = parseSafetyPluginArgument(optionStr, pluginLibArgs);
        if (!status)
        {
            safeLogError(*gSafeRecorder, "Unable to parse safety plugin library argument: " + arg);
        }
    }
    return matched && status;
}

// Use template to allow volume for either nvinfer1::Dims or nvinfer2::safe::PhysicalDims
template<typename TDims>
int64_t volume(TDims const& dims, TDims const& strides, uint64_t bytesPerComponent)
{
    if (dims.nbDims == 0 || strides.nbDims == 0)
    {
        return 0;
    }
    // product of all tensor dimensions
    int64_t volume = 1;
    for (int64_t i = 0; i < dims.nbDims; i++)
    {
        if (dims.d[i] < 1)
        {
            return 0;
        }
        SAFE_ASSERT(volume <= INT64_MAX / dims.d[i]);
        volume *= dims.d[i];
    }
    // real tensor volume is the max between the product of all dimensions and the dims.n * strides.n
    SAFE_ASSERT(dims.d[0] <= INT64_MAX / strides.d[0]);
    volume = std::max(volume, dims.d[0] * strides.d[0]);
    return volume * bytesPerComponent;
}

} // anonymous namespace

//!
//! \brief This function parses arguments specific to the sample
//!
bool parseSafeExecArgs(SafeExecArgs& args, int32_t argc, char* argv[])
{
    std::string val;
    SafetyPluginLibraryArgument pluginArg;

    bool hasRequired{false};
    safeLogInfo(*gSafeRecorder, "Parsing input arguments...");
    for (int32_t i = 1; i < argc; ++i)
    {
        if (parseString(argv[i], "loadEngine", val))
        {
            args.engineFile = val;
            hasRequired = true;
        }
        else if (parseSafetyPluginLibrary(argv[i], "safetyPlugins", pluginArg))
        {
            args.pluginLibraries.emplace_back(std::move(pluginArg));
        }
        else if (parseString(argv[i], "iterations", val))
        {
            args.iterations = stoi(val);
        }
        else if (parseString(argv[i], "avgRuns", val))
        {
            args.avgRuns = stoi(val);
        }
        else if (parseString(argv[i], "warmUp", val))
        {
            args.warmUp = stoi(val);
        }
        else if (parseString(argv[i], "device", val))
        {
            args.device = stoi(val);
        }
        else if (parseString(argv[i], "percentile", val))
        {
            args.percentile = stof(val);
        }
        else if (parseString(argv[i], "idleTime", val))
        {
            args.idle = stof(val);
        }
        else if (parseString(argv[i], "duration", val))
        {
            args.duration = stof(val);
        }
        else if (parseString(argv[i], "sleepTime", val))
        {
            args.sleep = stof(val);
        }
        else if (parseBool(argv[i], "spin"))
        {
            args.spin = true;
        }
        else if (parseBool(argv[i], "verbose"))
        {
            args.verbose = true;
        }
        else if (parseBool(argv[i], "debug"))
        {
            args.debug = true;
        }
        else if (parseBool(argv[i], "help") || parseBool(argv[i], "h", true))
        {
            args.help = true;
        }
        else if (parseBool(argv[i], "useCudaGraph"))
        {
            args.useCudaGraph = true;
        }
        else if (parseString(argv[i], "threads", val))
        {
            args.threads = stoi(val);
        }
        else if (parseBool(argv[i], "useScratch"))
        {
            args.useScratchMemory = true;
        }
        else if (parseBool(argv[i], "separateProfileRun"))
        {
            args.separateProfileRun = true;
        }
        else if (parseString(argv[i], "ioProfileId", val))
        {
            // Select I/O profile index for the TRTGraph
            args.ioProfile = std::stoll(val);
            if (args.ioProfile < 0)
            {
                safeLogError(*gSafeRecorder, "Invalid ioProfileId (must be >= 0): " + val);
                return false;
            }
        }
        else if (parseString(argv[i], "loadInputs", val))
        {
            args.loadInputs = parseLoadInputs(val);
            if (!val.empty() && args.loadInputs.empty())
            {
                safeLogError(*gSafeRecorder, "Invalid loadInputs format: " + val);
                return false;
            }
        }
        else
        {
            safeLogError(*gSafeRecorder, "Invalid Argument: " + std::string(argv[i]));
            return false;
        }
    }
    if (!hasRequired && !args.help)
    {
        safeLogError(*gSafeRecorder, "Engine file is required.");
        return false;
    }
    return true;
}

//!
//! \brief Prints the help information for running this sample.
//!
void printHelpInfo()
{
    std::cout << "Usage: ./trtexec_safe [--loadEngine=<path to engine file>]\n";
    std::cout << "Mandatory params:\n";
    std::cout << "  --loadEngine=<file path>   Load the serialized engine from the file.\n";
    std::cout << "General optional params:\n";
    std::cout << "  --help or -h               Display help information\n";
    std::cout << "  --verbose                  Use verbose logging\n";
    std::cout << "  --debug                    Use debug logging\n";
    std::cout << "  --useScratch               Use separately allocated scratch memory\n";
    std::cout << "  --safetyPlugins=spec       Load safety plugin libraries (can be specified multiple times)\n";
    std::cout << "                             Plugin spec ::= pluginLib[pluginNamespace::pluginName],[...]\n";
    std::cout << "                             Example: --safetyPlugins=myPlugin.so[MyNamespace::MyPlugin]\n";
    std::cout << "  --loadInputs=spec          Load input values from files (default = generate zero inputs). Input "
                 "names can be wrapped with single quotes (ex: 'Input:0')\n";
    std::cout << "                             Input values spec ::= Ival[\",\"spec]\n";
    std::cout << "                                          Ival ::= name\":\"file\n";
    std::cout << "                             Example: --loadInputs=\"input1\":data1.bin,\"input2\":data2.bin\n";
    std::cout << "Perf measurement params:\n";
    std::cout << "  --device=N                 Set cuda device to N (default = 0)\n";
    std::cout << "  --threads=N                Run in N threads (default = 1)\n";
    std::cout << "  --spin                     Actively wait for work completion. This option may decrease "
                 "multi-process synchronization time at the cost of additional CPU usage. (default = false)\n";
    std::cout << "  --iterations=N             Run N iterations (default = 10)\n";
    std::cout << "  --avgRuns=N                Set avgRuns to N - perf is measured as an average of avgRuns (default "
                 "= 10)\n";
    std::cout << "  --warmUp=N                 Run N iterations before actual perf measurement (default = 1)\n";
    std::cout << "  --idleTime=N               Sleep N milliseconds between two continuous iterations (default = 0)\n";
    std::cout << "  --percentile=P             For each iteration, report the percentile time at P percentage "
                 "(0<=P<=100, with 0 representing min, and 100 representing max; default = 99%)\n";
    std::cout << "  --useCudaGraph             Use CUDA graph to capture engine execution and then launch inference "
                 "(default = disabled)\n";
    std::cout << "  --duration=N               Run performance measurements for at least N seconds wallclock time "
                 "(default = 3.0s)\n";
    std::cout << "  --sleepTime=N              Delay inference start with a gap of N milliseconds between launch "
                 " and compute (default = 2)\n";
    std::cout << "  --separateProfileRun       Perform safe profile run separately (default = disabled)\n";
    std::cout << "I/O profile params:\n";
    std::cout << "  --ioProfileId=N            Select the I/O profile index to use (default = 0)\n";
}

void registerSafetyPlugins(nvinfer2::safe::ISafeRecorder& recorder, SafetyPluginArguments const& pluginArgs)
{
    std::string const pluginGetterSymbolName{"getSafetyPluginCreator"};
    auto const safePluginRegistry = nvinfer2::safe::getSafePluginRegistry(recorder);
    if (!safePluginRegistry)
    {
        safeLogError(recorder, "Safe Plugin Registry is not found.");
        return;
    }

    for (auto const& pluginArg : pluginArgs)
    {
        void* libraryHandle = safeLoadLibrary(pluginArg.libraryName);
        if (libraryHandle == nullptr)
        {
            safeLogError(recorder, "Not able to load plugin library: " + pluginArg.libraryName);
            continue;
        }

        typedef IPluginCreatorInterface* (*getPluginCreatorFn)(char const*, char const*);
        auto pluginCreatorGetter
            = reinterpret_cast<getPluginCreatorFn>(dlsym(libraryHandle, pluginGetterSymbolName.c_str()));
        if (pluginCreatorGetter == nullptr)
        {
            safeLogError(
                recorder, "Cannot find plugin creator getter symbol from plugin library: " + pluginArg.libraryName);
            safeLogError(recorder, "Please ensure interface function is correctly implemented and exported.");
            continue;
        }

        for (auto const& pluginAttr : pluginArg.pluginAttrs)
        {
            auto pluginCreator = static_cast<IPluginCreatorInterface*>(
                pluginCreatorGetter(pluginAttr.pluginNamespace.c_str(), pluginAttr.pluginName.c_str()));
            if (pluginCreator == nullptr)
            {
                safeLogWarning(recorder,
                    "Plugin interface getSafetyPluginCreator return nullptr for " + pluginAttr.pluginNamespace
                        + "::" + pluginAttr.pluginName + " in the safety plugin library: " + pluginArg.libraryName);
                safeLogWarning(recorder,
                    "Please ensure interface function is implemented correctly and plugin name/namespace is matched.");
                continue;
            }
            safeLogInfo(recorder, "Registering " + pluginAttr.pluginNamespace + "::" + pluginAttr.pluginName);
            ErrorCode errorCode
                = safePluginRegistry->registerCreator(*pluginCreator, pluginAttr.pluginNamespace.c_str(), recorder);
            if (errorCode != ErrorCode::kSUCCESS)
            {
                safeLogWarning(recorder,
                    "Failed to register safety plugin " + pluginAttr.pluginNamespace + "::" + pluginAttr.pluginName);
                if (errorCode == ErrorCode::kINVALID_ARGUMENT)
                {
                    safeLogWarning(recorder,
                        "Is getPluginName/getPluginNamespace/getPluginVersion interface implemented and return "
                        "non-nullptr?");
                }
            }
        }
    }
}

//!
//! \brief Load a prebuilt TensorRT safe engine.
//!
std::vector<char> loadEngine(std::string const& engineFile)
{
    std::string const& filename = engineFile;
    std::vector<char> modelBuffer;
    std::ifstream file(filename, std::ios::binary);
    if (!file.good())
    {
        safeLogError(*gSafeRecorder, "Could not open input engine file or file is empty. File name: " + filename);
        return modelBuffer;
    }
    file.seekg(0, std::ifstream::end);
    auto size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    modelBuffer.resize(size);
    file.read(modelBuffer.data(), size);
    file.close();
    return modelBuffer;
}

//!
//! \brief Common helper function to set up tensor buffer with optional file loading.
//!
//! This function handles the common logic for setting up tensor buffers, including
//! memory allocation, optional file loading, and tensor address assignment.
//!
//! \param[in] graph Pointer to the TRT graph
//! \param[in] recorder The safe recorder for error logging and API calls
//! \param[in] desc The tensor descriptor containing size and memory placement info
//! \param[in] tensorName The name of the tensor for logging and loadInputs lookup
//! \param[in] loadInputs Optional map of tensor names to file paths for loading custom input data
//!
//! \return ScopedSafeMemory object containing the allocated tensor buffer, or an invalid object on failure
//!
ScopedSafeMemory setupTensorBuffer(nvinfer2::safe::ITRTGraph* graph, nvinfer2::safe::ISafeRecorder& recorder,
    nvinfer2::safe::TensorDescriptor const& desc, std::string const& tensorName,
    std::unordered_map<std::string, std::string> const& loadInputs)
{
    std::stringstream ss;
    bool const onGpu = desc.memPlacement == nvinfer2::safe::MemoryPlacement::kGPU
        || desc.memPlacement == nvinfer2::safe::MemoryPlacement::kNONE;

    // Calculate expected size using volume calculation from upstream.
    // Tensor volume could be zero if using MSS engine build.
    uint64_t const expectedSize
        = std::max(static_cast<uint64_t>(volume(desc.shape, desc.stride, desc.bytesPerComponent)), desc.sizeInBytes);

    // Check if we have input data to load for this tensor
    auto const inputIt = loadInputs.find(tensorName);
    bool const hasInputFile = (inputIt != loadInputs.end() && !tensorName.empty());

    if (onGpu)
    {
        ScopedSafeMemory deviceBuf = hasInputFile ? allocateAndLoadFromFile(expectedSize, inputIt->second, recorder)
                                                  : allocateAndMemset(expectedSize, recorder);

        if (hasInputFile)
        {
            ss << "Loaded input data from " << inputIt->second << " for tensor " << tensorName;
            safeLogInfo(recorder, ss.str());
            ss.str("");
        }

        ss << "Set address of " << tensorName << " on device at " << std::hex << (uint64_t) deviceBuf.get() << std::dec;
        safeLogInfo(recorder, ss.str());
        return deviceBuf;
    }
    else if (desc.memPlacement == nvinfer2::safe::MemoryPlacement::kCPU)
    {
        ScopedSafeMemory hostBuf(expectedSize, kDEFAULT_ALIGNMENT, nvinfer2::safe::MemoryPlacement::kCPU,
            nvinfer2::safe::MemoryUsage::kIOTENSOR, recorder);
        if (!hostBuf)
        {
            safeLogError(recorder, "Failed to allocate host memory for tensor: " + tensorName);
            return ScopedSafeMemory(0, kDEFAULT_ALIGNMENT, nvinfer2::safe::MemoryPlacement::kCPU,
                nvinfer2::safe::MemoryUsage::kIOTENSOR, recorder);
        }

        if (hasInputFile)
        {
            // Load data from file for CPU tensors
            if (!loadDataFromFile(inputIt->second, hostBuf.get(), expectedSize, recorder))
            {
                safeLogError(recorder, "Failed to load input file for tensor: " + tensorName);
                return ScopedSafeMemory(0, kDEFAULT_ALIGNMENT, nvinfer2::safe::MemoryPlacement::kCPU,
                    nvinfer2::safe::MemoryUsage::kIOTENSOR, recorder);
            }
            ss << "Loaded input data from " << inputIt->second << " for tensor " << tensorName;
            safeLogInfo(recorder, ss.str());
            ss.str("");
        }
        else
        {
            memset(hostBuf.get(), 0, expectedSize);
        }

        ss << "Set address of " << tensorName << " on host at " << std::hex << (uint64_t) hostBuf.get() << std::dec;
        safeLogInfo(recorder, ss.str());
        return hostBuf;
    }
    else
    {
        safeLogError(recorder, "Invalid memory placement for tensor: " + tensorName);
        return ScopedSafeMemory(0, kDEFAULT_ALIGNMENT, nvinfer2::safe::MemoryPlacement::kGPU,
            nvinfer2::safe::MemoryUsage::kIOTENSOR, recorder);
    }
}

//!
//! \brief Set I/O tensor buffer with optional input file loading.
//!
//! This function allocates memory for a tensor and optionally loads data from a file.
//! If the tensor name is found in the loadInputs map, it loads data from the specified file.
//! Otherwise, it initializes the tensor with zeros. Supports both GPU and CPU memory placement.
//!
//! \param[in] graph Pointer to the TRT graph
//! \param[in] recorder The safe recorder for error logging and API calls
//! \param[in] tensorName The name of the tensor to set up
//! \param[in] loadInputs Optional map of tensor names to file paths for loading custom input data
//!
//! \return ScopedSafeMemory object containing the allocated tensor buffer
//!
ScopedSafeMemory setTensorBuffer(nvinfer2::safe::ITRTGraph* graph, nvinfer2::safe::ISafeRecorder& recorder,
    std::string const& tensorName, std::unordered_map<std::string, std::string> const& loadInputs = {})
{
    nvinfer2::safe::TensorDescriptor desc;
    SAFE_API_CALL(graph->getIOTensorDescriptor(desc, tensorName.c_str()), recorder);

    // Use common helper to set up the tensor buffer
    ScopedSafeMemory tensorBuffer = setupTensorBuffer(graph, recorder, desc, tensorName, loadInputs);
    if (!tensorBuffer)
    {
        return ScopedSafeMemory(0, kDEFAULT_ALIGNMENT, nvinfer2::safe::MemoryPlacement::kGPU,
            nvinfer2::safe::MemoryUsage::kIOTENSOR, recorder);
    }

    // Tensor volume could be zero if using MSS engine build.
    uint64_t expectedSize
        = std::max(static_cast<uint64_t>(volume(desc.shape, desc.stride, desc.bytesPerComponent)), desc.sizeInBytes);

    // Set the tensor address in the graph
    nvinfer2::safe::TypedArray const tensor
        = createTypedArray(tensorBuffer.get(), desc.dataType, expectedSize, recorder);
    SAFE_API_CALL(graph->setIOTensorAddress(tensorName.c_str(), tensor), recorder);

    return tensorBuffer;
}

//! \brief Function to CUDA Graph capture
bool graphCapture(cudaStream_t stream, TrtCudaGraphSafe& cudaGraph, nvinfer2::safe::ITRTGraph* graph,
    nvinfer2::safe::ISafeRecorder& recorder)
{
    // Avoid capturing initialization calls by executing the enqueue function at least
    // once before starting CUDA graph capture.

    ErrorCode executeRes = graph->executeAsync(stream);
    ErrorCode syncRes = graph->sync();

    if (executeRes != nvinfer1::ErrorCode::kSUCCESS || syncRes != nvinfer1::ErrorCode::kSUCCESS)
    {
        safeLogError(recorder, "The enqueue function before starting CUDA graph capture failed.");
        return false;
    }
    static_cast<void>(cudaStreamSynchronize(stream));
    cudaGraph.beginCapture(stream);

    // The built TRT engine may contain operations that are not permitted under CUDA graph capture mode.
    // When the stream is capturing, the enqueue call may return false if the current CUDA graph capture fails.

    executeRes = graph->executeAsync(stream);

    if (executeRes == nvinfer1::ErrorCode::kSUCCESS)
    {
        cudaGraph.endCapture(stream);
    }
    else
    {
        cudaGraph.endCaptureOnError(stream);
        // Ensure any CUDA error has been cleaned up.
        CUDA_CHECK(cudaGetLastError());
        safeLogError(recorder,
            "The built TensorRT engine contains operations that are not permitted under CUDA graph capture mode.");
        return false;
    }
    return true;
}

//!
//! \brief Thread task to run graph execution with optional input file loading.
//!
//! This function sets up tensor buffers (optionally loading from files specified in args.loadInputs),
//! executes the graph for the specified number of iterations, and measures performance.
//! It handles both profiling runs and regular inference runs.
//!
//! \param[in] args The execution arguments containing loadInputs map and other configuration
//! \param[in] graph Pointer to the TRT graph to execute
//! \param[in] recorder Pointer to the safe recorder for error logging and API calls
//! \param[in] isProfileRun Whether this is a profiling run or regular inference
//!
//! \return True if execution completed successfully, false otherwise
//!
bool task(SafeExecArgs const& args, nvinfer2::safe::ITRTGraph* graph, nvinfer2::safe::ISafeRecorder* recorder,
    bool isProfileRun)
{
    int64_t nbIOs{};
    SAFE_API_CALL(graph->getNbIOTensors(nbIOs), *recorder);
    std::vector<ScopedSafeMemory> buffers;
    buffers.reserve(nbIOs);
    // Set input tensor values
    for (int64_t i = 0; i < nbIOs; ++i)
    {
        char const* tensor;
        SAFE_API_CALL(graph->getIOTensorName(tensor, i), *recorder);
        buffers.emplace_back(setTensorBuffer(graph, *recorder, tensor, args.loadInputs));
    }
    cudaEvent_t inputConsumedEvent;
    cudaEventCreate(&inputConsumedEvent);
    SAFE_API_CALL(graph->setInputConsumedEvent(inputConsumedEvent), *recorder);

    cudaEvent_t retrievedEvent;
    SAFE_API_CALL(graph->getInputConsumedEvent(retrievedEvent), *recorder);
    SAFE_ASSERT(retrievedEvent != nullptr);
    cudaEventSynchronize(retrievedEvent);

    // Initialize main stream
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), *recorder);

    uint32_t const cudaEventFlags = args.spin ? cudaEventDefault : cudaEventBlockingSync;
    cudaEvent_t gpuStart;
    CUDA_CALL(cudaEventCreateWithFlags(&gpuStart, cudaEventFlags), *recorder);
    CUDA_CALL(cudaEventRecord(gpuStart, stream), *recorder);

    float absStartTime{0.0f};
    float absEndTime{0.0f};
    if (!isProfileRun)
    {
        safeLogInfo(*recorder, "Starting inference...");
    }
    // Warm up
    for (int32_t i = 0; i < args.warmUp; i++)
    {
        SAFE_API_CALL(graph->executeAsync(stream), *recorder);
        // Synchronize the network
        SAFE_API_CALL(graph->sync(), *recorder);
    }
    CUDA_CALL(cudaStreamSynchronize(stream), *recorder);
    if (!isProfileRun)
    {
        safeLogInfo(*recorder, "Warmup completed.");
        safeLogInfo(*recorder, ""); // empty line
        safeLogInfo(*recorder, "=== Trace details ===");
    }

    // Create cuda events for profiling
    cudaEvent_t startEvent, endEvent;
    CUDA_CALL(cudaEventCreateWithFlags(&startEvent, cudaEventFlags), *recorder);
    CUDA_CALL(cudaEventCreateWithFlags(&endEvent, cudaEventFlags), *recorder);
    cudaEvent_t syncEvent;
    CUDA_CALL(cudaEventCreateWithFlags(&syncEvent, cudaEventDisableTiming), *recorder);

    // Do inference
    auto const nbAvgRuns = args.avgRuns;
    auto const nbIterations = args.iterations;
    // GPU, host and enqueue times
    TimingMetrics totalTimes;
    using floatDurationMS = std::chrono::duration<float, std::milli>;
    floatDurationMS const maxDurationMs = floatDurationMS(args.duration * 1000);
    floatDurationMS durationMs{0};

    for (int32_t i = 0; i < nbIterations || durationMs.count() < maxDurationMs.count(); i++)
    {
        TrtCudaGraphSafe cudaGraph;

        float totalGpuTime{0.F};
        float totalHostTime{0.F};
        float totalEnqueueTime{0.F};

        if (args.useCudaGraph && !isProfileRun)
        {
            if (!graphCapture(stream, cudaGraph, graph, *recorder))
            {
                safeLogError(*recorder, "Failed to capture graph.");
                return false;
            }
        }

        for (int32_t j = 0; j < nbAvgRuns; j++)
        {
            auto const startTime = std::chrono::high_resolution_clock::now();
            if (isProfileRun)
            {
                if (graph->executeAsync(stream) != ErrorCode::kSUCCESS)
                {
                    safeLogError(*recorder, "Failed to run executeAsync during average runs.");
                    return false;
                }
                SAFE_API_CALL(graph->sync(), *recorder);
                auto const endTime = std::chrono::high_resolution_clock::now();
                durationMs += floatDurationMS(endTime - startTime);
                continue;
            }
            CUDA_CHECK(delayStream(stream, args.sleep));
            CUDA_CALL(cudaEventRecord(startEvent, stream), *recorder);
            CUDA_CALL(cudaStreamWaitEvent(stream, startEvent, 0), *recorder);
            if (args.useCudaGraph)
            {
                if (!cudaGraph.launch(stream))
                {
                    safeLogError(*recorder, "Failed to launch graph.");
                    return false;
                }
            }
            else
            {
                if (graph->executeAsync(stream) != ErrorCode::kSUCCESS)
                {
                    safeLogError(*recorder, "Failed to run executeAsync during average runs.");
                    return false;
                }
            }
            CUDA_CALL(cudaEventRecord(syncEvent, stream), *recorder);
            CUDA_CALL(cudaStreamWaitEvent(stream, syncEvent, 0), *recorder);

            auto const enqueueEndTime = std::chrono::high_resolution_clock::now();

            CUDA_CALL(cudaEventRecord(endEvent, stream), *recorder);
            CUDA_CALL(cudaEventSynchronize(endEvent), *recorder);

            if (i == 0 && j == 0)
            {
                CUDA_CALL(cudaEventElapsedTime(&absStartTime, gpuStart, startEvent), *recorder);
            }
            if ((i == nbIterations - 1) && (j == nbAvgRuns - 1))
            {
                CUDA_CALL(cudaEventElapsedTime(&absEndTime, gpuStart, endEvent), *recorder);
            }
            auto const endTime = std::chrono::high_resolution_clock::now();

            float gpuTime{0.F};
            CUDA_CALL(cudaEventElapsedTime(&gpuTime, startEvent, endEvent), *recorder);
            auto const enqueueTime = std::chrono::duration<float, std::milli>(enqueueEndTime - startTime).count();
            auto const hostTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();

            durationMs += floatDurationMS(hostTime);
            totalGpuTime += gpuTime;
            totalHostTime += hostTime;
            totalEnqueueTime += enqueueTime;

            // Mimic waiting for user input data (default = 0)
            std::this_thread::sleep_for(std::chrono::duration<float, std::milli>(args.idle));
        }

        if (isProfileRun)
        {
            continue;
        }

        auto const avgGpuTime = totalGpuTime / nbAvgRuns;
        auto const avgHostTime = totalHostTime / nbAvgRuns;
        auto const avgEnqueueTime = totalEnqueueTime / nbAvgRuns;
        totalTimes.push_back({avgGpuTime, avgHostTime, avgEnqueueTime});

        std::stringstream ss;
        ss << "Average over " << nbAvgRuns << " runs - GPU latency: " << avgGpuTime
           << " ms - Host latency: " << avgHostTime << " ms (enqueue " << avgEnqueueTime << " ms)";
        safeLogInfo(*recorder, ss.str());
    }

    if (!isProfileRun)
    {
        std::stringstream ss;

        // Sort GPU times
        std::sort(totalTimes.begin(), totalTimes.end(),
            [](TimingMetric const& a, TimingMetric const& b) { return a[0] < b[0]; });
        auto const gpuTimeResult = getSafePerformanceResult(totalTimes, 0, args.percentile);
        auto const hostTimeResult = getSafePerformanceResult(totalTimes, 1, args.percentile);
        auto const enqueueTimeResult = getSafePerformanceResult(totalTimes, 2, args.percentile);

        auto const totalWallTime = absEndTime - absStartTime;

        // Print final profiling result
        safeLogInfo(*recorder, ""); // empty line
        safeLogInfo(*recorder, "=== Performance summary ===");
        ss << "Total throughput: " << nbAvgRuns * nbIterations / totalWallTime * 1000 << " qps";
        safeLogInfo(*recorder, ss.str());
        ss.str("");
        ss << "Host Time: min = " << hostTimeResult.min << " ms, max = " << hostTimeResult.max
           << " ms, mean = " << hostTimeResult.mean << " ms, median = " << hostTimeResult.median << " ms,"
           << " percentile(" << args.percentile << "%) = " << hostTimeResult.percentile << " ms";
        safeLogInfo(*recorder, ss.str());
        ss.str("");
        ss << "Enqueue Time: min = " << enqueueTimeResult.min << " ms, max = " << enqueueTimeResult.max
           << " ms, mean = " << enqueueTimeResult.mean << " ms, median = " << enqueueTimeResult.median << " ms,"
           << " percentile(" << args.percentile << "%) = " << enqueueTimeResult.percentile << " ms";
        safeLogInfo(*recorder, ss.str());
        ss.str("");
        ss << "GPU Compute Time:  min = " << gpuTimeResult.min << " ms, max = " << gpuTimeResult.max
           << " ms, mean = " << gpuTimeResult.mean << " ms, median = " << gpuTimeResult.median << " ms,"
           << " percentile(" << args.percentile << "%) = " << gpuTimeResult.percentile << " ms";
        safeLogInfo(*recorder, ss.str());
        ss.str("");
        // Report warnings if the GPU Compute Time is unstable.
        constexpr float kUNSTABLE_PERF_REPORTING_THRESHOLD{1.0F};
        if (gpuTimeResult.coeffVar > kUNSTABLE_PERF_REPORTING_THRESHOLD)
        {
            ss << "* GPU compute time is unstable, with coefficient of variance = " << gpuTimeResult.coeffVar << "%.";
            safeLogWarning(*recorder, ss.str());
            ss.str("");
        }
    }

    // Destroy cuda events
    CUDA_CALL(cudaEventDestroy(startEvent), *recorder);
    CUDA_CALL(cudaEventDestroy(endEvent), *recorder);
    CUDA_CALL(cudaEventDestroy(syncEvent), *recorder);
    CUDA_CALL(cudaEventDestroy(gpuStart), *recorder);
    CUDA_CALL(cudaEventDestroy(inputConsumedEvent), *recorder);

    // Destroy main execution cuda stream
    CUDA_CALL(cudaStreamDestroy(stream), *recorder);

    // Buffers are automatically freed by ScopedSafeMemory destructors
    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample.
//!
//! \details This function is the main execution function of the sample. It loads the engine, allocates
//!          the buffer, executes the engine and reports the performance.
//!
//! \param isProfileRun If true, the function will launch a separate profile run and dump safe profiling data.
//!
bool doInference(SafeExecArgs const& args, std::chrono::high_resolution_clock::time_point const& initStartTime,
    bool isProfileRun = false)
{
    int32_t numThreads = args.threads;
    if (isProfileRun)
    {
        numThreads = 1;
    }

    // Configure recorder(s)
    std::vector<std::unique_ptr<sample::SampleSafeRecorder>> recorders(numThreads);
    for (int32_t k = 0; k < numThreads; ++k)
    {
        auto severity = nvinfer2::safe::Severity::kINFO;
        if (args.debug)
        {
            severity = nvinfer2::safe::Severity::kDEBUG;
        }
        else if (args.verbose)
            severity = nvinfer2::safe::Severity::kVERBOSE;
        recorders[k] = std::make_unique<sample::SampleSafeRecorder>(severity, k);
    }

    // Load safe engine blob
    std::vector<char> blob{loadEngine(args.engineFile)};
    if (blob.data() == nullptr)
    {
        safeLogError(*recorders[0], "Engine blob is empty.");
        return false;
    }
    registerSafetyPlugins(*gSafeRecorder, args.pluginLibraries);

    if (!isProfileRun)
    {
        auto const initEndTime = std::chrono::high_resolution_clock::now();
        auto const initTime = std::chrono::duration<float, std::milli>(initEndTime - initStartTime).count();
        safeLogInfo(*recorders[0], "TensorRT init time is " + std::to_string(initTime) + " ms.");
    }
    else
    {
        safeLogInfo(*recorders[0], "Starting separate safe profiling run.");
    }

    // Configure executor(s)
    std::vector<nvinfer2::safe::ITRTGraph*> graphs(numThreads);
    std::vector<void*> scratchs(numThreads);
    SAFE_API_CALL(nvinfer2::safe::createTRTGraph(graphs[0], blob.data(), blob.size(), *recorders[0],
                      !args.useScratchMemory, &nvinfer2::safe::getSafeMemAllocator()),
        *recorders[0]);
    SAFE_API_CALL(graphs[0]->setIOProfile(args.ioProfile), *recorders[0]);

    for (int32_t k = 1; k < numThreads; ++k)
    {
        SAFE_API_CALL(graphs[0]->clone(graphs[k], *recorders[k]), *recorders[0]);
        SAFE_API_CALL(graphs[k]->setIOProfile(args.ioProfile), *recorders[k]);
    }

    // Configure scratch memory
    if (args.useScratchMemory)
    {
        size_t scratchSize = 0;
        SAFE_API_CALL(graphs[0]->getScratchMemorySize(scratchSize), *recorders[0]);
        for (int32_t k = 0; k < numThreads; ++k)
        {
            CUDA_CALL(cudaMalloc(&scratchs[k], scratchSize), *recorders[k]);
            SAFE_API_CALL(graphs[k]->setScratchMemory(scratchs[k]), *recorders[k]);
        }
    }

    // Run the graphs in independent threads
    std::vector<std::future<bool>> futureResults;
    for (int32_t k = 0; k < numThreads; ++k)
    {
        // launch thread async
        futureResults.emplace_back(
            std::async(std::launch::async, task, args, graphs[k], recorders[k].get(), isProfileRun));
    }

    for (auto& future : futureResults)
    {
        if (!future.get())
        {
            safeLogError(*recorders[0], "Inference failed.");
            return false;
        }
    }

    if (args.useScratchMemory)
    {
        for (int32_t k = 0; k < numThreads; ++k)
        {
            CUDA_CALL(cudaFree(scratchs[k]), *recorders[k]);
            scratchs[k] = nullptr;
            SAFE_API_CALL(graphs[k]->setScratchMemory(nullptr), *recorders[k]);
        }
    }

    for (int32_t k = 0; k < numThreads; ++k)
    {
        SAFE_API_CALL(nvinfer2::safe::destroyTRTGraph(graphs[k]), *recorders[k]);
        graphs[k] = nullptr;
    }

    return true;
}

//!
//! \brief Set device and print device information
//!
bool setDevice(SafeExecArgs const& args)
{
    CUDA_CHECK(cudaSetDevice(args.device));
    int32_t numSMs{0};
    int32_t memoryBusWidth{0};
    int32_t major{0};
    int32_t minor{0};
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, args.device));
    CUDA_CHECK(cudaDeviceGetAttribute(&memoryBusWidth, cudaDevAttrGlobalMemoryBusWidth, args.device));
    // We print the actual SM in use.
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, args.device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, args.device));
    safeLogInfo(*gSafeRecorder,
        "Running on CUDA device number: " + std::to_string(args.device) + " (" + std::to_string(numSMs) + " SMs, "
            + std::to_string(memoryBusWidth) + " bits, Compute Capability " + std::to_string(major) + "."
            + std::to_string(minor) + ")");

    return true;
}

int32_t main(int32_t argc, char** argv)
{
    reportTestStart("TensorRT.trtexec_safe", argc, argv);
    safetyCompliance::setPromgrAbility();
    TestResult result = TestResult::kPASSED;
    // CUDA initialization
    int32_t currentDevice = 0;
    if (cudaGetDevice(&currentDevice) != cudaSuccess)
    {
        safeLogError(*gSafeRecorder, "CUDA initialization failed!");
        return EXIT_FAILURE;
    }

    SafeExecArgs args;
    auto const initStartTime = std::chrono::high_resolution_clock::now();
    if (!parseSafeExecArgs(args, argc, argv))
    {
        printHelpInfo();
        return EXIT_FAILURE;
    }

    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    if (!setDevice(args))
    {
        result = TestResult::kFAILED;
    }
    else
    {
        try
        {
            if (!doInference(args, initStartTime))
            {
                result = TestResult::kFAILED;
            }
        }
        catch (std::runtime_error& e)
        {
            safeLogError(*gSafeRecorder, e.what());
            result = TestResult::kFAILED;
        }

        // Separate profile run
        if (args.separateProfileRun)
        {
            setenv("ENABLE_SAFE_PROFILING", "1", 1);
            try
            {
                if (!doInference(args, initStartTime, /* isProfileRun = */ true))
                {
                    result = TestResult::kFAILED;
                }
            }
            catch (std::runtime_error& e)
            {
                safeLogError(*gSafeRecorder, e.what());
                result = TestResult::kFAILED;
            }
            unsetenv("ENABLE_SAFE_PROFILING");
        }
    }

    reportTestResult("TensorRT.trtexec_safe", result, argc, argv);

    return EXIT_SUCCESS;
}
