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
#include "companionSo.h"
#include "cuda_runtime.h"
#include "delayStreamKernel.h"
#include "idleWait.h"
#include "safeCommon.h"
#include "safeCudaAllocator.h"
#include "safeErrorRecorder.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

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
    std::string engineSoFile{};
    int32_t iterations{10};
    int32_t avgRuns{10};
    int32_t warmUp{1};
    int32_t device{0};
    int32_t streams{1};
    std::chrono::duration<float, std::milli> idle{0.F};
    std::chrono::duration<float> duration{3.F};
    std::chrono::duration<float, std::milli> sleep{2.F};
    float percentile{99.F};
    bool spin{false};
    bool verbose{false};
    bool debug{false};
    bool reportMemoryUsage{false};
    bool help{false};
    bool useCudaGraph{true};
    bool useScratchMemory{false};
    int32_t threads{1};
    int32_t memoryCycles{1};
    int64_t ioProfile{0};
    SafetyPluginArguments pluginLibraries;
    std::unordered_map<std::string, std::string>
        loadInputs; //!< Map of tensor names to file paths for loading custom input data
    //!< Directory to write output tensors as <dir>/<tensorName>.bin after the final iteration.
    //!< Empty = disabled.
    std::string dumpOutputDir{};
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

//! Maximum number of bytes requested by one engine file read.
constexpr size_t kFILE_READ_CHUNK_BYTES{64U * 1024U * 1024U};

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

private:
    void* mPtr;
    nvinfer2::safe::MemoryPlacement mPlacement;
    nvinfer2::safe::ISafeRecorder& mRecorder;
};

std::optional<std::string> loggedParseString(std::string const& arg, std::string const& name)
{
    auto result = parseString(arg, name);
    if (result)
    {
        safeLogInfo(*gSafeRecorder, name + " : " + *result);
    }
    return result;
}

bool loggedParseBool(std::string const& arg, std::string const& name, std::optional<char> singleChar = {})
{
    bool result = parseBool(arg, name, singleChar);
    if (result)
    {
        safeLogInfo(*gSafeRecorder, name + " : True");
    }
    return result;
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
    // make a copy w/o const qualifier
    TimingMetrics newTimes = times;
    std::ranges::sort(newTimes, {}, [metricIndex](TimingMetric const& a) { return a[metricIndex]; });
    SafePerformanceResult result;
    result.min = newTimes[0][metricIndex];
    result.max = newTimes[newTimes.size() - 1][metricIndex];
    result.mean = std::accumulate(newTimes.begin(), newTimes.end(), 0.F, [metricIndex](float acc, TimingMetric& a) {
        return acc + a[metricIndex];
    }) / newTimes.size();
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
    case DataType::kUINT8: return nvinfer2::safe::TypedArray(static_cast<uint8_t*>(ptr), bufferSize);
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
    if (!arg.starts_with(pattern))
    {
        return false;
    }
    std::string const optionStr = arg.substr(pattern.size());
    bool const status = parseSafetyPluginArgument(optionStr, pluginLibArgs);
    if (!status)
    {
        safeLogError(*gSafeRecorder, "Unable to parse safety plugin library argument: " + arg);
    }
    return status;
}

// Use template to allow volume for either nvinfer1::Dims or nvinfer2::safe::PhysicalDims
template <typename TDims>
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

//! Parses the bounded number of graph lifetimes without growing the general option parser.
bool parseMemoryCycles(std::string const& cycleCountText, int32_t& memoryCycles)
{
    constexpr int32_t kMIN_MEMORY_CYCLES{1};
    constexpr int32_t kMAX_MEMORY_CYCLES{1000};
    try
    {
        size_t parsedCharacterCount{};
        int32_t const requestedCycles = std::stoi(cycleCountText, &parsedCharacterCount);
        if (parsedCharacterCount != cycleCountText.size() || requestedCycles < kMIN_MEMORY_CYCLES
            || requestedCycles > kMAX_MEMORY_CYCLES)
        {
            throw std::out_of_range("memoryCycles");
        }
        memoryCycles = requestedCycles;
        return true;
    }
    catch (std::exception const&)
    {
        safeLogError(*gSafeRecorder,
            "memoryCycles must be an integer in [" + std::to_string(kMIN_MEMORY_CYCLES) + ", "
                + std::to_string(kMAX_MEMORY_CYCLES) + "].");
        return false;
    }
}

//! Validates dependencies between options and the required engine path.
bool validateSafeExecArgs(SafeExecArgs const& args, bool memoryCyclesRequested)
{
    if (memoryCyclesRequested && !args.reportMemoryUsage && !args.help)
    {
        safeLogError(*gSafeRecorder, "--memoryCycles requires --reportMemoryUsage.");
        return false;
    }
    bool const hasRequired = !args.engineFile.empty();
    if (!hasRequired && !args.help)
    {
        safeLogError(*gSafeRecorder, "Engine file is required.");
        return false;
    }
    return true;
}

//!
//! \brief This function parses arguments specific to the sample
//!
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool parseSafeExecArgs(SafeExecArgs& args, int32_t argc, char* argv[])
{
    bool memoryCyclesRequested{false};
    safeLogInfo(*gSafeRecorder, "Parsing input arguments...");
    for (int32_t i = 1; i < argc; ++i)
    {
        std::string const arg = argv[i];
        if (auto value = loggedParseString(arg, "loadEngine"))
        {
            args.engineFile = std::move(*value);
        }
        else if (auto value = loggedParseString(arg, "loadEngineSo"))
        {
            args.engineSoFile = std::move(*value);
        }
        else if (SafetyPluginLibraryArgument pluginArg; parseSafetyPluginLibrary(arg, "safetyPlugins", pluginArg))
        {
            args.pluginLibraries.emplace_back(std::move(pluginArg));
        }
        else if (auto const value = loggedParseString(arg, "iterations"))
        {
            args.iterations = stoi(*value);
        }
        else if (auto const value = loggedParseString(arg, "avgRuns"))
        {
            args.avgRuns = stoi(*value);
        }
        else if (auto const value = loggedParseString(arg, "warmUp"))
        {
            args.warmUp = stoi(*value);
        }
        else if (auto const value = loggedParseString(arg, "device"))
        {
            args.device = stoi(*value);
        }
        else if (auto const value = loggedParseString(arg, "percentile"))
        {
            args.percentile = stof(*value);
        }
        else if (auto const value = loggedParseString(arg, "idleTime"))
        {
            args.idle = std::chrono::duration<float, std::milli>(stof(*value));
        }
        else if (auto const value = loggedParseString(arg, "duration"))
        {
            args.duration = std::chrono::duration<float>(stof(*value));
        }
        else if (auto const value = loggedParseString(arg, "sleepTime"))
        {
            args.sleep = std::chrono::duration<float, std::milli>(stof(*value));
        }
        else if (loggedParseBool(arg, "spin"))
        {
            args.spin = true;
        }
        else if (loggedParseBool(arg, "verbose"))
        {
            args.verbose = true;
        }
        else if (loggedParseBool(arg, "debug"))
        {
            args.debug = true;
        }
        else if (loggedParseBool(arg, "reportMemoryUsage"))
        {
            args.reportMemoryUsage = true;
        }
        else if (auto const cycleCountText = loggedParseString(arg, "memoryCycles"))
        {
            memoryCyclesRequested = true;
            if (!parseMemoryCycles(*cycleCountText, args.memoryCycles))
            {
                return false;
            }
        }
        else if (loggedParseBool(arg, "help", 'h'))
        {
            args.help = true;
        }
        else if (loggedParseBool(arg, "useCudaGraph"))
        {
            // Deprecated: CUDA graph is now enabled by default.
            safeLogWarning(
                *gSafeRecorder, "--useCudaGraph is deprecated (now enabled by default). Use --noCudaGraph to disable.");
        }
        else if (loggedParseBool(arg, "noCudaGraph"))
        {
            args.useCudaGraph = false;
        }
        else if (auto const value = loggedParseString(arg, "threads"))
        {
            args.threads = stoi(*value);
        }
        else if (loggedParseBool(arg, "useScratch"))
        {
            args.useScratchMemory = true;
        }
        else if (loggedParseBool(arg, "separateProfileRun"))
        {
            // Deprecated: separate profile run is now always enabled.
            safeLogWarning(*gSafeRecorder,
                "--separateProfileRun is deprecated (now always enabled). This flag will be removed in a future "
                "release.");
        }
        else if (auto const value = loggedParseString(arg, "ioProfileId"))
        {
            // Select I/O profile index for the TRTGraph
            args.ioProfile = std::stoll(*value);
            if (args.ioProfile < 0)
            {
                safeLogError(*gSafeRecorder, "Invalid ioProfileId (must be >= 0): " + *value);
                return false;
            }
        }
        else if (auto const value = loggedParseString(arg, "loadInputs"))
        {
            args.loadInputs = parseLoadInputs(*value);
            if (!value->empty() && args.loadInputs.empty())
            {
                safeLogError(*gSafeRecorder, "Invalid loadInputs format: " + *value);
                return false;
            }
        }
        else if (auto const value = loggedParseString(arg, "dumpOutputDir"))
        {
            if (value->empty())
            {
                safeLogError(*gSafeRecorder, "--dumpOutputDir requires a non-empty directory path.");
                return false;
            }
            args.dumpOutputDir = std::move(*value);
        }
        else
        {
            safeLogError(*gSafeRecorder, "Invalid Argument: " + arg);
            return false;
        }
    }
    return validateSafeExecArgs(args, memoryCyclesRequested);
}

//!
//! \brief Prints the help information for running this sample.
//!
void printHelpInfo()
{
    SafeExecArgs const defArgs{};
    std::cout << R"(Usage: trtexec_safe --loadEngine=<file> [options]
Required params:
  --loadEngine=FILE  Load the serialized engine from FILE.

General optional params:
  --loadEngineSo=FILE
                     Load the safe engine's companion library from FILE. Without it,
                     <loadEngine>.so is used when that file exists.
  --help or -h       Display help information
  --verbose          Use verbose logging
  --debug            Use debug logging
  --reportMemoryUsage
                     Report tracked memory and cleanup; on Linux, also launch Compute Sanitizer memcheck
  --useScratch       Use separately allocated scratch memory
  --safetyPlugins=spec
                     Load safety plugin libraries (can be specified multiple times)
                     Plugin spec ::= pluginLib[pluginNamespace::pluginName],[...]
                     Example: --safetyPlugins=myPlugin.so[MyNamespace::MyPlugin]
  --loadInputs=spec  Load input values from files (default = generate zero inputs).
                     Input names can be wrapped with single quotes (ex: 'Input:0')
                     Input values spec ::= Ival[\",\"spec]
                                  Ival ::= name\":\"file
                     Example: --loadInputs=\"input1\":data1.bin,\"input2\":data2.bin
  --dumpOutputDir=DIR
                     Write each output tensor's raw bytes to DIR/<tensorName>.bin
                     after the final inference iteration. DIR must already exist.
                     Format is bit-exact bytes, suitable for round-tripping via
                     --loadInputs on a subsequent run. (default = no dump)

Perf measurement params:
  --device=N         Set cuda device to N (default = )"
              << defArgs.device << R"()
  --threads=N        Run in N threads (default = )"
              << defArgs.threads << R"()
  --spin             Actively wait for work completion. This may decrease multi-process
                     synchronization time at the cost of additional CPU usage. (default = false)
  --iterations=N     Run N iterations (default = )"
              << defArgs.iterations << R"()
  --avgRuns=N        Set avgRuns to N - perf is measured as an average of avgRuns (default = )"
              << defArgs.avgRuns << R"()
  --warmUp=N         Run N iterations before actual perf measurement (default = )"
              << defArgs.warmUp << R"()
  --idleTime=F       Sleep F milliseconds between two continuous iterations (default = )"
              << defArgs.idle.count() << R"()
  --percentile=P     For each iteration, report the percentile time at P percentage
                     (0<=P<=100, with 0 representing min, and 100 representing max; default = )"
              << defArgs.percentile << R"(%)
  --noCudaGraph      Disable CUDA graph capture and launch (default = CUDA graph enabled)
  --useCudaGraph     [Deprecated] CUDA graph is now enabled by default. This flag is a no-op.
  --duration=F       Run performance measurements for at least F seconds of wallclock time (default = )"
              << defArgs.duration.count() << R"(s)
  --sleepTime=F      Delay inference start with a gap of F msec between launch and compute (default = )"
              << defArgs.sleep.count() << R"()
  --separateProfileRun
                     [Deprecated] Separate profile run is now always enabled. This flag is a no-op.

I/O profile params:
  --memoryCycles=N  Repeat create/infer/destroy for memory diagnostics; requires --reportMemoryUsage (default = 1).
  --ioProfileId=N    Select the I/O profile index to use (default = )"
              << defArgs.ioProfile << R"()
)";
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

        using GetPluginCreatorFn = IPluginCreatorInterface* (*) (char const*, char const*);
        auto pluginCreatorGetter
            = reinterpret_cast<GetPluginCreatorFn>(dlsym(libraryHandle, pluginGetterSymbolName.c_str()));
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
//! \brief Load a prebuilt TensorRT safe engine using bounded read requests.
//! \param engineFile Path to the serialized engine.
//! \return Buffer containing the complete serialized engine.
//! \throws std::runtime_error if the file size is invalid or any read is incomplete.
std::vector<char> loadEngine(std::string const& engineFile)
{
    std::ifstream file(engineFile, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("Failed to open engine file: " + engineFile);
    }

    std::streamoff const fileSize{file.tellg()};
    if (fileSize <= 0)
    {
        throw std::runtime_error("Engine file is empty or has an invalid size: " + engineFile);
    }
    if (static_cast<uint64_t>(fileSize) > std::numeric_limits<size_t>::max())
    {
        throw std::runtime_error("Engine file is too large to load into memory: " + engineFile);
    }

    file.seekg(0, std::ios::beg);
    if (!file)
    {
        throw std::runtime_error("Failed to seek to the beginning of engine file: " + engineFile);
    }

    size_t const size{static_cast<size_t>(fileSize)};
    std::vector<char> modelBuffer(size);
    size_t offset{0U};
    while (offset < size)
    {
        size_t const bytesToRead{std::min(kFILE_READ_CHUNK_BYTES, size - offset)};
        file.read(modelBuffer.data() + offset, static_cast<std::streamsize>(bytesToRead));
        std::streamsize const bytesRead{file.gcount()};
        if (bytesRead != static_cast<std::streamsize>(bytesToRead) || file.fail() || file.bad())
        {
            std::ostringstream message;
            message << "Failed to read complete engine file: " << engineFile << " at offset " << offset << ". Expected "
                    << bytesToRead << " bytes, read " << bytesRead << " (eof=" << file.eof() << ", fail=" << file.fail()
                    << ", bad=" << file.bad() << ")";
            throw std::runtime_error(message.str());
        }
        offset += bytesToRead;
    }
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

//!
//! \brief Write each output tensor's raw bytes to <dumpDir>/<tensorName>.bin.
//!
//! Filters by ioMode == kOUTPUT; inputs and constants are skipped. For GPU-placed
//! tensors, performs a synchronous D2H copy via a temporary pinned host buffer
//! using the safety-approved stream-create / async-copy / sync / destroy pattern.
//! CPU-placed tensors are written directly. Per-tensor failures are logged and
//! the function continues with the remaining outputs; the return value is true
//! iff every output wrote cleanly.
//!
//! \param[in] dumpDir Directory (must already exist) to write output files into
//! \param[in] tensorNames Parallel vector of I/O tensor names
//! \param[in] tensorDescs Parallel vector of I/O tensor descriptors
//! \param[in] buffers Parallel vector of allocated I/O tensor buffers
//! \param[in] recorder The safe recorder for error logging and API calls
//!
//! \return True if all output tensors were written successfully, false otherwise
//!
[[nodiscard]] bool dumpOutputTensors(std::string const& dumpDir, std::vector<std::string> const& tensorNames,
    std::vector<nvinfer2::safe::TensorDescriptor> const& tensorDescs, std::vector<ScopedSafeMemory> const& buffers,
    nvinfer2::safe::ISafeRecorder& recorder)
{
    SAFE_ASSERT(tensorNames.size() == tensorDescs.size());
    SAFE_ASSERT(tensorNames.size() == buffers.size());

    bool allOk{true};
    int32_t nbWritten{0};

    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), recorder);
    for (size_t i = 0; i < tensorNames.size(); ++i)
    {
        auto const& desc = tensorDescs[i];
        if (desc.ioMode != nvinfer2::safe::TensorIOMode::kOUTPUT)
        {
            continue;
        }
        std::string const& name = tensorNames[i];

        uint64_t const expectedSize = std::max(
            static_cast<uint64_t>(volume(desc.shape, desc.stride, desc.bytesPerComponent)), desc.sizeInBytes);
        if (expectedSize == 0)
        {
            safeLogInfo(recorder, "Skipping dump for empty output tensor: " + name);
            continue;
        }

        std::string const path = dumpDir + "/" + samplesSafeCommon::genFilenameSafeString(name) + ".bin";
        bool const onGpu = desc.memPlacement == nvinfer2::safe::MemoryPlacement::kGPU
            || desc.memPlacement == nvinfer2::safe::MemoryPlacement::kNONE;

        // Stage GPU data into a pinned host buffer; CPU data is written directly.
        // hostBuf is only used in the GPU branch but is declared here so srcPtr stays valid through the file write.
        ScopedSafeMemory hostBuf(onGpu ? expectedSize : 0, kDEFAULT_ALIGNMENT,
            nvinfer2::safe::MemoryPlacement::kCPU_PINNED, nvinfer2::safe::MemoryUsage::kIOTENSOR, recorder);
        void const* srcPtr{nullptr};

        if (onGpu)
        {
            if (!hostBuf)
            {
                safeLogError(recorder, "Failed to allocate host buffer for output tensor: " + name);
                allOk = false;
                continue;
            }

            CUDA_CALL(cudaMemcpyAsync(hostBuf.get(), buffers[i].get(), expectedSize, cudaMemcpyDeviceToHost, stream),
                recorder);
            CUDA_CALL(cudaStreamSynchronize(stream), recorder);
            srcPtr = hostBuf.get();
        }
        else if (desc.memPlacement == nvinfer2::safe::MemoryPlacement::kCPU)
        {
            srcPtr = buffers[i].get();
        }
        else
        {
            safeLogError(recorder, "Invalid memory placement for output tensor: " + name);
            allOk = false;
            continue;
        }

        std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!file.is_open())
        {
            safeLogError(recorder, "Cannot open output file: " + path);
            allOk = false;
            continue;
        }
        file.write(reinterpret_cast<char const*>(srcPtr), static_cast<std::streamsize>(expectedSize));
        if (!file.good())
        {
            std::ostringstream msg;
            msg << "Failed to write " << expectedSize << " bytes to " << path;
            safeLogError(recorder, msg.str());
            allOk = false;
            file.close();
            continue;
        }
        file.close();

        std::ostringstream msg;
        msg << "Dumped output tensor " << name << " (" << expectedSize << " bytes) to " << path;
        safeLogInfo(recorder, msg.str());
        ++nbWritten;
    }

    CUDA_CALL(cudaStreamDestroy(stream), recorder);
    std::ostringstream summary;
    summary << "Output dump complete: " << nbWritten << " tensor(s) written to " << dumpDir;
    safeLogInfo(recorder, summary.str());
    return allOk;
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
//! \param[in] threadIdx Index of this worker thread (only thread 0 dumps outputs when --dumpOutputDir is set)
//!
//! \return True if execution completed successfully, false otherwise
//!
bool task(SafeExecArgs const& args, nvinfer2::safe::ITRTGraph* graph, nvinfer2::safe::ISafeRecorder* recorder,
    bool isProfileRun, int32_t threadIdx)
{
    int64_t nbIOs{};
    SAFE_API_CALL(graph->getNbIOTensors(nbIOs), *recorder);
    std::vector<ScopedSafeMemory> buffers;
    buffers.reserve(nbIOs);
    std::vector<std::string> tensorNames;
    tensorNames.reserve(nbIOs);
    std::vector<nvinfer2::safe::TensorDescriptor> tensorDescs;
    tensorDescs.reserve(nbIOs);
    // Set input tensor values
    for (int64_t i = 0; i < nbIOs; ++i)
    {
        char const* tensor;
        SAFE_API_CALL(graph->getIOTensorName(tensor, i), *recorder);
        nvinfer2::safe::TensorDescriptor desc;
        SAFE_API_CALL(graph->getIOTensorDescriptor(desc, tensor), *recorder);
        tensorNames.emplace_back(tensor);
        tensorDescs.emplace_back(desc);
        buffers.emplace_back(setTensorBuffer(graph, *recorder, tensor, args.loadInputs));
    }

    bool const dumpOutputs = !isProfileRun && !args.dumpOutputDir.empty() && threadIdx == 0;
    if (!isProfileRun && !args.dumpOutputDir.empty() && args.threads > 1 && threadIdx == 0)
    {
        safeLogInfo(*recorder,
            "--dumpOutputDir: dumping outputs only on thread 0 (threads=" + std::to_string(args.threads) + ").");
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

    // Setup as many auxiliary streams as the graph requires - destroyed at scope end.
    auto auxStreamsDeleter = samplesSafeCommon::setUpAuxStreamsOn(*graph, *recorder);

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
    floatDurationMS const maxDurationMs = args.duration;
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
            if (int32_t const error = samplesCommon::waitForIdle(args.idle.count()); error != 0)
            {
                safeLogError(*recorder, "Idle wait failed: " + std::string(std::strerror(error)));
                return false;
            }
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
        std::ranges::sort(totalTimes, {}, [](TimingMetric const& a) { return a[0]; });
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

    bool taskOk{true};
    if (dumpOutputs)
    {
        CUDA_CALL(cudaStreamSynchronize(stream), *recorder);
        if (!dumpOutputTensors(args.dumpOutputDir, tensorNames, tensorDescs, buffers, *recorder))
        {
            safeLogError(*recorder, "Output dump completed with errors.");
            taskOk = false;
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
    return taskOk;
}

//! Reports a synchronized lifecycle boundary; callback peaks capture transient allocations between boundaries.
void reportMemoryPhase(SafeExecArgs const& args, char const* phase)
{
    if (!args.reportMemoryUsage)
    {
        return;
    }
    auto const stats = nvinfer2::safe::SafeMemAllocator::instance().memoryTracker().snapshot();
    safeLogInfo(*gSafeRecorder,
        std::string("Memory lifecycle: ") + phase + ": live=" + std::to_string(stats.currentBytes) + " bytes, peak="
            + std::to_string(stats.peakBytes) + " bytes, allocations=" + std::to_string(stats.outstandingAllocations));
}

//! Runs create/infer/destroy; plugin registration is performed only on the first cycle.
//! @param isProfileRun Launch a separate run to dump safe tensor profiling data.
//! @param registerPlugins Register plugin creators before creating the first graph.
bool doInference(SafeExecArgs const& args, std::chrono::steady_clock::time_point const& initStartTime,
    bool isProfileRun = false, bool registerPlugins = true)
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
    std::vector<char> blob;
    try
    {
        blob = loadEngine(args.engineFile);
    }
    catch (std::exception const& error)
    {
        safeLogError(*recorders[0], std::string{"Failed to load engine: "} + error.what());
        return false;
    }
    if (blob.empty())
    {
        safeLogError(*recorders[0], "Engine blob is empty.");
        return false;
    }
    if (!isProfileRun)
    {
        // Register plugins only on the first run. doInference() is called twice (normal run then profile run);
        // registering again would assert with "existing plugins" error.
        if (registerPlugins)
        {
            registerSafetyPlugins(*gSafeRecorder, args.pluginLibraries);
            auto const initEndTime = std::chrono::steady_clock::now();
            auto const initTime = std::chrono::duration<float, std::milli>(initEndTime - initStartTime).count();
            safeLogInfo(*recorders[0], "TensorRT init time is " + std::to_string(initTime) + " ms.");
        }
    }
    else
    {
        safeLogInfo(*recorders[0], "Starting separate safe profiling run.");
    }

    // Configure executor(s)
    std::vector<nvinfer2::safe::ITRTGraph*> graphs(numThreads);
    std::vector<void*> scratchs(numThreads);
    bool cleanupSucceeded{true};
    // Declared before futures so every worker finishes before graph cleanup during unwinding.
    auto resources = std::shared_ptr<void>(nullptr, [&](void*) noexcept {
        for (int32_t k = 0; k < numThreads; ++k)
        {
            if (graphs[k] != nullptr && nvinfer2::safe::destroyTRTGraph(graphs[k]) != ErrorCode::kSUCCESS)
            {
                recorders[k]->reportError(ErrorCode::kFAILED_ALLOCATION, "Graph destruction failed during cleanup.");
                cleanupSucceeded = false;
            }
            if (scratchs[k] != nullptr)
            {
                bool released{false};
                try
                {
                    auto& tracker = nvinfer2::safe::SafeMemAllocator::instance().memoryTracker();
                    released = tracker.release(scratchs[k], [&] { return cudaFree(scratchs[k]) == cudaSuccess; });
                }
                catch (...)
                {
                }
                if (!released)
                {
                    recorders[k]->reportError(ErrorCode::kFAILED_ALLOCATION, "Scratch cleanup failed.");
                    cleanupSucceeded = false;
                }
            }
        }
    });
    // Whether an engine needs a companion library is recorded in the engine, so nothing is decided here:
    // the library beside it is offered when it is there, and the runtime rejects a pair that does not match.
    auto const companionSo = toolsCompanionSo::resolveCompanionSoPath(args.engineFile, args.engineSoFile);
    std::string const companionSoPath = companionSo.value_or(std::string{});
    if (companionSo)
    {
        safeLogInfo(*recorders[0], "Using companion library " + companionSoPath);
    }
    reportMemoryPhase(args, "before create");
    SAFE_API_CALL(nvinfer2::safe::createTRTGraph(graphs[0], blob.data(), blob.size(),
                      companionSoPath.empty() ? nullptr : companionSoPath.c_str(), *recorders[0],
                      !args.useScratchMemory, &nvinfer2::safe::getSafeMemAllocator()),
        *recorders[0]);
    SAFE_API_CALL(graphs[0]->setIOProfile(args.ioProfile), *recorders[0]);

    for (int32_t k = 1; k < numThreads; ++k)
    {
        SAFE_API_CALL(graphs[0]->clone(graphs[k], *recorders[k]), *recorders[0]);
        SAFE_API_CALL(graphs[k]->setIOProfile(args.ioProfile), *recorders[k]);
    }

    reportMemoryPhase(args, "after create and clone");

    // Configure scratch memory
    if (args.useScratchMemory)
    {
        size_t scratchSize = 0;
        SAFE_API_CALL(graphs[0]->getScratchMemorySize(scratchSize), *recorders[0]);
        for (int32_t k = 0; k < numThreads; ++k)
        {
            CUDA_CALL(cudaMalloc(&scratchs[k], scratchSize), *recorders[k]);
            using Tracker = samplesSafeCommon::SafeMemoryTracker;
            nvinfer2::safe::SafeMemAllocator::instance().memoryTracker().recordAllocation(
                scratchs[k], scratchSize, Tracker::Kind::kGPU, Tracker::Purpose::kSCRATCH);
            SAFE_API_CALL(graphs[k]->setScratchMemory(scratchs[k]), *recorders[k]);
        }
    }

    reportMemoryPhase(args, "after scratch setup");

    // Run the graphs in independent threads
    std::vector<std::future<bool>> futureResults;
    for (int32_t k = 0; k < numThreads; ++k)
    {
        // launch thread async
        futureResults.emplace_back(
            std::async(std::launch::async, task, args, graphs[k], recorders[k].get(), isProfileRun, k));
    }

    for (auto& future : futureResults)
    {
        if (!future.get())
        {
            safeLogError(*recorders[0], "Inference failed.");
            return false;
        }
    }

    reportMemoryPhase(args, "after inference and I/O cleanup");
    resources.reset();
    reportMemoryPhase(args, "after graph and scratch cleanup");
    return cleanupSucceeded;
}

//! Reports allocation accounting after one inference run has released its graph and I/O.
bool reportMemoryUsage(bool isProfileRun, samplesSafeCommon::SafeMemoryTracker::Checkpoint const& before)
{
    auto const stats = nvinfer2::safe::SafeMemAllocator::instance().memoryTracker().snapshot();
    std::ostringstream message;
    message
        << "Memory allocation summary (" << (isProfileRun ? "tensor profiling" : "normal inference") << ")"
        << "\n  Tracked allocation peak: " << stats.peakBytes << " bytes ("
        << static_cast<double>(stats.peakBytes) / (1024.0 * 1024.0) << " MiB)"
        << "\n  Outstanding after run cleanup: " << stats.currentBytes << " bytes in " << stats.outstandingAllocations
        << " allocations"
        << "\n  Scope: requested CPU, pinned and GPU allocations through the tool allocator, plus explicit scratch; "
           "excludes CUDA driver/context memory, engine-file buffers and plugin-private allocations.";
    constexpr std::array<char const*, 4> kKINDS{"CPU", "pinned", "GPU", "unknown"};
    constexpr std::array<char const*, 4> kPURPOSES{"other", "immutable/weights", "scratch", "I/O"};
    auto append = [&](auto const& labels, auto const& counters) {
        for (size_t i = 0; i < labels.size(); ++i)
        {
            message << "\n  " << labels[i] << ": peak=" << counters[i].peakBytes
                    << " bytes, live=" << counters[i].currentBytes << " bytes in " << counters[i].outstandingAllocations
                    << " allocations";
        }
    };
    message << "\n  Independent peaks by placement:";
    append(kKINDS, stats.byKind);
    message << "\n  Independent peaks by purpose:";
    append(kPURPOSES, stats.byPurpose);
    safeLogInfo(*gSafeRecorder, message.str());
    if (!stats.complete)
    {
        safeLogWarning(
            *gSafeRecorder, "Memory allocation accounting is incomplete; the reported peak is a lower bound.");
    }
    auto const released = nvinfer2::safe::SafeMemAllocator::instance().memoryTracker().checkReleased(before);
    if (released.outstandingAllocations != 0)
    {
        safeLogError(*gSafeRecorder,
            "Tracked allocation cleanup check: FAILED; " + std::to_string(released.outstandingBytes) + " bytes in "
                + std::to_string(released.outstandingAllocations)
                + " allocations created during this run remain live (possible leak).");
        return false;
    }
    if (!released.complete)
    {
        safeLogWarning(*gSafeRecorder, "Tracked allocation cleanup check: INCONCLUSIVE; accounting is incomplete.");
    }
    else
    {
        safeLogInfo(*gSafeRecorder,
            "Tracked allocation cleanup check: PASSED; all allocations recorded during this "
            "run were released. This does not check allocations outside the tracked allocator.");
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

//! Reexecutes Linux memory reports under memcheck before CUDA initialization.
//! QNX retains allocator tracking without requiring an external sanitizer.
bool prepareMemorySanitizer(SafeExecArgs const& args, int32_t argc, char** argv)
{
    if (!args.reportMemoryUsage)
    {
        return true;
    }
#if defined(__linux__)
    constexpr char const* kSANITIZER_ACTIVE_ENV{"TRTEXEC_SAFE_MEMORY_SANITIZER_ACTIVE"};
    if (std::getenv(kSANITIZER_ACTIVE_ENV) != nullptr)
    {
        // Consume the marker so unrelated subprocesses can launch their own sanitizer.
        if (unsetenv(kSANITIZER_ACTIVE_ENV) != 0)
        {
            safeLogError(*gSafeRecorder, "Cannot clear the Compute Sanitizer launch marker.");
            return false;
        }
        return true;
    }
    std::vector<std::string> sanitizerArguments{
        "compute-sanitizer", "--tool", "memcheck", "--leak-check", "full", "--error-exitcode", "99", "--"};
    for (int32_t i = 0; i < argc; ++i)
    {
        sanitizerArguments.emplace_back(argv[i]);
    }
    std::vector<char*> sanitizerArgv;
    for (auto& argument : sanitizerArguments)
    {
        sanitizerArgv.push_back(argument.data());
    }
    sanitizerArgv.push_back(nullptr);
    if (setenv(kSANITIZER_ACTIVE_ENV, "1", 1) != 0)
    {
        safeLogError(*gSafeRecorder, "Cannot set the Compute Sanitizer launch marker.");
        return false;
    }
    safeLogInfo(*gSafeRecorder,
        "Launching Compute Sanitizer memcheck with full leak checking; timing includes sanitizer overhead.");
    std::cout.flush();
    std::cerr.flush();
    execvp(sanitizerArgv[0], sanitizerArgv.data());
    int32_t const launchError = errno;
    unsetenv(kSANITIZER_ACTIVE_ENV);
    safeLogError(*gSafeRecorder,
        std::string("Cannot launch compute-sanitizer: ") + std::strerror(launchError)
            + ". Install Compute Sanitizer and make compute-sanitizer available on PATH.");
    return false;
#else
    safeLogInfo(
        *gSafeRecorder, "Compute Sanitizer is not launched on this platform; reporting allocator-tracked memory only.");
    return true;
#endif
}

} // namespace

int32_t main(int32_t argc, char** argv)
{
    safetyCompliance::setPromgrAbility();
    SafeExecArgs args;
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

    if (!prepareMemorySanitizer(args, argc, argv))
    {
        return EXIT_FAILURE;
    }

    reportTestStart("TensorRT.trtexec_safe", argc, argv);
    TestResult result = TestResult::kPASSED;
    // CUDA initialization
    int32_t currentDevice = 0;
    if (cudaGetDevice(&currentDevice) != cudaSuccess)
    {
        safeLogError(*gSafeRecorder, "CUDA initialization failed!");
        return EXIT_FAILURE;
    }

    auto const initStartTime = std::chrono::steady_clock::now();

    if (args.reportMemoryUsage)
    {
        nvinfer2::safe::SafeMemAllocator::instance().memoryTracker().enable();
    }

    if (!setDevice(args))
    {
        result = TestResult::kFAILED;
    }
    else
    {
        auto& tracker = nvinfer2::safe::SafeMemAllocator::instance().memoryTracker();
        auto run = [&](bool isProfileRun, bool registerPlugins) {
            auto const before = tracker.checkpoint();
            bool success{false};
            try
            {
                success = doInference(args, initStartTime, isProfileRun, registerPlugins);
            }
            catch (std::exception const& error)
            {
                safeLogError(*gSafeRecorder, error.what());
            }
            catch (ErrorCode)
            {
                // SAFE_API_CALL already recorded the failed API; scoped graph cleanup has run.
            }
            catch (cudaError_t)
            {
                // CUDA_CALL already recorded the failed CUDA operation.
            }
            if (args.reportMemoryUsage)
            {
                success = reportMemoryUsage(isProfileRun, before) && success;
                tracker.resetPeak();
            }
            return success;
        };
        for (int32_t cycle = 0; cycle < args.memoryCycles; ++cycle)
        {
            if (args.reportMemoryUsage)
            {
                safeLogInfo(*gSafeRecorder,
                    "Memory lifecycle cycle " + std::to_string(cycle + 1) + "/" + std::to_string(args.memoryCycles));
            }
            if (!run(false, cycle == 0))
            {
                result = TestResult::kFAILED;
                break;
            }
        }

        if (result == TestResult::kPASSED)
        {
            setenv("ENABLE_SAFE_PROFILING", "1", 1);
            if (!run(true, false))
            {
                result = TestResult::kFAILED;
            }
            unsetenv("ENABLE_SAFE_PROFILING");
        }
    }

    reportTestResult("TensorRT.trtexec_safe", result, argc, argv);

    return result == TestResult::kPASSED ? EXIT_SUCCESS : EXIT_FAILURE;
}
