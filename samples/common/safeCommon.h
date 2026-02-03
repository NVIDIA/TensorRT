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

#ifndef TENSORRT_SAFE_COMMON_H
#define TENSORRT_SAFE_COMMON_H

#include "NvInferRuntimeBase.h"
#include "NvInferSafeRecorder.h"
#include "cuda_runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// For safeLoadLibrary
#ifdef _MSC_VER
// Needed so that the max/min definitions in windows.h do not conflict with std::max/min.
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#else
#include <dlfcn.h>
#endif
#if IS_QNX_SAFE
#include <cuda_runtime_api_safe_ex.h>
#include <sys/procmgr.h>
#endif // IS_QNX_SAFE

using namespace nvinfer1;

#undef CHECK_WITH_STREAM
#define CHECK_WITH_STREAM(status, stream)                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if ((status) != cudaSuccess)                                                                                   \
        {                                                                                                              \
            stream << "Cuda failure at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(status)          \
                   << std::endl;                                                                                       \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#undef CUDA_CHECK
#define CUDA_CHECK(status) CHECK_WITH_STREAM(status, std::cerr)

#define SAFE_LOG std::cerr

inline std::string getTimestampStr()
{
    std::time_t timestamp = std::time(nullptr);
    tm* tm_local = std::localtime(&timestamp);
    std::stringstream ss;
    ss << "[";
    ss << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon << "/";
    ss << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
    ss << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year << "-";
    ss << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
    ss << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
    ss << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
    return ss.str();
}

inline void safeLogDebug(nvinfer2::safe::ISafeRecorder& recorder, std::string desc)
{
    desc = getTimestampStr() + "[D] " + desc;
    recorder.reportDebug(desc.c_str());
}

inline void safeLogVerbose(nvinfer2::safe::ISafeRecorder& recorder, std::string desc)
{
    desc = getTimestampStr() + "[V] " + desc;
    recorder.reportVerbose(desc.c_str());
}

inline void safeLogInfo(nvinfer2::safe::ISafeRecorder& recorder, std::string desc)
{
    desc = getTimestampStr() + "[I] " + desc;
    recorder.reportInfo(desc.c_str());
}

inline void safeLogWarning(nvinfer2::safe::ISafeRecorder& recorder, std::string desc)
{
    desc = getTimestampStr() + "[W] " + desc;
    recorder.reportWarn(desc.c_str());
}

inline void safeLogError(
    nvinfer2::safe::ISafeRecorder& recorder, std::string desc, ErrorCode val = ErrorCode::kFAILED_EXECUTION)
{
    desc = getTimestampStr() + "[E] " + desc;
    recorder.reportError(val, desc.c_str());
}

#undef SAFE_ASSERT
#define SAFE_ASSERT(condition)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            std::cerr << "Assertion failure: " << #condition << std::endl;                                             \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define SAFE_API_CALL(api_call, recorder)                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        std::stringstream ss;                                                                                          \
        const ErrorCode ret = (api_call);                                                                              \
        if (ret != ErrorCode::kSUCCESS)                                                                                \
        {                                                                                                              \
            ss << "SAFE API Error: [" << #api_call << "]: " << toString(ret);                                          \
            safeLogError(recorder, ss.str(), ret);                                                                     \
            throw ret;                                                                                                 \
        }                                                                                                              \
        ss << "SAFE API:[" << #api_call << "]: PASSED";                                                                \
        safeLogVerbose(recorder, ss.str());                                                                            \
    } while (0)

#define CUDA_CALL(cuda_api_call, recorder)                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        std::stringstream ss;                                                                                          \
        cudaError_t error = (cuda_api_call);                                                                           \
        if (error != cudaSuccess)                                                                                      \
        {                                                                                                              \
            ss << "CUDA Error: [" << #cuda_api_call << "]: " << cudaGetErrorString(error);                             \
            safeLogError(recorder, ss.str(), ErrorCode::kFAILED_EXECUTION);                                            \
            throw ErrorCode::kFAILED_EXECUTION;                                                                        \
        }                                                                                                              \
        ss << "CUDA:[" << #cuda_api_call << "]: PASSED";                                                               \
        safeLogVerbose(recorder, ss.str());                                                                            \
    } while (0)

inline std::string toString(ErrorCode ec)
{
    static const auto ecStrings = [] {
        std::unordered_map<ErrorCode, std::string> result;
#define INSERT_ELEMENT(p, s) result.emplace(p, s);
        INSERT_ELEMENT(ErrorCode::kSUCCESS, "SUCCESS")
        INSERT_ELEMENT(ErrorCode::kUNSPECIFIED_ERROR, "UNSPECIFIED_ERROR")
        INSERT_ELEMENT(ErrorCode::kINTERNAL_ERROR, "INTERNAL_ERROR")
        INSERT_ELEMENT(ErrorCode::kINVALID_ARGUMENT, "INVALID_ARGUMENT")
        INSERT_ELEMENT(ErrorCode::kINVALID_CONFIG, "INVALID_CONFIG")
        INSERT_ELEMENT(ErrorCode::kFAILED_ALLOCATION, "FAILED_ALLOCATION")
        INSERT_ELEMENT(ErrorCode::kFAILED_INITIALIZATION, "FAILED_INITIALIZATION")
        INSERT_ELEMENT(ErrorCode::kFAILED_EXECUTION, "FAILED_EXECUTION")
        INSERT_ELEMENT(ErrorCode::kFAILED_COMPUTATION, "FAILED_COMPUTATION")
        INSERT_ELEMENT(ErrorCode::kINVALID_STATE, "INVALID_STATE")
        INSERT_ELEMENT(ErrorCode::kUNSUPPORTED_STATE, "UNSUPPORTED_STATE")
#undef INSERT_ELEMENT
        return result;
    }();
    return ecStrings.at(ec);
}

//! Locate path to file, given its filename or filepath suffix and possible dirs it might lie in.
//! Function will also walk back MAX_DEPTH dirs from CWD to check for such a file path.
inline std::string locateFile(
    const std::string& filepathSuffix, const std::vector<std::string>& directories, bool reportError = true)
{
    const int MAX_DEPTH{10};
    bool found{false};
    std::string filepath;

    for (auto& dir : directories)
    {
        if (!dir.empty() && dir.back() != '/')
        {
#ifdef _MSC_VER
            filepath = dir + "\\" + filepathSuffix;
#else
            filepath = dir + "/" + filepathSuffix;
#endif
        }
        else
        {
            filepath = dir + filepathSuffix;
        }

        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            const std::ifstream checkFile(filepath);
            found = checkFile.is_open();
            if (found)
            {
                break;
            }

            filepath = "../" + filepath; // Try again in parent dir
        }

        if (found)
        {
            break;
        }

        filepath.clear();
    }

    // Could not find the file
    if (filepath.empty())
    {
        const std::string dirList = std::accumulate(directories.begin() + 1, directories.end(), directories.front(),
            [](const std::string& a, const std::string& b) { return a + "\n\t" + b; });
        std::cout << "Could not find " << filepathSuffix << " in data directories:\n\t" << dirList << std::endl;

        if (reportError)
        {
            std::cout << "&&&& FAILED" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return filepath;
}

inline void readPGMFile(const std::string& fileName, uint8_t* buffer, int32_t inH, int32_t inW)
{
    std::ifstream infile(fileName, std::ifstream::binary);
    SAFE_ASSERT(infile.is_open() && "Attempting to read from a file that is not open.");
    std::string magic, w, h, max;
    infile >> magic >> w >> h >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), inH * inW);
}

namespace samplesSafeCommon
{
//! Represents the compute capability of a device.
//! This pertains to virtual architectures represented by the intermediate PTX format.
//! This is distinct from the SM version.
//! See https://forums.developer.nvidia.com/t/how-should-i-use-correctly-the-sm-xx-and-compute-xx/219160
struct ComputeCapability
{
    int32_t major{};
    int32_t minor{};

    //! \return the compute capability of the CUDA device with the given \p deviceIndex.
    [[nodiscard]] static ComputeCapability forDevice(int32_t deviceIndex)
    {
        int32_t major{0};
        int32_t minor{0};
        CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceIndex));
        CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceIndex));
        return {major, minor};
    }
};

inline int32_t getSmVersion()
{
    int32_t deviceIndex{};
    CUDA_CHECK(cudaGetDevice(&deviceIndex));

    auto const cc = ComputeCapability::forDevice(deviceIndex);
    return ((cc.major << 8) | cc.minor);
}

inline bool isSmSafe()
{
    const int32_t smVersion = getSmVersion();
    return smVersion == 0x0705 || smVersion == 0x0800 || smVersion == 0x0806 || smVersion == 0x0807;
}

inline int32_t calculateSoftmax(float* const prob, int32_t const numDigits)
{
    SAFE_ASSERT(prob != nullptr);
    SAFE_ASSERT(numDigits == 10);
    float sum{0.0F};
    std::transform(prob, prob + numDigits, prob, [&sum](float v) -> float {
        sum += exp(v);
        return exp(v);
    });

    SAFE_ASSERT(sum != 0.0F);
    std::transform(prob, prob + numDigits, prob, [sum](float v) -> float { return v / sum; });
    int32_t idx = std::max_element(prob, prob + numDigits) - prob;
    return idx;
}

//!
//! \brief generate a command line string from the given (argc, argv) values
//!        Note: It simply joins the arguments without proper escaping. If spaces is part
//!        of an argument, they will be joined with single space.
//!
static std::string genCmdlineString(int32_t argc, char const* const* argv)
{
    std::stringstream ss;
    for (int32_t i = 0; i < argc; i++)
    {
        if (i > 0)
        {
            ss << " ";
        }
        ss << argv[i];
    }
    return ss.str();
}

//!
//! \enum TestResult
//! \brief Represents the state of a given test
//!
enum class TestResult
{
    kFAILED, //!< The test failed
    kPASSED, //!< The test passed
};


//!
//! \brief method that implements logging test start
//!
inline void reportTestStart(std::string testName, int32_t argc, char const* const* argv)
{
    SAFE_LOG << "&&&& RUNNING " << testName << " [TensorRT v" << std::to_string(NV_TENSORRT_VERSION) << "] [b"
             << std::to_string(NV_TENSORRT_BUILD) << "]" << " # " << genCmdlineString(argc, argv) << std::endl;
}

//!
//! \brief method that implements logging test results
//!
inline void reportTestResult(std::string testName, TestResult result, int32_t argc, char const* const* argv)
{
    SAFE_LOG << "&&&& " << (result == TestResult::kPASSED ? "PASSED" : "FAILED") << " " << testName << " [TensorRT v"
             << std::to_string(NV_TENSORRT_VERSION) << "] [b" << std::to_string(NV_TENSORRT_BUILD) << "]"
             << " # " << genCmdlineString(argc, argv) << std::endl;
}

//!
//! \class TrtCudaGraphSafe
//! \brief Managed CUDA graph
//!
class TrtCudaGraphSafe
{
public:
    explicit TrtCudaGraphSafe() = default;

    TrtCudaGraphSafe(const TrtCudaGraphSafe&) = delete;

    TrtCudaGraphSafe& operator=(const TrtCudaGraphSafe&) = delete;

    TrtCudaGraphSafe(TrtCudaGraphSafe&&) = delete;

    TrtCudaGraphSafe& operator=(TrtCudaGraphSafe&&) = delete;

    ~TrtCudaGraphSafe()
    {
        if (mGraphExec)
        {
            cudaGraphExecDestroy(mGraphExec);
        }
    }

    void beginCapture(cudaStream_t& stream)
    {
        // cudaStreamCaptureModeGlobal is the only allowed mode in SAFE CUDA
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    }

    bool launch(cudaStream_t& stream)
    {
        return cudaGraphLaunch(mGraphExec, stream) == cudaSuccess;
    }

    void endCapture(cudaStream_t& stream)
    {
        CUDA_CHECK(cudaStreamEndCapture(stream, &mGraph));
        CUDA_CHECK(cudaGraphInstantiate(&mGraphExec, mGraph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(mGraph));
    }

    void endCaptureOnError(cudaStream_t& stream)
    {
        // There are two possibilities why stream capture would fail:
        // (1) stream is in cudaErrorStreamCaptureInvalidated state.
        // (2) TRT reports a failure.
        // In case (1), the returning mGraph should be nullptr.
        // In case (2), the returning mGraph is not nullptr, but it should not be used.
        const auto ret = cudaStreamEndCapture(stream, &mGraph);
        if (ret == cudaErrorStreamCaptureInvalidated)
        {
            SAFE_ASSERT(mGraph == nullptr);
        }
        else
        {
            SAFE_ASSERT(ret == cudaSuccess);
            SAFE_ASSERT(mGraph != nullptr);
            CUDA_CHECK(cudaGraphDestroy(mGraph));
            mGraph = nullptr;
        }
        // Clean up any CUDA error.
        cudaGetLastError();
        SAFE_LOG << "The CUDA graph capture on the stream has failed." << std::endl;
    }

private:
    cudaGraph_t mGraph{};
    cudaGraphExec_t mGraphExec{};
};

inline void* safeLoadLibrary(const std::string& path)
{
#ifdef _MSC_VER
    void* handle = LoadLibraryA(path.c_str());
#else
    int32_t flags{RTLD_LAZY};
    void* handle = dlopen(path.c_str(), flags);
#endif
    if (handle == nullptr)
    {
#ifdef _MSC_VER
        sample::gLogError << "Could not load plugin library: " << path << std::endl;
#else
        SAFE_LOG << "Could not load plugin library: " << path << ", due to: " << dlerror() << std::endl;
#endif
    }
    return handle;
}

//!
//! \class SafetyPluginAttribute
//! \brief Represents a safety plugin with its namespace and name
//!
class SafetyPluginAttribute
{
public:
    std::string pluginNamespace; //!< Plugin namespace (optional, can be empty)
    std::string pluginName;      //!< Plugin name
};

//!
//! \class SafetyPluginLibraryArgument
//! \brief Represents a safety plugin library with its name and associated plugin attributes
//!        Used for parsing command line arguments in the format: libraryName[namespace::pluginName1,pluginName2]
//!
class SafetyPluginLibraryArgument
{
public:
    std::string libraryName;                        //!< Name of the plugin library
    std::vector<SafetyPluginAttribute> pluginAttrs; //!< Vector of plugin attributes contained in this library
};

inline std::vector<std::string> safeSplitString(std::string str, char delimiter = ',')
{
    std::vector<std::string> splitVect;
    std::stringstream ss(str);
    std::string substr;

    while (ss.good())
    {
        getline(ss, substr, delimiter);
        splitVect.emplace_back(std::move(substr));
    }
    return splitVect;
}

// Safety plugin cmd argument example: safetyPluginLibrary[namespace::pluginName1,pluginName2]
inline bool parseSafetyPluginArgument(std::string const& option, SafetyPluginLibraryArgument& args)
{
    auto const leftBracketIdx = option.find('[');
    auto const rightBracketIdx = option.find(']');
    if (leftBracketIdx == std::string::npos || rightBracketIdx == std::string::npos || leftBracketIdx > rightBracketIdx)
    {
        SAFE_LOG << "Invalid safety plugin argument: " << option << std::endl;
        return false;
    }
    args.libraryName = option.substr(0, leftBracketIdx);
    auto const pluginOptionStr = option.substr(leftBracketIdx + 1, rightBracketIdx - leftBracketIdx - 1);
    auto const pluginOptions = safeSplitString(pluginOptionStr, ',');
    if (args.libraryName.empty() || pluginOptions.empty())
    {
        SAFE_LOG << "Invalid safety plugin argument: " << option << std::endl;
        return false;
    }

    auto parsePluginOption = [](std::string const& pluginOption) {
        SafetyPluginAttribute attr{};
        // Check if namespace is used, leave as empty if not exist
        auto const sepratorIdx = pluginOption.find("::");
        if (sepratorIdx == std::string::npos)
        {
            attr.pluginName = pluginOption;
        }
        else
        {
            attr.pluginNamespace = pluginOption.substr(0, sepratorIdx);
            attr.pluginName = pluginOption.substr(sepratorIdx + 2, pluginOption.length() - sepratorIdx - 2);
        }
        return attr;
    };

    for (auto const& pluginOption : pluginOptions)
    {
        auto attr = parsePluginOption(pluginOption);
        if (!attr.pluginName.empty())
        {
            args.pluginAttrs.push_back(attr);
        }
    }

    return true;
}

} // namespace samplesSafeCommon

namespace safetyCompliance
{
inline void initSafeCuda()
{
    // According to CUDA initialization in NVIDIA CUDA SAFETY API REFERENCE FOR DRIVE OS
    // We will need to do the following in order
    // 1. Initialize the calling thread with CUDA specific information (Call any CUDA RT API identified as init)
    // 2. Query/Configure and choose the desired CUDA device
    // 3. CUDA context initialization. (Call cudaDeviceGetLimit or cuCtxCreate)
    size_t stackSizeLimit = 0;
    int32_t deviceIndex = 0;
    CUDA_CHECK(cudaGetDevice(&deviceIndex));
    CUDA_CHECK(cudaDeviceGetLimit(&stackSizeLimit, cudaLimitStackSize));
#if IS_QNX_SAFE
    CUDA_CHECK(cudaSafeExSelectAPIMode(cudaSafeExAPIModeAsilB));
#endif // IS_QNX_SAFE
}

inline void setPromgrAbility()
{
#if IS_QNX_SAFE
    // Comply with DEEPLRN_RES_117 on QNX-safe by dropping PROCMGR_AID_MEM_PHYS ability and locking out any further
    // changes
    procmgr_ability(
        0, PROCMGR_ADN_NONROOT | PROCMGR_AOP_DENY | PROCMGR_AOP_LOCK | PROCMGR_AID_MEM_PHYS, PROCMGR_AID_EOL);
#endif // IS_QNX_SAFE
}

} // namespace safetyCompliance

#endif // TENSORRT_SAFE_COMMON_H
