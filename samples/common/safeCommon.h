/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "NvInferSafeRuntime.h"
#include "cuda_runtime.h"
#include "sampleEntrypoints.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

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

#undef CHECK
#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << std::endl;                                                         \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

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

namespace samplesCommon
{
template <typename T>
inline std::shared_ptr<T> infer_object(T* obj)
{
    if (!obj)
    {
        throw std::runtime_error("Failed to create object");
    }
    return std::shared_ptr<T>(obj);
}

inline uint32_t elementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16: return 2;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kFP8: return 1;
    case nvinfer1::DataType::kINT4:
        SAFE_ASSERT(false && "Element size is not implemented for sub-byte data-types");
    }
    return 0;
}

template <typename A, typename B>
inline A divUp(A x, B n)
{
    return (x + n - 1) / n;
}

inline int64_t volume(nvinfer1::Dims const& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}

//! Return m rounded up to nearest multiple of n
template <typename T1, typename T2>
inline T1 roundUp(T1 m, T2 n)
{
    static_assert(std::is_integral<T1>::value && std::is_integral<T2>::value, "arguments must be integers");
    static_assert(std::is_signed<T1>::value == std::is_signed<T2>::value, "mixed signedness not allowed");
    static_assert(sizeof(T1) >= sizeof(T2), "first type must be as least as wide as second type");
    return ((m + n - 1) / n) * n;
}

//! comps is the number of components in a vector. Ignored if vecDim < 0.
inline int64_t volume(nvinfer1::Dims dims, int32_t vecDim, int32_t comps, int32_t batch)
{
    if (vecDim >= 0)
    {
        dims.d[vecDim] = roundUp(dims.d[vecDim], comps);
    }
    return samplesCommon::volume(dims) * std::max(batch, 1);
}

inline int32_t getSMVersion()
{
#if 0
    // Use default value for 4090
    int32_t major{8};
    int32_t minor{9};
#else
    int32_t major{};
    int32_t minor{};
    int32_t deviceIndex{};
    CHECK(cudaGetDevice(&deviceIndex));
    CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceIndex));
    CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceIndex));
#endif
    return ((major << 8) | minor);
}

inline bool isSMSafe()
{
    const int32_t smVersion = getSMVersion();
    return smVersion == 0x0700 || smVersion == 0x0705 || smVersion == 0x0800 || smVersion == 0x0806
        || smVersion == 0x0807;
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
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    }

    bool launch(cudaStream_t& stream)
    {
        return cudaGraphLaunch(mGraphExec, stream) == cudaSuccess;
    }

    void endCapture(cudaStream_t& stream)
    {
        CHECK(cudaStreamEndCapture(stream, &mGraph));
        CHECK(cudaGraphInstantiate(&mGraphExec, mGraph, nullptr, nullptr, 0));
        CHECK(cudaGraphDestroy(mGraph));
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
            CHECK(cudaGraphDestroy(mGraph));
            mGraph = nullptr;
        }
        // Clean up any CUDA error.
        cudaGetLastError();
        sample::gLogError << "The CUDA graph capture on the stream has failed." << std::endl;
    }

private:
    cudaGraph_t mGraph{};
    cudaGraphExec_t mGraphExec{};
};

inline void safeLoadLibrary(const std::string& path)
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
        sample::gLogError << "Could not load plugin library: " << path << ", due to: " << dlerror() << std::endl;
#endif
    }
}

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

} // namespace samplesCommon

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
    CHECK(cudaGetDevice(&deviceIndex));
    CHECK(cudaDeviceGetLimit(&stackSizeLimit, cudaLimitStackSize));
#if IS_QNX_SAFE
    CHECK(cudaSafeExSelectAPIMode(cudaSafeExAPIModeAsilB));
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
