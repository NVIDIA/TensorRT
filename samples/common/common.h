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

#ifndef TENSORRT_COMMON_H
#define TENSORRT_COMMON_H
#include "NvInfer.h"
#if !TRT_WINML
#include "NvInferPlugin.h"
#endif
#include "logger.h"
#include "sampleEntrypoints.h"
#include "utils/cacheUtils.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <ratio>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _MSC_VER
// For loadLibrary
// Needed so that the max/min definitions in windows.h do not conflict with std::max/min.
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#else
#include <dlfcn.h>
#endif

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

#if defined(__aarch64__) || defined(__QNX__)
#define ENABLE_DLA_API 1
#endif

using namespace nvinfer1;

#define CHECK_RETURN_W_MSG(status, val, errMsg)                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(status))                                                                                                 \
        {                                                                                                              \
            sample::gLogError << errMsg << " Error in " << __FILE__ << ", function " << FN_NAME << "(), line "         \
                              << __LINE__ << std::endl;                                                                \
            return val;                                                                                                \
        }                                                                                                              \
    } while (0)

#undef ASSERT
#define ASSERT(condition)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            sample::gLogError << "Assertion failure: " << #condition << std::endl;                                     \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define CHECK_RETURN(status, val) CHECK_RETURN_W_MSG(status, val, "")

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

#undef CHECK
#define CHECK(status) CHECK_WITH_STREAM(status, std::cerr)

namespace samplesCommon
{
using nvinfer1::utils::buildTimingCacheFromFile;
using nvinfer1::utils::updateTimingCacheFile;

inline bool isDebug()
{
    return std::getenv("TENSORRT_DEBUG") != nullptr;
}

static auto StreamDeleter = [](cudaStream_t* pStream) {
    if (pStream)
    {
        static_cast<void>(cudaStreamDestroy(*pStream));
        delete pStream;
    }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
{
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        pStream.reset(nullptr);
    }

    return pStream;
}

//! Return vector of indices that puts magnitudes of sequence in descending order.
template <class Iter>
std::vector<size_t> argMagnitudeSort(Iter begin, Iter end)
{
    std::vector<size_t> indices(end - begin);
    std::iota(indices.begin(), indices.end(), 0);
    std::ranges::sort(indices, std::greater<>{}, [&begin](size_t i) { return std::abs(begin[i]); });
    return indices;
}

template <typename T>
std::vector<std::string> classify(
    const std::vector<std::string>& refVector, const std::vector<T>& output, const size_t topK)
{
    const auto inds = samplesCommon::argMagnitudeSort(output.cbegin(), output.cend());
    std::vector<std::string> result;
    result.reserve(topK);
    for (size_t k = 0; k < topK; ++k)
    {
        result.push_back(refVector[inds[k]]);
    }
    return result;
}

inline void enableDLA(
    nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, int useDLACore, bool allowGPUFallback = true)
{
    if (useDLACore >= 0)
    {
        if (builder->getNbDLACores() == 0)
        {
            std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores"
                      << std::endl;
            assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
        }
        if (allowGPUFallback)
        {
            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        }
#if ENABLE_FEATURE_WEAK_TYPING
        if (!config->getFlag(nvinfer1::BuilderFlag::kINT8))
        {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
#endif // ENABLE_FEATURE_WEAK_TYPING
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(useDLACore);
    }
}

inline size_t getNbBytes(nvinfer1::DataType t, int64_t vol) noexcept
{
    switch (t)
    {
    case nvinfer1::DataType::kINT64: return 8 * vol;
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kFLOAT: return 4 * vol;
    case nvinfer1::DataType::kBF16:
    case nvinfer1::DataType::kHALF: return 2 * vol;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8: return vol;
    case nvinfer1::DataType::kFP8:
#if CUDA_VERSION < 11060
        ASSERT(false && "FP8 is not supported");
#else
        return vol;
#endif
    case nvinfer1::DataType::kE8M0:
#if CUDA_VERSION < 12080
        ASSERT(false && "E8M0 is not supported");
#else
        return vol;
#endif // CUDA_VERSION < 12080
    case nvinfer1::DataType::kINT4:
    case nvinfer1::DataType::kFP4: return (vol + 1) / 2;
    }
    ASSERT(false && "Unknown element type");
}

// Return least integer no less than exact value of m/n.
template <typename A, typename B>
inline auto divUp(A m, B n) -> std::enable_if_t<std::is_integral<A>::value && std::is_integral<B>::value, A>
{
    ASSERT(n > 0);
    return (m + n - 1) / n;
}

inline int64_t volume(nvinfer1::Dims const& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}

inline int64_t volume(nvinfer1::Dims const& dims, int32_t start, int32_t stop)
{
    ASSERT(start >= 0);
    ASSERT(start <= stop);
    ASSERT(stop <= dims.nbDims);
    ASSERT(std::all_of(dims.d + start, dims.d + stop, [](int32_t x) { return x >= 0; }));
    return std::accumulate(dims.d + start, dims.d + stop, int64_t{1}, std::multiplies<int64_t>{});
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
    ASSERT(infile.is_open() && "Attempting to read from a file that is not open.");
    std::string magic, w, h, max;
    infile >> magic >> w >> h >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), inH * inW);
}

//! Platform-agnostic wrapper around dynamic libraries.
class DynamicLibrary
{
public:
    explicit DynamicLibrary(std::string name)
        : mLibName{std::move(name)}
    {
#if defined(_WIN32)
        mHandle = LoadLibraryA(mLibName.c_str());
#else // defined(_WIN32)
        int32_t flags{RTLD_LAZY};
#if ENABLE_ASAN
        // https://github.com/google/sanitizers/issues/89
        // asan doesn't handle module unloading correctly and there are no plans on doing
        // so. In order to get proper stack traces, don't delete the shared library on
        // close so that asan can resolve the symbols correctly.
        flags |= RTLD_NODELETE;
#endif // ENABLE_ASAN

        mHandle = dlopen(mLibName.c_str(), flags);
#endif // defined(_WIN32)

        if (mHandle == nullptr)
        {
            std::string errorStr{};
#if !defined(_WIN32)
            errorStr = std::string{" due to "} + std::string{dlerror()};
#endif
            throw std::runtime_error("Unable to open library: " + mLibName + errorStr);
        }
    }

    DynamicLibrary(DynamicLibrary const&) = delete;
    DynamicLibrary(DynamicLibrary const&&) = delete;

    //! \return a pointer to a symbol from the DynamicLibrary with the given function signature.
    //!
    //! \throw std::invalid_argument if loading the symbol failed.
    //!
    //! \note Type checking is not possible, so if `Signature` is incorrect, the behavior is undefined.

    template <typename Signature>
    [[nodiscard]] Signature& symbolAddress(char const* name) const
    {
        static_assert(std::is_function_v<Signature>, "Signature must be a function type.");
        if (mHandle == nullptr)
        {
            throw std::runtime_error("Handle to library is nullptr.");
        }
        void* const ret =
#if defined(_MSC_VER)
            static_cast<void*>(GetProcAddress(static_cast<HMODULE>(mHandle), name));
#else
            dlsym(mHandle, name);
#endif
        if (ret == nullptr)
        {
            throw std::invalid_argument(mLibName + ": error loading symbol: " + std::string(name));
        }
        return *reinterpret_cast<Signature*>(ret);
    }

    ~DynamicLibrary()
    {
        try
        {
#if defined(_WIN32)
            ASSERT(static_cast<bool>(FreeLibrary(static_cast<HMODULE>(mHandle))));
#else
            ASSERT(dlclose(mHandle) == 0);
#endif
        }
        catch (...)
        {
            sample::gLogError << "Unable to close library: " << mLibName << std::endl;
        }
    }

private:
    std::string mLibName{}; //!< Name of the DynamicLibrary
    void* mHandle{};        //!< Handle to the DynamicLibrary
};

[[nodiscard]] inline std::unique_ptr<DynamicLibrary> loadLibrary(std::string name)
{
    return std::make_unique<DynamicLibrary>(std::move(name));
}

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
        CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceIndex));
        CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceIndex));
        // Redirect 12.1 to 12.0 to since dependencies do not support 12.1 yet and 12.1 can reuse 12.0 cubins to save
        // lib size/compile time..
        if (major == 12 && minor == 1)
        {
            minor = 0;
        }
        return {major, minor};
    }
};

inline int32_t getSmVersion()
{
    int32_t deviceIndex = 0;
    CHECK(cudaGetDevice(&deviceIndex));

    auto const cc = ComputeCapability::forDevice(deviceIndex);
    return ((cc.major << 8) | cc.minor);
}

inline bool isSmSafe()
{
    int32_t const smVersion = getSmVersion();
    return smVersion == 0x0705 || smVersion == 0x0800 || smVersion == 0x0806 || smVersion == 0x0807
        || smVersion == 0x0A00 || smVersion == 0x0B00;
}

} // namespace samplesCommon

inline std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& dims)
{
    os << "(";
    for (int i = 0; i < dims.nbDims; ++i)
    {
        os << (i ? ", " : "") << dims.d[i];
    }
    return os << ")";
}

#endif // TENSORRT_COMMON_H
