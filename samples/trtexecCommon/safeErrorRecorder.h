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

#ifndef SAFE_ERROR_RECORDER_H
#define SAFE_ERROR_RECORDER_H

#include "NvInferSafeRecorder.h"
#include "safeCommon.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <string_view>

#if ENABLE_NVLOG
#include <nvos_s3_tegra_log.h>
#endif // ENABLE_NVLOG

namespace sample
{
using namespace nvinfer2::safe;

namespace detail
{
//! Copy the contents of a std::string_view into a buffer, truncating if it doesn't fit, and always null-terminating the
//! result (unless dst == nullptr or dstSize == 0). Returns the number of bytes written, including the null terminator.
//! Behavior is undefined if dstSize < 0.
//! Behavior is undefined if dst == nullpltr but dstSize > 0.
inline int64_t truncatingCopyAsCString(std::string_view const src, AsciiChar* const dst, int64_t const dstSize)
{
    SAFE_ASSERT(0 <= dstSize);
    SAFE_ASSERT(dst != nullptr || dstSize == 0);
    if (dstSize == 0)
    {
        return 0;
    }
    SAFE_ASSERT(0 < dstSize); //< We at least have room for a null terminator.
    auto const toWrite = std::min(static_cast<int64_t>(src.size()), dstSize - 1);
    std::copy_n(src.data(), toWrite, dst);
    dst[toWrite] = '\0';
    return toWrite + 1;
}

//! Copy the contents of a std::string_view into a fixed-size buffer, null terminated (unless the buffer has zero size)
//! and return it.
template <typename TArray>
[[nodiscard]] constexpr TArray truncatedCopyAsCString(std::string_view const src)
{
    TArray result{};
    // Expecting TArray to be a std::array or similar, constructing to its size and having a static size:
    SAFE_ASSERT(std::tuple_size<TArray>::value == result.size());
    truncatingCopyAsCString(src, result.data(), result.size());
    return result;
}

#if ENABLE_NVLOG
//! \return a severity number corresponding to an `nvinfer2::safe::Severity`.
//! Unlisted enumerators map to `NVOS_LOG_SEVERITY_INFO`
[[nodiscard]] constexpr uint8_t toNvOsLogSeverity(nvinfer2::safe::Severity sev)
{
    switch (sev)
    {
    case nvinfer2::safe::Severity::kINFO: return NVOS_LOG_SEVERITY_INFO;
    case nvinfer2::safe::Severity::kWARNING: return NVOS_LOG_SEVERITY_WARNING;
    case nvinfer2::safe::Severity::kVERBOSE: return NVOS_LOG_SEVERITY_DEBUG1;
    default: return NVOS_LOG_SEVERITY_INFO;
    }
}
#endif // ENABLE_NVLOG
} // namespace detail

//! The SampleSafeRecorder implementation of the ISafeRecorder interface.
class SampleSafeRecorder : public ISafeRecorder
{
    using DescHolder = std::array<AsciiChar, ISafeRecorder::kMAX_SAFE_DESC_LENGTH + 1U>;
    using errorPair = std::pair<ErrorCode, DescHolder>;

public:
    SampleSafeRecorder(nvinfer2::safe::Severity severity = nvinfer2::safe::Severity::kINFO, int32_t index = -1,
        char const* filename = "TRTErrors.log")
        : ISafeRecorder(severity, index)
    {
        fatalErrorLogFile = fopen(filename, "w+");
        if (fatalErrorLogFile == nullptr)
        {
            std::cerr << "Failed to open error log file: " << filename << std::endl;
        }
    }

    virtual ~SampleSafeRecorder() noexcept
    {
        if (mRefCount != 0)
        {
            reportError(ErrorCode::kINTERNAL_ERROR, "Non-zero reference count for recorder upon deallocation.");
        }
    }

    int32_t getNbErrors() const noexcept final
    {
        return nbErrors;
    }
    ErrorCode getErrorCode(int32_t errorIdx) const noexcept final
    {
        return invalidIndexCheck(errorIdx) ? ErrorCode::kINVALID_ARGUMENT : (*this)[errorIdx].first;
    };
    ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept final
    {
        return invalidIndexCheck(errorIdx) ? "ErrorIdx is out of range." : (*this)[errorIdx].second.data();
    }

    bool hasOverflowed() const noexcept final
    {
        return (nbErrors >= kMAX_NB_ERRORS);
    }

    int32_t getMaxNbErrors() const
    {
        return kMAX_NB_ERRORS;
    }

    // Empty the errorStack.
    void clear() noexcept final
    {
        try
        {
            // grab a lock so that there is no addition while clearing.
            std::lock_guard<std::mutex> guard(mStackLock);
            nbErrors = 0;
        }
        catch (std::exception const& e)
        {
#if ENABLE_NVLOG
            NvOsDebugPrintStr(NVOS_LOG_CODE_START, NVOS_LOG_SEVERITY_ERROR, e.what());
#else
            std::cerr << "Internal Error: " << e.what() << std::endl;
#endif // ENABLE_NVLOG
        }
    };

    //! Simple helper function that checks if the error stack is empty.
    bool empty() const noexcept
    {
        return (nbErrors == 0);
    }

    bool reportError(ErrorCode val, ErrorDesc desc) noexcept final
    {
        try
        {
            std::string_view const descView = desc; //< This implicitly calls strlen once.
            std::cerr << descView << std::endl;
            DescHolder descArr = detail::truncatedCopyAsCString<DescHolder>(descView);

            std::lock_guard<std::mutex> guard(mStackLock);
#if ENABLE_NVLOG
            NvOsDebugPrintStrInt(
                NVOS_LOG_CODE_START, NVOS_LOG_SEVERITY_ERROR, descArr.data(), static_cast<int32_t>(val));
#else
            // Only write to the array if there's space available
            if (nbErrors < kMAX_NB_ERRORS)
            {
                mErrorStack.at(nbErrors) = errorPair(val, descArr);
                nbErrors++;
            }
#endif // ENABLE_NVLOG
        }
        catch (std::exception const& e)
        {
#if ENABLE_NVLOG
            NvOsDebugPrintStr(NVOS_LOG_CODE_START, NVOS_LOG_SEVERITY_ERROR, e.what());
#else
            // `std::ofstream` uses heap allocation which is not allowed for safe samples
            // Hence, C functions are used here to write data to file.
            if (fatalErrorLogFile != nullptr)
            {
                setbuf(fatalErrorLogFile, NULL);
                fwrite(e.what(), strlen(e.what()), 1, fatalErrorLogFile);
                fwrite("\n", 1, 1, fatalErrorLogFile);
                fflush(fatalErrorLogFile);
            }
            std::cerr << e.what() << std::endl;
#endif // ENABLE_NVLOG
        }
        // All errors are considered fatal.
        return true;
    }

    bool reportInfo(ErrorDesc desc) noexcept final
    {
        return reportIfSevere(nvinfer2::safe::Severity::kINFO, desc);
    }

    bool reportWarn(ErrorDesc desc) noexcept final
    {
        return reportIfSevere(nvinfer2::safe::Severity::kWARNING, desc);
    }

    bool reportVerbose(ErrorDesc desc) noexcept final
    {
        return reportIfSevere(nvinfer2::safe::Severity::kVERBOSE, desc);
    }

    bool reportDebug(ErrorDesc desc) noexcept final
    {
        return reportIfSevere(nvinfer2::safe::Severity::kDEBUG, desc);
    }

    // Atomically increment or decrement the ref counter.
    RefCount incRefCount() noexcept final
    {
        return ++mRefCount;
    }
    RefCount decRefCount() noexcept final
    {
        return --mRefCount;
    }

private:
    // Simple helper functions.
    errorPair const& operator[](int32_t index) const noexcept
    {
        return mErrorStack[index];
    }

    bool invalidIndexCheck(int32_t index) const noexcept
    {
        return index >= nbErrors;
    }

    bool reportIfSevere(nvinfer2::safe::Severity msgSev, ErrorDesc desc) noexcept
    {
#if ENABLE_NVLOG
        if (mSeverity >= msgSev)
        {
            auto const severity = detail::toNvOsLogSeverity(msgSev);
            std::lock_guard<std::mutex> guard(mStackLock);
            NvOsDebugPrintStr(NVOS_LOG_CODE_START, severity, desc);
            return true;
        }
        return false;
#else
        if (mSeverity >= msgSev)
        {
            std::lock_guard<std::mutex> guard(mStackLock);
            std::cout << desc << std::endl;
            return true;
        }
        return false;
#endif // ENABLE_NVLOG
    }

    // Used to store the logs that are Fatal
    FILE* fatalErrorLogFile;

    // Mutex to hold when locking mErrorStack for thread safety.
    std::mutex mStackLock;

    // Reference count of the class. Destruction of the class when mRefCount
    // is not zero causes undefined behavior.
    std::atomic<int32_t> mRefCount{0};

    // Number of errors that occurred so far.
    int32_t nbErrors{0};

    // Maximum number of errors that can be stored.
    static constexpr int32_t kMAX_NB_ERRORS = 10;

    // The error stack that holds the errors recorded by TensorRT.
    std::array<errorPair, kMAX_NB_ERRORS> mErrorStack;
}; // class SampleRecorder

} // namespace sample

#endif // SAFE_ERROR_RECORDER_H
