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
#ifndef VC_CHECK_MACROS_PLUGIN_H
#define VC_CHECK_MACROS_PLUGIN_H

#include "NvInfer.h"
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

namespace nvinfer1::plugin
{
template <ILogger::Severity kSeverity>
class LogStream : public std::ostream
{
    class Buf : public std::stringbuf
    {
    public:
        int32_t sync() override;
    };

    Buf mBuffer;
    std::mutex mLogStreamMutex;

public:
    std::mutex& getMutex()
    {
        return mLogStreamMutex;
    }
    LogStream()
        : std::ostream(&mBuffer)
    {
    }
};

// Use mutex to protect multi-stream write to buffer
template <ILogger::Severity kSeverity, typename T>
LogStream<kSeverity>& operator<<(LogStream<kSeverity>& stream, T const& msg)
{
    std::lock_guard<std::mutex> guard(stream.getMutex());
    auto& os = static_cast<std::ostream&>(stream);
    os << msg;
    return stream;
}

// Special handling static numbers
template <ILogger::Severity kSeverity>
inline LogStream<kSeverity>& operator<<(LogStream<kSeverity>& stream, int32_t num)
{
    std::lock_guard<std::mutex> guard(stream.getMutex());
    auto& os = static_cast<std::ostream&>(stream);
    os << num;
    return stream;
}

// Special handling std::endl
template <ILogger::Severity kSeverity>
inline LogStream<kSeverity>& operator<<(LogStream<kSeverity>& stream, std::ostream& (*f)(std::ostream&) )
{
    std::lock_guard<std::mutex> guard(stream.getMutex());
    auto& os = static_cast<std::ostream&>(stream);
    os << f;
    return stream;
}

extern LogStream<ILogger::Severity::kERROR> gLogError;
extern LogStream<ILogger::Severity::kWARNING> gLogWarning;
extern LogStream<ILogger::Severity::kINFO> gLogInfo;
extern LogStream<ILogger::Severity::kVERBOSE> gLogVerbose;

void reportValidationFailure(char const* msg, char const* file, int32_t line);
void reportAssertion(char const* msg, char const* file, int32_t line);
void logError(char const* msg, char const* file, char const* fn, int32_t line);

//! \throw CudaError carrying \p msg, after logging it via \c gLogError.
//! \param file, function, line Source location reported in the log line. The strings are not
//!     copied, so the buffers must outlive the exception (e.g., with static storage duration; typically \c __FILE__ and
//!     \c FN_NAME).
//! \param status Subsystem status code (e.g. a \c cudaError_t value).
//! \param msg Human-readable message; ownership is transferred to the thrown exception, so
//!     transient \c std::string values (including those built from a temporary) are safe.
//!     \c std::nullopt indicates "no message"; an explicit empty string is preserved as such.
[[noreturn]] void throwCudaError(char const* file, char const* function, int32_t line, int32_t status,
    std::optional<std::string> msg = std::nullopt);

//! \throw PluginError carrying \p msg, after reporting a validation failure via the plugin logger.
//! \param file, function, line Source location reported in the log line. The strings are not
//!     copied, so the buffers must outlive the exception (e.g., with static storage duration; typically \c __FILE__ and
//!     \c FN_NAME).
//! \param status Subsystem status code.
//! \param msg Human-readable message; ownership is transferred to the thrown exception, so
//!     transient \c std::string values (including those built from a temporary) are safe.
//!     \c std::nullopt indicates "no message"; an explicit empty string is preserved as such.
[[noreturn]] void throwPluginError(char const* file, char const* function, int32_t line, int32_t status,
    std::optional<std::string> msg = std::nullopt);

//! Base class for plugin-side TensorRT exceptions.
//! \note Owns its message: callers may pass a temporary \c std::string (or a \c char const*
//!     that decays into one). The message survives stack unwinding, so \c what() and
//!     \c log() are safe to call on a caught instance.
class TRTException : public std::exception
{
public:
    //! Constructs an exception describing a failure at the given source location.
    //! \param file, function, line Source location. The C-string pointers are not copied and must
    //!     outlive the exception (use \c __FILE__ and \c FN_NAME).
    //! \param status Subsystem status code.
    //! \param message Human-readable message. Moved into the exception and returned by \c what().
    //!     \c std::nullopt indicates "no message" and is stored as an empty string.
    //! \param name Static C-string naming the subsystem (e.g. \c "Cuda", \c "Plugin"). Not copied.
    explicit TRTException(char const* file, char const* function, int32_t line, int32_t status,
        std::optional<std::string> message, char const* name) noexcept
        : mFile{file}
        , mFunction{function}
        , mLine{line}
        , mStatus{status}
        , mMessage{std::move(message).value_or(std::string{})}
        , mName{name}
    {
    }

    //! Writes a description (file, line, name, function, status, then optional parenthesized
    //! message) terminated by \c std::endl to \p logStream. The message is emitted verbatim,
    //! so any embedded newlines appear as line breaks in the output.
    virtual void log(std::ostream& logStream) const;

    //! Replaces the stored message.
    //! \param msg New message; moved into the exception.
    void setMessage(std::string msg)
    {
        mMessage = std::move(msg);
    }

    //! \return Pointer to the owned message, valid for the lifetime of \c *this.
    //!     Returns a pointer to an empty string when no message was supplied.
    [[nodiscard]] char const* what() const noexcept override
    {
        return mMessage.c_str();
    }

protected:
    char const* mFile{nullptr};
    char const* mFunction{nullptr};
    int32_t mLine{0};
    int32_t mStatus{0};
    std::string mMessage;
    char const* mName{nullptr};
};

//! \c TRTException specialization for CUDA driver/runtime failures.
class CudaError : public TRTException
{
public:
    //! \see TRTException::TRTException. Sets the subsystem name to \c "Cuda".
    explicit CudaError(char const* fl, char const* fn, int32_t ln, int32_t stat,
        std::optional<std::string> msg = std::nullopt) noexcept
        : TRTException(fl, fn, ln, stat, std::move(msg), "Cuda")
    {
    }
};

//! \c TRTException specialization for plugin-API failures.
class PluginError : public TRTException
{
public:
    //! \see TRTException::TRTException. Sets the subsystem name to \c "Plugin".
    explicit PluginError(char const* fl, char const* fn, int32_t ln, int32_t stat,
        std::optional<std::string> msg = std::nullopt) noexcept
        : TRTException(fl, fn, ln, stat, std::move(msg), "Plugin")
    {
    }
};

//! Logs `e.what()` to \c gLogError followed by \c std::endl.
inline void caughtError(std::exception const& e)
{
    gLogError << e.what() << std::endl;
}
} // namespace nvinfer1::plugin

#define PLUGIN_API_CHECK(condition)                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if ((condition) == false)                                                                                      \
        {                                                                                                              \
            nvinfer1::plugin::logError(#condition, __FILE__, FN_NAME, __LINE__);                                       \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define PLUGIN_API_CHECK_RETVAL(condition, retval)                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        if ((condition) == false)                                                                                      \
        {                                                                                                              \
            nvinfer1::plugin::logError(#condition, __FILE__, FN_NAME, __LINE__);                                       \
            return retval;                                                                                             \
        }                                                                                                              \
    } while (0)

#define PLUGIN_API_CHECK_ENUM_RANGE(Type, val) PLUGIN_API_CHECK(int32_t(val) >= 0 && int32_t(val) < EnumMax<Type>())
#define PLUGIN_API_CHECK_ENUM_RANGE_RETVAL(Type, val, retval)                                                          \
    PLUGIN_API_CHECK_RETVAL(int32_t(val) >= 0 && int32_t(val) < EnumMax<Type>(), retval)

#define PLUGIN_CHECK_CUDA(call)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

#define PLUGIN_CUASSERT(status_)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != cudaSuccess)                                                                                         \
        {                                                                                                              \
            char const* msg = cudaGetErrorString(s_);                                                                  \
            nvinfer1::plugin::throwCudaError(__FILE__, FN_NAME, __LINE__, s_, msg);                                    \
        }                                                                                                              \
    } while (0)

// On MSVC, nested macros don't expand correctly without some help, so use TRT_EXPAND to help it out.
#define TRT_EXPAND(x) x
#define GET_MACRO(_1, _2, NAME, ...) NAME
#define PLUGIN_VALIDATE(...)                                                                                           \
    TRT_EXPAND(GET_MACRO(__VA_ARGS__, PLUGIN_VALIDATE_MSG, PLUGIN_VALIDATE_DEFAULT, )(__VA_ARGS__))

//! Compile-time guard: rejects conditions that decay to \c char \c const*.
//! The bug this catches is \c PLUGIN_VALIDATE("some message") (or with \c .c_str()), where the
//! string is non-null and the check silently passes. Use \c PLUGIN_ERROR(msg) for fatal messages.
#define PLUGIN_DETAIL_REJECT_STRING_CONDITION(expr)                                                                    \
    static_assert(!std::is_convertible_v<std::decay_t<decltype(expr)>, char const*>,                                   \
        "PLUGIN_VALIDATE/PLUGIN_ASSERT condition must not be a string; use PLUGIN_ERROR(msg) for a fatal message")

// Logs failed condition and throws a PluginError.
// PLUGIN_ASSERT will eventually perform this function, at which point PLUGIN_VALIDATE
// will be removed.
#define PLUGIN_VALIDATE_DEFAULT(condition)                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        PLUGIN_DETAIL_REJECT_STRING_CONDITION(condition);                                                              \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            nvinfer1::plugin::throwPluginError(__FILE__, FN_NAME, __LINE__, 0, #condition);                            \
        }                                                                                                              \
    } while (0)

#define PLUGIN_VALIDATE_MSG(condition, msg)                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        PLUGIN_DETAIL_REJECT_STRING_CONDITION(condition);                                                              \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            nvinfer1::plugin::throwPluginError(__FILE__, FN_NAME, __LINE__, 0, msg);                                   \
        }                                                                                                              \
    } while (0)

//! Logs failed assertion and aborts.
//! Aborting is undesirable and will be phased-out from the plugin module, at which point
//! PLUGIN_ASSERT will perform the same function as PLUGIN_VALIDATE.
#define PLUGIN_ASSERT(assertion)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        PLUGIN_DETAIL_REJECT_STRING_CONDITION(assertion);                                                              \
        if (!(assertion))                                                                                              \
        {                                                                                                              \
            nvinfer1::plugin::reportAssertion(#assertion, __FILE__, __LINE__);                                         \
        }                                                                                                              \
    } while (0)

//! Unconditionally logs failed assertion and aborts.
#define PLUGIN_FAIL(msg)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        nvinfer1::plugin::reportAssertion(msg, __FILE__, __LINE__);                                                    \
    } while (0)

// Consider wrapping in do{...} while(0):
//! Logs a plugin error and throws a PluginError.
#define PLUGIN_ERROR(msg)                                                                                              \
    {                                                                                                                  \
        nvinfer1::plugin::throwPluginError(__FILE__, FN_NAME, __LINE__, 0, msg);                                       \
    }

#define PLUGIN_CUERROR(status_)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if (cudaError_t const s_ = (status_); s_ != cudaSuccess)                                                       \
        {                                                                                                              \
            nvinfer1::plugin::logError(#status_ " failure.", __FILE__, FN_NAME, __LINE__);                             \
        }                                                                                                              \
    } while (0)

#endif /*VC_CHECK_MACROS_PLUGIN_H*/
