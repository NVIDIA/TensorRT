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
#ifndef CHECK_MACROS_PLUGIN_H
#define CHECK_MACROS_PLUGIN_H

#include "NvInfer.h"
#include <sstream>

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif
#if __cplusplus < 201103L
#define OVERRIDE
#define NORETURN
#else
#define OVERRIDE override
#define NORETURN [[noreturn]]
#endif

namespace nvinfer1
{
namespace plugin
{
template <ILogger::Severity kSeverity>
class LogStream : public std::ostream
{
    class Buf : public std::stringbuf
    {
    public:
        int sync() override;
    };

    Buf buffer;

public:
    LogStream()
        : std::ostream(&buffer){};
};

extern LogStream<ILogger::Severity::kERROR> gLogError;
extern LogStream<ILogger::Severity::kWARNING> gLogWarning;
extern LogStream<ILogger::Severity::kINFO> gLogInfo;

void reportAssertion(const char* msg, const char* file, int line);
void logError(const char* msg, const char* file, const char* fn, int line);

NORETURN void throwCudaError(const char* file, const char* function, int line, int status, const char* msg = nullptr);
NORETURN void throwCudnnError(const char* file, const char* function, int line, int status, const char* msg = nullptr);
NORETURN void throwCublasError(const char* file, const char* function, int line, int status, const char* msg = nullptr);

class TRTException : public std::exception
{
public:
    TRTException(const char* fl, const char* fn, int ln, int st, const char* msg, const char* nm)
        : file(fl)
        , function(fn)
        , line(ln)
        , status(st)
        , message(msg)
        , name(nm)
    {
    }
    virtual void log(std::ostream& logStream) const;
    void setMessage(const char* msg)
    {
        message = msg;
    }

protected:
    const char* file{nullptr};
    const char* function{nullptr};
    int line{0};
    int status{0};
    const char* message{nullptr};
    const char* name{nullptr};
};

class CudaError : public TRTException
{
public:
    CudaError(const char* fl, const char* fn, int ln, int stat, const char* msg = nullptr)
        : TRTException(fl, fn, ln, stat, msg, "Cuda")
    {
    }
};

class CudnnError : public TRTException
{
public:
    CudnnError(const char* fl, const char* fn, int ln, int stat, const char* msg = nullptr)
        : TRTException(fl, fn, ln, stat, msg, "Cudnn")
    {
    }
};

class CublasError : public TRTException
{
public:
    CublasError(const char* fl, const char* fn, int ln, int stat, const char* msg = nullptr)
        : TRTException(fl, fn, ln, stat, msg, "cuBLAS")
    {
    }
};

} // namespace plugin

} // namespace nvinfer1

#define API_CHECK(condition)                                                                                           \
    {                                                                                                                  \
        if ((condition) == false)                                                                                      \
        {                                                                                                              \
            nvinfer1::plugin::logError(#condition, __FILE__, FN_NAME, __LINE__);                                       \
            return;                                                                                                    \
        }                                                                                                              \
    }

#define API_CHECK_RETVAL(condition, retval)                                                                            \
    {                                                                                                                  \
        if ((condition) == false)                                                                                      \
        {                                                                                                              \
            nvinfer1::plugin::logError(#condition, __FILE__, FN_NAME, __LINE__);                                       \
            return retval;                                                                                             \
        }                                                                                                              \
    }

#define API_CHECK_WEIGHTS(Name)                                                                                        \
    API_CHECK((Name).values != nullptr);                                                                               \
    API_CHECK((Name).count > 0);                                                                                       \
    API_CHECK(int((Name).type) >= 0 && int((Name).type) < EnumMax<DataType>());

#define API_CHECK_WEIGHTS0(Name)                                                                                       \
    API_CHECK((Name).count >= 0);                                                                                      \
    API_CHECK((Name).count > 0 ? ((Name).values != nullptr) : ((Name).values == nullptr));                             \
    API_CHECK(int((Name).type) >= 0 && int((Name).type) < EnumMax<DataType>());

#define API_CHECK_WEIGHTS_RETVAL(Name, retval)                                                                         \
    API_CHECK_RETVAL((Name).values != nullptr, retval);                                                                \
    API_CHECK_RETVAL((Name).count > 0, retval);                                                                        \
    API_CHECK_RETVAL(int((Name).type) >= 0 && int((Name).type) < EnumMax<DataType>(), retval);

#define API_CHECK_WEIGHTS0_RETVAL(Name, retval)                                                                        \
    API_CHECK_RETVAL((Name).count >= 0, retval);                                                                       \
    API_CHECK_RETVAL((Name).count > 0 ? ((Name).values != nullptr) : ((Name).values == nullptr), retval);              \
    API_CHECK_RETVAL(int((Name).type) >= 0 && int((Name).type) < EnumMax<DataType>(), retval);

#define API_CHECK_NULL(param) API_CHECK((param) != nullptr)
#define API_CHECK_NULL_RETVAL(param, retval) API_CHECK_RETVAL((param) != nullptr, retval)
#define API_CHECK_NULL_RET_NULL(ptr) API_CHECK_NULL_RETVAL(ptr, nullptr)

#define API_CHECK_ENUM_RANGE(Type, val) API_CHECK(int(val) >= 0 && int(val) < EnumMax<Type>())
#define API_CHECK_ENUM_RANGE_RETVAL(Type, val, retval)                                                                 \
    API_CHECK_RETVAL(int(val) >= 0 && int(val) < EnumMax<Type>(), retval)

#ifndef TRT_PLUGIN_H
#define CUBLASASSERTMSG(status_, msg)                                                                                  \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != CUBLAS_STATUS_SUCCESS)                                                                               \
        {                                                                                                              \
            nvinfer1::plugin::throwCublasError(__FILE__, FN_NAME, __LINE__, s_, msg);                                  \
        }                                                                                                              \
    }

#define CUBLASASSERT(status_)                                                                                          \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != CUBLAS_STATUS_SUCCESS)                                                                               \
        {                                                                                                              \
            nvinfer1::plugin::throwCublasError(__FILE__, FN_NAME, __LINE__, s_);                                       \
        }                                                                                                              \
    }

#define CUDNNASSERTMSG(status_, msg)                                                                                   \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != CUDNN_STATUS_SUCCESS)                                                                                \
        {                                                                                                              \
            nvinfer1::plugin::throwCudnnError(__FILE__, FN_NAME, __LINE__, s_, msg);                                   \
        }                                                                                                              \
    }

#define CUDNNASSERT(status_)                                                                                           \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != CUDNN_STATUS_SUCCESS)                                                                                \
        {                                                                                                              \
            const char* msg = cudnnGetErrorString(s_);                                                                 \
            nvinfer1::plugin::throwCudnnError(__FILE__, FN_NAME, __LINE__, s_, msg);                                   \
        }                                                                                                              \
    }

#define CUASSERTMSG(status_, msg)                                                                                      \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != cudaSuccess)                                                                                         \
        {                                                                                                              \
            nvinfer1::plugin::throwCudaError(__FILE__, FN_NAME, __LINE__, s_, msg);                                    \
        }                                                                                                              \
    }

#define CUASSERT(status_)                                                                                              \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != cudaSuccess)                                                                                         \
        {                                                                                                              \
            const char* msg = cudaGetErrorString(s_);                                                                  \
            nvinfer1::plugin::throwCudaError(__FILE__, FN_NAME, __LINE__, s_, msg);                                    \
        }                                                                                                              \
    }

#define ASSERT(assertion)                                                                                              \
    {                                                                                                                  \
        if (!(assertion))                                                                                              \
        {                                                                                                              \
            nvinfer1::plugin::reportAssertion(#assertion, __FILE__, __LINE__);                                         \
        }                                                                                                              \
    }

#define FAIL(msg)                                                                                                      \
    {                                                                                                                  \
        nvinfer1::plugin::reportAssertion(msg, __FILE__, __LINE__);                                                    \
    }

#define CUERRORMSG(status_)                                                                                            \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != 0)                                                                                                   \
            nvinfer1::plugin::logError(#status_ " failure.", __FILE__, FN_NAME, __LINE__);                             \
    }

#endif // TRT_PLUGIN_H
#endif /*CHECK_MACROS_PLUGIN_H*/
