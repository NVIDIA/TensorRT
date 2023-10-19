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

#include "common/checkMacrosPlugin.h"
#include "common/vfcCommon.h"
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace nvinfer1
{
namespace plugin
{

// This will be populated by the logger supplied by the user to initLibNvInferPlugins()
ILogger* gLogger{};

template <ILogger::Severity kSeverity>
int32_t LogStream<kSeverity>::Buf::sync()
{
    std::string s = str();
    while (!s.empty() && s.back() == '\n')
    {
        s.pop_back();
    }
    if (gLogger != nullptr)
    {
        gLogger->log(kSeverity, s.c_str());
    }
    str("");
    return 0;
}

// These use gLogger, and therefore require initLibNvInferPlugins() to be called with a logger
// (otherwise, it will not log)
LogStream<ILogger::Severity::kERROR> gLogError;
LogStream<ILogger::Severity::kWARNING> gLogWarning;
LogStream<ILogger::Severity::kINFO> gLogInfo;
LogStream<ILogger::Severity::kVERBOSE> gLogVerbose;

// break-pointable
void throwCudaError(char const* file, char const* function, int32_t line, int32_t status, char const* msg)
{
    CudaError error(file, function, line, status, msg);
    error.log(gLogError);
    // NOLINTNEXTLINE(misc-throw-by-value-catch-by-reference)
    throw error;
}

// break-pointable
void throwCublasError(char const* file, char const* function, int32_t line, int32_t status, char const* msg)
{
    if (msg == nullptr)
    {
        auto s_ = static_cast<cublasStatus_t>(status);
        switch (s_)
        {
        case CUBLAS_STATUS_SUCCESS: msg = "CUBLAS_STATUS_SUCCESS"; break;
        case CUBLAS_STATUS_NOT_INITIALIZED: msg = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
        case CUBLAS_STATUS_ALLOC_FAILED: msg = "CUBLAS_STATUS_ALLOC_FAILED"; break;
        case CUBLAS_STATUS_INVALID_VALUE: msg = "CUBLAS_STATUS_INVALID_VALUE"; break;
        case CUBLAS_STATUS_ARCH_MISMATCH: msg = "CUBLAS_STATUS_ARCH_MISMATCH"; break;
        case CUBLAS_STATUS_MAPPING_ERROR: msg = "CUBLAS_STATUS_MAPPING_ERROR"; break;
        case CUBLAS_STATUS_EXECUTION_FAILED: msg = "CUBLAS_STATUS_EXECUTION_FAILED"; break;
        case CUBLAS_STATUS_INTERNAL_ERROR: msg = "CUBLAS_STATUS_INTERNAL_ERROR"; break;
        case CUBLAS_STATUS_NOT_SUPPORTED: msg = "CUBLAS_STATUS_NOT_SUPPORTED"; break;
        case CUBLAS_STATUS_LICENSE_ERROR: msg = "CUBLAS_STATUS_LICENSE_ERROR"; break;
        }
    }
    CublasError error(file, function, line, status, msg);
    error.log(gLogError);
    // NOLINTNEXTLINE(misc-throw-by-value-catch-by-reference)
    throw error;
}

// break-pointable
void throwCudnnError(char const* file, char const* function, int32_t line, int32_t status, char const* msg)
{
    CudnnError error(file, function, line, status, msg);
    error.log(gLogError);
    // NOLINTNEXTLINE(misc-throw-by-value-catch-by-reference)
    throw error;
}

// break-pointable
void throwPluginError(char const* file, char const* function, int32_t line, int32_t status, char const* msg)
{
    PluginError error(file, function, line, status, msg);
    reportValidationFailure(msg, file, line);
    // NOLINTNEXTLINE(misc-throw-by-value-catch-by-reference)
    throw error;
}

void logError(char const* msg, char const* file, char const* fn, int32_t line)
{
    gLogError << "Parameter check failed at: " << file << "::" << fn << "::" << line;
    gLogError << ", condition: " << msg << std::endl;
}

void reportValidationFailure(char const* msg, char const* file, int32_t line)
{
    std::ostringstream stream;
    stream << "Validation failed: " << msg << "\n" << file << ':' << line << "\n";
#ifdef COMPILE_VFC_PLUGIN
    ILogger* logger = getPluginLogger();
    if (logger != nullptr)
    {
        logger->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
    }
#else
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
#endif
}

// break-pointable
void reportAssertion(char const* msg, char const* file, int32_t line)
{
    std::ostringstream stream;
    stream << "Assertion failed: " << msg << "\n"
           << file << ':' << line << "\n"
           << "Aborting..."
           << "\n";
#ifdef COMPILE_VFC_PLUGIN
    ILogger* logger = getPluginLogger();
    if (logger != nullptr)
    {
        logger->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
    }
#else
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
#endif
    PLUGIN_CUASSERT(cudaDeviceReset());
    exit(EXIT_FAILURE);
}

void TRTException::log(std::ostream& logStream) const
{
    logStream << file << " (" << line << ") - " << name << " Error in " << function << ": " << status;
    if (message != nullptr)
    {
        logStream << " (" << message << ")";
    }
    logStream << std::endl;
}

} // namespace plugin

} // namespace nvinfer1
