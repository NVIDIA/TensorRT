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

#include "vc/checkMacrosPlugin.h"
#include "vc/vfcCommon.h"
#include <cstdlib>
#include <cuda_runtime.h>

namespace nvinfer1::plugin
{

// Populated by the logger supplied to initLibNvInferPlugins(). TRT-RTX statically links plugin
// code and may not call initLibNvInferPlugins, so LogStream::sync() falls back
// to the runtime's ILogger* (via the global ::getLogger()) when this is null.
ILogger* gLogger{};

template <ILogger::Severity tSeverity>
int32_t LogStream<tSeverity>::Buf::sync()
{
    ILogger* logger = gLogger;
    if (logger != nullptr)
    {
        std::string s = std::move(*this).str();
        while (!s.empty() && s.back() == '\n')
        {
            s.pop_back();
        }
        logger->log(tSeverity, s.c_str());
    }
    str("");
    return 0;
}

LogStream<ILogger::Severity::kERROR> gLogError;
LogStream<ILogger::Severity::kWARNING> gLogWarning;
LogStream<ILogger::Severity::kINFO> gLogInfo;
LogStream<ILogger::Severity::kVERBOSE> gLogVerbose;

// break-pointable
void throwCudaError(
    char const* file, char const* function, int32_t line, int32_t status, std::optional<std::string> msg)
{
    CudaError error(file, function, line, status, std::move(msg));
    error.log(gLogError);
    // NOLINTNEXTLINE(misc-throw-by-value-catch-by-reference)
    throw error;
}

// break-pointable
void throwPluginError(
    char const* file, char const* function, int32_t line, int32_t status, std::optional<std::string> msg)
{
    reportValidationFailure(msg ? msg->c_str() : "", file, line);
    PluginError error(file, function, line, status, std::move(msg));
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
    logStream << mFile << " (" << mLine << ") - " << mName << " Error in " << mFunction << ": " << mStatus;
    if (!mMessage.empty())
    {
        logStream << " (" << mMessage << ")";
    }
    logStream << std::endl;
}

} // namespace nvinfer1::plugin
