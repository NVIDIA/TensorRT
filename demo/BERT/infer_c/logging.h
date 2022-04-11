/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef INFER_C_LOGGING_H
#define INFER_C_LOGGING_H

#include <NvInfer.h>
#include <cassert>
#include <iostream>
#include <map>

using namespace nvinfer1;
using Severity = nvinfer1::ILogger::Severity;

class Logger : public ILogger
{
public:
    Logger(Severity severity)
        : mOstream(&std::cout)
        , mReportableSeverity(severity)
    {
    }

    template <typename T>
    Logger& operator<<(T const& obj)
    {
        if (mOstream != nullptr)
        {
            *mOstream << obj;
        }
        return *this;
    }

    Logger& report(Severity severity, const char* msg)
    {

        if (severity <= mReportableSeverity)
        {
            const std::map<Severity, std::string> prefixMapping = {{Severity::kINTERNAL_ERROR, "[DemoBERT][F] "},
                {Severity::kERROR, "[DemoBERT][E] "}, {Severity::kWARNING, "[DemoBERT][W] "},
                {Severity::kINFO, "[DemoBERT][I] "}, {Severity::kVERBOSE, "[DemoBERT][V] "}};

            assert(prefixMapping.find(severity) != prefixMapping.end());

            mOstream = &std::cout;

            *this << prefixMapping.at(severity) << msg;

            return *this;
        }
        mOstream = nullptr;
        return *this;
    }

private:
    void log(Severity severity, const char* msg) noexcept override
    {
        report(severity, msg) << "\n";
    }

    std::ostream* mOstream;
    Severity mReportableSeverity;
};

extern Logger gLogger;
#define gLogFatal gLogger.report(Severity::kINTERNAL_ERROR, "")
#define gLogError gLogger.report(Severity::kERROR, "")
#define gLogWarning gLogger.report(Severity::kWARNING, "")
#define gLogInfo gLogger.report(Severity::kINFO, "")
#define gLogVerbose gLogger.report(Severity::kVERBOSE, "")

#endif // INFER_C_LOGGING_H
