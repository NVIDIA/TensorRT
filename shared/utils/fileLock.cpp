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
#include "fileLock.h"
#include "NvInfer.h"
#include <sstream>
#include <stdexcept>
#include <string>

namespace nvinfer1::utils
{

FileLock::FileLock(ILogger& logger, std::string fileName)
    : mLogger(logger)
    , mFileName(std::move(fileName))
{
    std::string lockFileName = mFileName + ".lock";
#ifdef _MSC_VER
    {
        std::stringstream ss;
        ss << "Trying to set exclusive file lock " << lockFileName << std::endl;
        mLogger.log(ILogger::Severity::kVERBOSE, ss.str().c_str());
    }
    // MS docs said this is a blocking IO if "FILE_FLAG_OVERLAPPED" is not provided
    mHandle = CreateFileA(lockFileName.c_str(), GENERIC_WRITE, 0, NULL, OPEN_ALWAYS, 0, NULL);
    if (mHandle == INVALID_HANDLE_VALUE)
    {
        throw std::runtime_error("Failed to lock " + lockFileName + "!");
    }
#elif defined(__QNX__)
    // Calling lockf(F_TLOCK) on QNX returns -1; the reported error is 89 (function not implemented).
    mLogger.log(ILogger::Severity::kVERBOSE, "FileLock is not supported on QNX or GOS.");
#else
    mHandle = fopen(lockFileName.c_str(), "wb+");
    if (mHandle == nullptr)
    {
        throw std::runtime_error("Cannot open " + lockFileName + "!");
    }
    {
        std::stringstream ss;
        ss << "Trying to set exclusive file lock " << lockFileName << std::endl;
        mLogger.log(ILogger::Severity::kVERBOSE, ss.str().c_str());
    }
    mDescriptor = fileno(mHandle);
    auto ret = lockf(mDescriptor, F_LOCK, 0);
    if (ret != 0)
    {
        mDescriptor = -1;
        fclose(mHandle);
        throw std::runtime_error("Failed to lock " + lockFileName + "!");
    }
#endif
}

FileLock::~FileLock()
{
    std::string lockFileName = mFileName + ".lock";
#ifdef _MSC_VER
    if (mHandle != INVALID_HANDLE_VALUE)
    {
        CloseHandle(mHandle);
    }
#elif defined(__QNX__)
    // Calling lockf(F_TLOCK) on QNX returns -1; the reported error is 89 (function not implemented).
    mLogger.log(ILogger::Severity::kVERBOSE, "FileLock is not supported on QNX or GOS.");
#else
    if (mDescriptor != -1)
    {
        auto ret = lockf(mDescriptor, F_ULOCK, 0);
        if (mHandle != nullptr)
        {
            fclose(mHandle);
        }
        if (ret != 0)
        {
            std::stringstream ss;
            ss << "Failed to unlock " << lockFileName << ", please remove " << lockFileName << ".lock manually!"
               << std::endl;
            mLogger.log(ILogger::Severity::kVERBOSE, ss.str().c_str());
        }
    }
#endif
}

} // namespace nvinfer1::utils
