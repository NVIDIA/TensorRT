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

#ifndef TRT_SHARED_FILELOCK_H_
#define TRT_SHARED_FILELOCK_H_
#include "NvInfer.h"
#ifdef _MSC_VER
// Needed so that the max/min definitions in windows.h do not conflict with std::max/min.
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#else
#include <stdio.h>  // fileno
#include <unistd.h> // lockf
#endif
#include <string>

namespace nvinfer1::utils
{

//! \brief RAII object that locks the specified file.
//!
//! The FileLock class uses a lock file to specify that the
//! current file is being used by a TensorRT tool or sample
//! so that things like the TimingCache can be updated across
//! processes without having conflicts.
class FileLock
{
public:
    explicit FileLock(nvinfer1::ILogger& logger, std::string fileName);
    ~FileLock();
    FileLock() = delete;                           // no default ctor
    FileLock(FileLock const&) = delete;            // no copy ctor
    FileLock& operator=(FileLock const&) = delete; // no copy assignment
    FileLock(FileLock&&) = delete;                 // no move ctor
    FileLock& operator=(FileLock&&) = delete;      // no move assignment

private:
    //!
    //! The logger that emits any error messages that might show up.
    //!
    nvinfer1::ILogger& mLogger;

    //!
    //! The filename that the FileLock is protecting from multiple
    //! TensorRT processes from writing to.
    //!
    std::string const mFileName;

#ifdef _MSC_VER
    //!
    //! The file handle on windows for the file lock.
    //!
    HANDLE mHandle{};
#elif !defined(__QNX__)
    //!
    //! The file handle on linux for the file lock.
    //!
    FILE* mHandle{};
    //!
    //! The file descriptor on linux of the file lock.
    //!
    int32_t mDescriptor{-1};
#endif
}; // class FileLock

} // namespace nvinfer1::utils

#endif // TRT_SHARED_FILELOCK_H_
