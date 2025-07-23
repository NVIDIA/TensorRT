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
#ifndef TRT_SHARED_TIMINGCACHE_H_
#define TRT_SHARED_TIMINGCACHE_H_

#include "NvInfer.h"
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace nvinfer1::utils
{

//! \brief Loads the binary contents of a cache file into a char vector. Used for both timing cache and runtime cache.
//!
//! \note This is a blocking operation, as this method will acquire an exclusive file lock on the cache file for
//! the duration of the read. \returns The binary data from the file, or an empty vector if an error occurred.
std::vector<char> loadCacheFile(nvinfer1::ILogger& logger, std::string const& inFileName);

//! \brief Helper method to load a timing cache from a file, build an ITimingCache with the data, and then set the new
//! timing cache to the builder config. If the file is blank, or cannot be read, a new timing cache will be created from
//! scratch.
//!
//! \returns The newly created timing cache, or nullptr if an error occurred during creation.
std::unique_ptr<ITimingCache> buildTimingCacheFromFile(
    ILogger& logger, IBuilderConfig& config, std::string const& timingCacheFile);

//! \brief Saves the contents of a cache object to a binary file.
//!
//! \note This is a blocking operation, as this method will acquire an exclusive file lock on the cache file for
//! the duration of the write.
void saveCacheFile(nvinfer1::ILogger& logger, std::string const& outFileName, nvinfer1::IHostMemory const* blob);

//! \brief Updates the contents of a timing cache binary file.
//! This operation loads the timing cache file, combines it with the passed timingCache, and serializes the combined
//! timing cache.
//!
//! \note This is a blocking operation, as this method will acquire an exclusive file lock on the timing cache file for
//! the duration of the write.
void updateTimingCacheFile(nvinfer1::ILogger& logger, std::string const& fileName,
    nvinfer1::ITimingCache const* timingCache, nvinfer1::IBuilder& builder);

} // namespace nvinfer1::utils

#endif // TRT_SHARED_TIMINGCACHE_H_
