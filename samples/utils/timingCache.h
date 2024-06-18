/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef TENSORRT_SAMPLES_COMMON_TIMINGCACHE_H_
#define TENSORRT_SAMPLES_COMMON_TIMINGCACHE_H_
#include "NvInfer.h"
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace utils
{
std::vector<char> loadTimingCacheFile(nvinfer1::ILogger& logger, std::string const& inFileName);
std::unique_ptr<ITimingCache> buildTimingCacheFromFile(
    ILogger& logger, IBuilderConfig& config, std::string const& timingCacheFile, std::ostream& err);
void saveTimingCacheFile(nvinfer1::ILogger& logger, std::string const& outFileName, nvinfer1::IHostMemory const* blob);
void updateTimingCacheFile(nvinfer1::ILogger& logger, std::string const& fileName,
    nvinfer1::ITimingCache const* timingCache, nvinfer1::IBuilder& builder);
} // namespace utils
} // namespace nvinfer1

#endif // TENSORRT_SAMPLES_COMMON_TIMINGCACHE_H_
