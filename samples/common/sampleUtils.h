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

#ifndef TRT_SAMPLE_UTILS_H
#define TRT_SAMPLE_UTILS_H

#include <cstdint>
#include <string>
#include <vector>

namespace sample
{

// ==== Common argument parsing utilities ====

//! Validate that a value is not empty, log error if it is
bool validateNonEmpty(std::string const& value, std::string const& flagName);

//! Validate remote auto tuning config format
bool validateRemoteAutoTuningConfig(std::string const& config);

//! Ensure directory path ends with '/'
inline std::string normalizeDirectoryPath(std::string const& dirPath)
{
    std::string result = dirPath;
    if (!result.empty() && result.back() != '/')
    {
        result.push_back('/');
    }
    return result;
}

//! Sanitizes the remote auto tuning config string by removing sensitive credentials
//! Removes usernames and passwords from URL-style config strings for security.
//! Example: "ssh://user:pass@host:22" becomes "ssh://***:***@host:22"
std::string sanitizeRemoteAutoTuningConfig(std::string const& config);

//! Sanitizes command line arguments for logging, removing sensitive credentials
//! Processes argv array and sanitizes sensitive arguments like remoteAutoTuningConfig
//! @param argc Number of arguments
//! @param argv Array of argument strings
//! @return Vector of sanitized argument strings
std::vector<std::string> sanitizeArgv(int32_t argc, char** argv);

} // namespace sample
#endif // TRT_SAMPLE_UTILS_H
