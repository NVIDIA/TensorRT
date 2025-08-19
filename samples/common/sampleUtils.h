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

#ifndef TRT_SAMPLE_UTILS_H
#define TRT_SAMPLE_UTILS_H

#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>

#include "NvInfer.h"

#include "common.h"
#include "logger.h"

#define SMP_RETVAL_IF_FALSE(condition, msg, retval, err)                                                               \
    {                                                                                                                  \
        if ((condition) == false)                                                                                      \
        {                                                                                                              \
            (err) << (msg) << std::endl;                                                                               \
            return retval;                                                                                             \
        }                                                                                                              \
    }

namespace sample
{

template <typename T>
inline T roundUp(T m, T n)
{
    return ((m + n - 1) / n) * n;
}

//! comps is the number of components in a vector. Ignored if vecDim < 0.
int64_t volume(nvinfer1::Dims const& dims, nvinfer1::Dims const& strides, int32_t vecDim, int32_t comps, int32_t batch);

using samplesCommon::volume;

nvinfer1::Dims toDims(std::vector<int64_t> const& vec);

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
void fillBuffer(void* buffer, int64_t volume, int32_t min, int32_t max);

template <typename T, typename std::enable_if<!std::is_integral<T>::value, bool>::type = true>
void fillBuffer(void* buffer, int64_t volume, float min, float max);

template <typename T>
void dumpBuffer(void const* buffer, std::string const& separator, std::ostream& os, nvinfer1::Dims const& dims,
    nvinfer1::Dims const& strides, int32_t vectorDim, int32_t spv);

void dumpInt4Buffer(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv);

void loadFromFile(std::string const& fileName, char* dst, size_t size);

std::vector<std::string> splitToStringVec(std::string const& option, char separator, int64_t maxSplit = -1);

bool broadcastIOFormats(std::vector<IOFormat> const& formats, size_t nbBindings, bool isInput = true);

int32_t getCudaDriverVersion();

int32_t getCudaRuntimeVersion();

void sparsify(nvinfer1::INetworkDefinition& network, std::vector<std::vector<int8_t>>& sparseWeights);
void sparsify(nvinfer1::Weights const& weights, int32_t k, int32_t rs, std::vector<int8_t>& sparseWeights);

// Walk the weights elements and overwrite (at most) 2 out of 4 elements to 0.
template <typename T>
void sparsify(T const* values, int64_t count, int32_t k, int32_t rs, std::vector<int8_t>& sparseWeights);

template <typename L>
void setSparseWeights(L& l, int32_t k, int32_t rs, std::vector<int8_t>& sparseWeights);

// Sparsify the weights of Constant layers that are fed to MatMul via Shuffle layers.
// Forward analysis on the API graph to determine which weights to sparsify.
void sparsifyMatMulKernelWeights(
    nvinfer1::INetworkDefinition& network, std::vector<std::vector<int8_t>>& sparseWeights);

template <typename T>
void transpose2DWeights(void* dst, void const* src, int32_t const m, int32_t const n);

//! A helper function to match a target string with a pattern where the pattern can contain up to one wildcard ('*')
//! character that matches to any strings.
bool matchStringWithOneWildcard(std::string const& pattern, std::string const& target);

//! A helper method to find an item from an unordered_map. If the exact match exists, this is identical to
//! map.find(target). If the exact match does not exist, it returns the first plausible match, taking up to one wildcard
//! into account. If there is no plausible match, then it returns map.end().
template <typename T>
typename std::unordered_map<std::string, T>::const_iterator findPlausible(
    std::unordered_map<std::string, T> const& map, std::string const& target)
{
    auto res = map.find(target);
    if (res == map.end())
    {
        res = std::find_if(
            map.begin(), map.end(), [&](typename std::unordered_map<std::string, T>::value_type const& item) {
                return matchStringWithOneWildcard(item.first, target);
            });
    }
    return res;
}

// ==== Common argument parsing utilities ====

//! Simple implementation of startsWith for strings
inline bool startsWith(const std::string& str, const std::string& prefix)
{
    return str.size() >= prefix.size() && str.substr(0, prefix.size()) == prefix;
}

//! Holds a flag and value, so `--foo=bar` becomes `FlagValue{"foo", "bar"}`.
struct FlagValue
{
    std::string flag;
    std::string value;
};

//! Parse a command line argument into flag and value components
//! Returns pair with flag and value, empty flag if parsing failed
std::pair<std::string, std::string> parseFlag(const std::string& arg);

//! Check if argument is a boolean flag (--flag format or whitelisted short flags: -h, -v, -d)
//! Returns flag name if valid, empty string otherwise
std::string parseBooleanFlag(const std::string& arg);

//! Validate that a value is not empty, log error if it is
bool validateNonEmpty(const std::string& value, const std::string& flagName);

//! Validate remote auto tuning config format
bool validateRemoteAutoTuningConfig(const std::string& config);

//! Ensure directory path ends with '/'
inline std::string normalizeDirectoryPath(const std::string& dirPath)
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
std::string sanitizeRemoteAutoTuningConfig(const std::string& config);

//! Sanitizes command line arguments for logging, removing sensitive credentials
//! Processes argv array and sanitizes sensitive arguments like remoteAutoTuningConfig
//! @param argc Number of arguments
//! @param argv Array of argument strings
//! @return Vector of sanitized argument strings
std::vector<std::string> sanitizeArgv(int32_t argc, char** argv);

} // namespace sample

#endif // TRT_SAMPLE_UTILS_H
