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

#include <cmath>
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
#include <optional>

#include "NvInfer.h"

#include "common.h"
#include "logger.h"
#include "logging.h"
#include "sampleOptions.h"

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

//! Interface for accuracy validation
//! This interface provides a way to calculate the accuracy gap between the actual and reference outputs.
//! Since all the return value is a "loss value", the lower the return value, the better accuracy it is.
template <typename T>
class IAccuracyValidator
{
public:
    virtual ~IAccuracyValidator() = default;
    virtual double calculateAccuracy(std::vector<T> const& actual, std::vector<T> const& reference) = 0;
};

//! L0 accuracy validator calculates element-wise accuracy using the PyTorch/NumPy allclose formula.
//! An element matches if: |actual[i] - ref[i]| <= atol + rtol * |ref[i]|
//! accuracy = (number of mismatching elements) / N
//! Returns the mismatch ratio (0.0 means perfect match, 1.0 means all elements mismatch).
template <typename T>
class L0AccuracyValidator : public IAccuracyValidator<T>
{
public:
    L0AccuracyValidator(double atol, double rtol)
        : mAtol(atol)
        , mRtol(rtol)
    {
    }

    double calculateAccuracy(std::vector<T> const& actual, std::vector<T> const& reference) override;

private:
    double mAtol;
    double mRtol;
};

//! L1 accuracy validator calculates mean absolute error.
//! accuracy = Sum(|actual[i] - ref[i]|) / N
//! Returns the mean absolute error (0.0 means perfect match).
template <typename T>
class L1AccuracyValidator : public IAccuracyValidator<T>
{
public:
    double calculateAccuracy(std::vector<T> const& actual, std::vector<T> const& reference) override;
};

//! L2 accuracy validator calculates mean squared error.
//! accuracy = Sum(|actual[i] - ref[i]|^2) / N
//! Returns the mean squared error (0.0 means perfect match).
template <typename T>
class L2AccuracyValidator : public IAccuracyValidator<T>
{
public:
    double calculateAccuracy(std::vector<T> const& actual, std::vector<T> const& reference) override;
};

//! LInf accuracy validator calculates maximum absolute error.
//! accuracy = Max(|actual[i] - ref[i]|)
//! Returns the max absolute error (0.0 means perfect match).
template <typename T>
class LInfAccuracyValidator : public IAccuracyValidator<T>
{
public:
    double calculateAccuracy(std::vector<T> const& actual, std::vector<T> const& reference) override;
};

//! Cosine similarity validator calculates 1 - cosine_similarity.
//! cosine_sim = Sum(actual[i] * ref[i]) / (sqrt(Sum(actual[i]^2)) * sqrt(Sum(ref[i]^2)))
//! accuracy loss = 1 - cosine_sim
//! Returns 1 - cosine_similarity (0.0 means perfect match).
template <typename T>
class CosineSimilarityValidator : public IAccuracyValidator<T>
{
public:
    double calculateAccuracy(std::vector<T> const& actual, std::vector<T> const& reference) override;
};

//! \brief Get human-readable name string for an accuracy validation algorithm.
//! \param[in] algorithm The accuracy validation algorithm enum value.
//! \return Name string (e.g., "L0", "L1", "Cosine").
inline std::string getAlgorithmName(AccuracyValidationAlgorithm algorithm)
{
    switch (algorithm)
    {
    case AccuracyValidationAlgorithm::kL0: return "L0";
    case AccuracyValidationAlgorithm::kL1: return "L1";
    case AccuracyValidationAlgorithm::kL2: return "L2";
    case AccuracyValidationAlgorithm::kLInf: return "LInf";
    case AccuracyValidationAlgorithm::kCosineSimilarity: return "Cosine";
    default: return "Unknown";
    }
}

//! \brief Factory function to create an accuracy validator based on algorithm type.
//! \param[in] algorithm The accuracy validation algorithm to use.
//! \param[in] atol Absolute tolerance (only used by L0 algorithm).
//! \param[in] rtol Relative tolerance (only used by L0 algorithm).
//! \return Unique pointer to the appropriate IAccuracyValidator implementation.
template <typename T>
std::unique_ptr<IAccuracyValidator<T>> createAccuracyValidator(
    AccuracyValidationAlgorithm algorithm, float atol = 1e-5F, float rtol = 1e-5F)
{
    switch (algorithm)
    {
    case AccuracyValidationAlgorithm::kL0: return std::make_unique<L0AccuracyValidator<T>>(atol, rtol);
    case AccuracyValidationAlgorithm::kL1: return std::make_unique<L1AccuracyValidator<T>>();
    case AccuracyValidationAlgorithm::kL2: return std::make_unique<L2AccuracyValidator<T>>();
    case AccuracyValidationAlgorithm::kLInf: return std::make_unique<LInfAccuracyValidator<T>>();
    case AccuracyValidationAlgorithm::kCosineSimilarity: return std::make_unique<CosineSimilarityValidator<T>>();
    }
    ASSERT(false && "Unknown Accuracy Validation Algorithm");
    return nullptr;
}

//! \brief Cheap argv pre-scan. Returns true if some `argv[i]` exactly equals `flag`
//! or starts with `flag` + "=". Used by main() before option parsing to dispatch
//! between trtexec single-run mode and the tuning loop.
[[nodiscard]] bool peekArg(int32_t argc, char** argv, char const* flag);

// ============================================================================
// Tuning cache I/O (used by --tuneBuildRoutes / --continue).
// Header is a single JSON object on line 1; iterations are JSON Lines after.
// ============================================================================

//! \brief Reconstruct a shell-safe command line string from argc/argv.
std::string buildShellQuotedCmdLine(int32_t argc, char** argv);

//! \brief Resolve a file path to an absolute path using POSIX realpath().
//! Empty input or realpath() failure returns the input unchanged.
std::string resolveAbsolutePath(std::string const& path);

//! \brief Write the tuning cache file header (line 1, JSON object).
void writeTuningCacheHeader(std::string const& cacheFilePath, AllOptions const& options, int32_t argc, char** argv,
    std::string const& tunerVersion, std::string const& defaultBuildRoute);

//! \brief Append one iteration line to the cache file. Fields: iter, build_route, crash,
//! error_message, accuracy_loss, gpu_time. Crashed iterations have null accuracy/gpu.
void writeTuningCacheIteration(std::string const& cacheFilePath, uint64_t iter, std::string const& buildRoute,
    bool crashed, std::string const& errorMessage, std::unordered_map<std::string, double> const& accuracyLossValues,
    double gpuTimeMs);

//! \struct TuningCacheHeader
//! \brief Parsed contents of the cache header, returned by readTuningCacheHeader().
struct TuningCacheHeader
{
    std::vector<std::string> argv;          //!< Original command line with file paths absolute.
    std::string tuningExpr;                  //!< Expanded --tuneBuildRoutes expression.
    int64_t completedIterations{0};          //!< Number of iteration lines after the header.
};

//! \brief Read and parse the cache file's header line + count completed iteration lines.
std::optional<TuningCacheHeader> readTuningCacheHeader(std::string const& cacheFilePath);

//! \brief Rebuild argv for a --continue resume. argv[0] is replaced with currentExePath;
//! --tuneBuildRoutes is set to the cached expanded expression; --continue and
//! --tuningCacheFile are stripped from the stored argv and the cache path is re-appended.
std::vector<std::string> reconstructArgvFromCacheHeader(
    TuningCacheHeader const& header, std::string const& currentExePath, std::string const& cacheFilePath);

//! \struct CachedIterationResult
//! \brief Minimal per-iteration fields from the cache, used to reconstruct mixed-mode positive knobs.
struct CachedIterationResult
{
    bool crashed{true};
    double gpuTimeMs{0.0};
};

//! \brief Read up to maxIterations iteration lines from the cache and extract (crashed, gpu_time).
std::vector<CachedIterationResult> readCachedIterationResults(std::string const& cacheFilePath, int64_t maxIterations);

namespace tuningCache
{
constexpr char const* kIter = "iter";
constexpr char const* kBuildRoute = "build_route";
constexpr char const* kCrash = "crash";
constexpr char const* kErrorMessage = "error_message";
constexpr char const* kAccuracyLoss = "accuracy_loss";
constexpr char const* kGpuTime = "gpu_time";
} // namespace tuningCache

} // namespace sample
#endif // TRT_SAMPLE_UTILS_H
