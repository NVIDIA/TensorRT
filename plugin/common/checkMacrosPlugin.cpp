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

#include "common/checkMacrosPlugin.h"
#include "common/cublasWrapper.h"
#include "vc/vfcCommon.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <string_view>

using namespace nvinfer1::pluginInternal;

namespace nvinfer1::plugin
{

namespace
{
//! \return Static identifier string for \p status, or an empty view if unrecognized.
[[nodiscard]] constexpr std::string_view cublasStatusName(cublasStatus_t status)
{
    switch (status)
    {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return {};
}
} // namespace

// break-pointable
void throwCublasError(
    char const* file, char const* function, int32_t line, int32_t status, std::optional<std::string> msg)
{
    if (!msg)
    {
        msg.emplace(cublasStatusName(static_cast<cublasStatus_t>(status)));
    }
    CublasError error(file, function, line, status, std::move(msg));
    error.log(gLogError);
    // NOLINTNEXTLINE(misc-throw-by-value-catch-by-reference)
    throw error;
}

// break-pointable
void throwCudnnError(
    char const* file, char const* function, int32_t line, int32_t status, std::optional<std::string> msg)
{
    CudnnError error(file, function, line, status, std::move(msg));
    error.log(gLogError);
    // NOLINTNEXTLINE(misc-throw-by-value-catch-by-reference)
    throw error;
}

} // namespace nvinfer1::plugin
