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

#include "common/checkMacrosPlugin.h"
#include "common/cublasWrapper.h"
#include "vc/vfcCommon.h"
#include <cstdlib>
#include <cuda_runtime.h>

using namespace nvinfer1::pluginInternal;

namespace nvinfer1::plugin
{

// break-pointable
void throwCublasError(char const* file, char const* function, int32_t line, int32_t status, char const* msg)
{
    if (msg == nullptr)
    {
        auto s_ = static_cast<cublasStatus_t>(status);
        switch (s_)
        {
        case CUBLAS_STATUS_SUCCESS: msg = "CUBLAS_STATUS_SUCCESS"; break;
        case CUBLAS_STATUS_NOT_INITIALIZED: msg = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
        case CUBLAS_STATUS_ALLOC_FAILED: msg = "CUBLAS_STATUS_ALLOC_FAILED"; break;
        case CUBLAS_STATUS_INVALID_VALUE: msg = "CUBLAS_STATUS_INVALID_VALUE"; break;
        case CUBLAS_STATUS_ARCH_MISMATCH: msg = "CUBLAS_STATUS_ARCH_MISMATCH"; break;
        case CUBLAS_STATUS_MAPPING_ERROR: msg = "CUBLAS_STATUS_MAPPING_ERROR"; break;
        case CUBLAS_STATUS_EXECUTION_FAILED: msg = "CUBLAS_STATUS_EXECUTION_FAILED"; break;
        case CUBLAS_STATUS_INTERNAL_ERROR: msg = "CUBLAS_STATUS_INTERNAL_ERROR"; break;
        case CUBLAS_STATUS_NOT_SUPPORTED: msg = "CUBLAS_STATUS_NOT_SUPPORTED"; break;
        case CUBLAS_STATUS_LICENSE_ERROR: msg = "CUBLAS_STATUS_LICENSE_ERROR"; break;
        }
    }
    CublasError error(file, function, line, status, msg);
    error.log(gLogError);
    // NOLINTNEXTLINE(misc-throw-by-value-catch-by-reference)
    throw error;
}

// break-pointable
void throwCudnnError(char const* file, char const* function, int32_t line, int32_t status, char const* msg)
{
    CudnnError error(file, function, line, status, msg);
    error.log(gLogError);
    // NOLINTNEXTLINE(misc-throw-by-value-catch-by-reference)
    throw error;
}

} // namespace nvinfer1::plugin
