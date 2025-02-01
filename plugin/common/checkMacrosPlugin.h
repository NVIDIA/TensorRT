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
#ifndef CHECK_MACROS_PLUGIN_H
#define CHECK_MACROS_PLUGIN_H

#include "NvInfer.h"
#include "common/cudnnWrapper.h"
#include "vc/checkMacrosPlugin.h"
#include <mutex>
#include <sstream>

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

namespace nvinfer1
{
namespace plugin
{

[[noreturn]] void throwCudnnError(
    char const* file, char const* function, int32_t line, int32_t status, char const* msg = nullptr);
[[noreturn]] void throwCublasError(
    char const* file, char const* function, int32_t line, int32_t status, char const* msg = nullptr);

class CudnnError : public TRTException
{
public:
    CudnnError(char const* fl, char const* fn, int32_t ln, int32_t stat, char const* msg = nullptr)
        : TRTException(fl, fn, ln, stat, msg, "Cudnn")
    {
    }
};

class CublasError : public TRTException
{
public:
    CublasError(char const* fl, char const* fn, int32_t ln, int32_t stat, char const* msg = nullptr)
        : TRTException(fl, fn, ln, stat, msg, "cuBLAS")
    {
    }
};

} // namespace plugin

} // namespace nvinfer1

#define PLUGIN_CHECK_CUDNN(call)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        cudnnStatus_t status = call;                                                                                   \
        if (status != CUDNN_STATUS_SUCCESS)                                                                            \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

#define PLUGIN_CUBLASASSERT(status_)                                                                                   \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != nvinfer1::pluginInternal::CUBLAS_STATUS_SUCCESS)                                                     \
        {                                                                                                              \
            nvinfer1::plugin::throwCublasError(__FILE__, FN_NAME, __LINE__, s_);                                       \
        }                                                                                                              \
    }

#define PLUGIN_CUDNNASSERT(status_)                                                                                    \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != CUDNN_STATUS_SUCCESS)                                                                                \
        {                                                                                                              \
            nvinfer1::pluginInternal::CudnnWrapper& wrapper                                                            \
                = nvinfer1::pluginInternal::getCudnnWrapper(/* plugin caller name */ nullptr);                         \
            const char* msg = wrapper.cudnnGetErrorString(s_);                                                         \
            nvinfer1::plugin::throwCudnnError(__FILE__, FN_NAME, __LINE__, s_, msg);                                   \
        }                                                                                                              \
    }

#endif /*CHECK_MACROS_PLUGIN_H*/
