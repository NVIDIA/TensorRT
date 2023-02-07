/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_CAFFE_PARSER_MACROS_H
#define TRT_CAFFE_PARSER_MACROS_H
#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

#define CHECK_NULL(ptr)                                                                                                \
    if ((ptr) == nullptr)                                                                                              \
    {                                                                                                                  \
        std::cout << "Error: input " << #ptr << " is NULL in " << FN_NAME << std::endl;                                \
        return;                                                                                                        \
    }
#define CHECK_NULL_RET_NULL(ptr)                                                                                       \
    if ((ptr) == nullptr)                                                                                              \
    {                                                                                                                  \
        std::cout << "Error: input " << #ptr << " is NULL in " << FN_NAME << std::endl;                                \
        return nullptr;                                                                                                \
    }
#define CHECK_NULL_RET_VAL(ptr, val)                                                                                   \
    if ((ptr) == nullptr)                                                                                              \
    {                                                                                                                  \
        std::cout << "Error: input " << #ptr << " is NULL in " << FN_NAME << std::endl;                                \
        return val;                                                                                                    \
    }

#include "parserUtils.h"
#define RETURN_AND_LOG_ERROR(ret, message) RETURN_AND_LOG_ERROR_IMPL(ret, message, "CaffeParser: ")
#endif // TRT_CAFFE_PARSER_MACROS_H
