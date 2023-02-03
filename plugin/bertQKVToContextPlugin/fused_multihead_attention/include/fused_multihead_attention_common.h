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
#ifndef _BERT_FMHA_FMHA_COMMON
#define _BERT_FMHA_FMHA_COMMON

#include "common/checkMacrosPlugin.h"
#include "common/plugin.h"
#include <cstdint>
namespace nvinfer1
{
namespace plugin
{
namespace bert
{
enum Data_type
{
    DATA_TYPE_BOOL,
    DATA_TYPE_E8M10,
    DATA_TYPE_E8M7,
    DATA_TYPE_FP16,
    DATA_TYPE_FP32,
    DATA_TYPE_INT4,
    DATA_TYPE_INT8,
    DATA_TYPE_INT32
};
} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // _BERT_FMHA_FMHA_COMMON
