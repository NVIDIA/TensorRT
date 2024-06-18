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

#ifndef TRT_PLUGIN_VFC_COMMON_H
#define TRT_PLUGIN_VFC_COMMON_H
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"

using namespace nvinfer1;

namespace nvinfer1
{
namespace plugin
{

ILogger* getPluginLogger();

} // namespace plugin
} // namespace nvinfer1

extern "C" TENSORRTAPI void setLoggerFinder(nvinfer1::ILoggerFinder* finder);

extern "C" TENSORRTAPI IPluginCreatorInterface* const* getCreators(int32_t& nbCreators);
#endif // TRT_PLUGIN_VFC_COMMON_H
