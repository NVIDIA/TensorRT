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
#ifndef TRT_NMS_UTILS_H
#define TRT_NMS_UTILS_H

#include "common/plugin.h"

namespace nvinfer1
{
namespace plugin
{
size_t detectionInferenceWorkspaceSize(bool shareLocation, int32_t N, int32_t C1, int32_t C2, int32_t numClasses,
    int32_t numPredsPerClass, int32_t topK, nvinfer1::DataType DT_BBOX, nvinfer1::DataType DT_SCORE);
} // namespace plugin
} // namespace nvinfer1
#endif
