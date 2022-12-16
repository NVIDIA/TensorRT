/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/plugin.h"
#include "cuda_fp16.h"
#include <algorithm>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

size_t detectionForwardBBoxDataSize(int N, int C1, DataType DT_BBOX)
{
    if (DT_BBOX == DataType::kFLOAT)
    {
        return N * C1 * sizeof(float);
    }
    if (DT_BBOX == DataType::kHALF)
    {
        return N * C1 * sizeof(__half);
    }

    printf("Only FP32/FP16 type bounding boxes are supported.\n");
    return (size_t) -1;
}

size_t detectionForwardBBoxPermuteSize(bool shareLocation, int N, int C1, DataType DT_BBOX)
{
    if (DT_BBOX == DataType::kFLOAT)
    {
        return shareLocation ? 0 : N * C1 * sizeof(float);
    }
    if (DT_BBOX == DataType::kHALF)
    {
        return shareLocation ? 0 : N * C1 * sizeof(__half);
    }

    printf("Only FP32/FP16 type bounding boxes are supported.\n");
    return (size_t) -1;
}

size_t detectionForwardPreNMSSize(int N, int C2)
{
    PLUGIN_ASSERT(sizeof(float) == sizeof(int));
    return N * C2 * sizeof(float);
}

size_t detectionForwardPostNMSSize(int N, int numClasses, int topK)
{
    PLUGIN_ASSERT(sizeof(float) == sizeof(int));
    return N * numClasses * topK * sizeof(float);
}
