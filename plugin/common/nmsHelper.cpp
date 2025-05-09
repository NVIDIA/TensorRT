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

#include "common/plugin.h"
#include <algorithm>
#include <cuda_fp16.h>

namespace nvinfer1::plugin
{

size_t detectionForwardBBoxDataSize(int32_t N, int32_t C1, DataType DT_BBOX)
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

size_t detectionForwardBBoxPermuteSize(bool shareLocation, int32_t N, int32_t C1, DataType DT_BBOX)
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

size_t detectionForwardPreNMSSize(int32_t N, int32_t C2)
{
    PLUGIN_ASSERT(sizeof(float) == sizeof(int32_t));
    return N * C2 * sizeof(float);
}

size_t detectionForwardPostNMSSize(int32_t N, int32_t numClasses, int32_t topK)
{
    PLUGIN_ASSERT(sizeof(float) == sizeof(int32_t));
    return N * numClasses * topK * sizeof(float);
}
} // namespace nvinfer1::plugin
