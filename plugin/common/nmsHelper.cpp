/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "cuda_fp16.h"
#include "plugin.h"
#include <algorithm>

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using half = __half;

size_t detectionForwardBBoxDataSize(int N, int C1, DataType DT_BBOX)
{
    if (DT_BBOX == DataType::kFLOAT)
    {
        return N * C1 * sizeof(float);
    }
    if (DT_BBOX == DataType::kHALF)
    {
        return N * C1 * sizeof(half);
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
        return shareLocation ? 0 : N * C1 * sizeof(half);
    }
    printf("Only FP32/FP16 type bounding boxes are supported.\n");
    return (size_t) -1;
}

size_t detectionForwardPreNMSSize(int N, int C2, DataType dtype=DataType::kFLOAT)
{
    // for scores: FP32 or FP16
    if (dtype == DataType::kFLOAT) {
        return N * C2 * sizeof(float);
    }
    if (dtype == DataType::kHALF) {
        return N * C2 * sizeof(half);
    }
    // for indices: has to be INT32 for now
    if (dtype == DataType::kINT32) {
        return N * C2 * sizeof(int);
    }
    printf("Only FP32/FP16/INT type are supported.\n");
    return (size_t) -1;
}

size_t detectionForwardPostNMSSize(int N, int numClasses, int topK, DataType dtype=DataType::kFLOAT)
{
    ASSERT(sizeof(float) == sizeof(int));
    // for scores: FP32 or FP16
    if (dtype == DataType::kFLOAT) {
        return N * numClasses * topK * sizeof(float);
    }
    if (dtype == DataType::kHALF) {
        return N * numClasses * topK * sizeof(half);
    }
    // for indices: has to be INT32 for now
    if (dtype == DataType::kINT32) {
        return N * numClasses * topK * sizeof(int);
    }
    printf("Only FP32/FP16/INT32 type are supported.\n");
    return (size_t) -1;
}
