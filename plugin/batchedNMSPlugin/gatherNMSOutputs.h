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
#ifndef TRT_BATCHED_NMS_HELPER_H
#define TRT_BATCHED_NMS_HELPER_H
#include "common/plugin.h"

pluginStatus_t gatherNMSOutputs(cudaStream_t stream, bool shareLocation, int32_t numImages, int32_t numPredsPerClass,
    int32_t numClasses, int32_t topK, int32_t keepTopK, nvinfer1::DataType DT_BBOX, nvinfer1::DataType DT_SCORE,
    void const* indices, void const* scores, void const* bboxData, void* keepCount, void* nmsedBoxes, void* nmsedScores,
    void* nmsedClasses, bool clipBoxes, float const scoreShift);

#endif
