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
#ifndef TRT_BATCHED_NMS_HELPER_H
#define TRT_BATCHED_NMS_HELPER_H
#include "common/plugin.h"

pluginStatus_t gatherNMSOutputs(cudaStream_t stream, bool shareLocation, int numImages, int numPredsPerClass,
    int numClasses, int topK, int keepTopK, nvinfer1::DataType DT_BBOX, nvinfer1::DataType DT_SCORE,
    const void* indices, const void* scores, const void* bboxData, void* keepCount, void* nmsedBoxes, void* nmsedScores,
    void* nmsedClasses, bool clipBoxes, const float scoreShift);

#endif
