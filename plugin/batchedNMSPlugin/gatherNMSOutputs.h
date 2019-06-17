/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#ifndef TRT_BATCHED_NMS_HELPER_H
#define TRT_BATCHED_NMS_HELPER_H
#include "plugin.h"
using namespace nvinfer1;
using namespace nvinfer1::plugin;

pluginStatus_t gatherNMSOutputs(cudaStream_t stream, bool shareLocation, int numImages, int numPredsPerClass,
    int numClasses, int topK, int keepTopK, DataType DT_BBOX, DataType DT_SCORE, const void* indices,
    const void* scores, const void* bboxData, void* keepCount, void* nmsedBoxes, void* nmsedScores, void* nmsedClasses,
    bool clipBoxes = true);

#endif
