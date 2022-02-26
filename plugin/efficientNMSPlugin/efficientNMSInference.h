/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef TRT_EFFICIENT_NMS_INFERENCE_H
#define TRT_EFFICIENT_NMS_INFERENCE_H

#include "plugin.h"

#include "efficientNMSParameters.h"

size_t EfficientNMSWorkspaceSize(int batchSize, int numScoreElements, int numClasses, nvinfer1::DataType datatype);

pluginStatus_t EfficientNMSInference(EfficientNMSParameters param, const void* boxesInput, const void* scoresInput,
    const void* anchorsInput, void* numDetectionsOutput, void* nmsBoxesOutput, void* nmsScoresOutput,
    void* nmsClassesOutput, void* nmsIndicesOutput, void* workspace, cudaStream_t stream);

#endif
