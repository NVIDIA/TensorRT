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

#ifndef TRT_EFFICIENT_NMS_PARAMETERS_H
#define TRT_EFFICIENT_NMS_PARAMETERS_H

#include "plugin.h"

using namespace nvinfer1::plugin;
namespace nvinfer1
{
namespace plugin
{

struct EfficientNMSParameters
{
    // Related to NMS Options
    float iouThreshold = 0.5f;
    float scoreThreshold = 0.5f;
    int numOutputBoxes = 100;
    int numOutputBoxesPerClass = -1;
    int backgroundClass = -1;
    bool scoreSigmoid = false;
    int boxCoding = 0;

    // Related to NMS Internals
    int numSelectedBoxes = 4096;
    int scoreBits = 10;
    bool outputONNXIndices = false;

    // Related to Tensor Configuration
    // (These are set by the various plugin configuration methods, no need to define them during plugin creation.)
    int batchSize = -1;
    int numClasses = 1;
    int numBoxElements = -1;
    int numScoreElements = -1;
    int numAnchors = -1;
    bool shareLocation = true;
    bool shareAnchors = true;
    bool boxDecoder = false;
    nvinfer1::DataType datatype = nvinfer1::DataType::kFLOAT;
};

} // namespace plugin
} // namespace nvinfer1

#endif
