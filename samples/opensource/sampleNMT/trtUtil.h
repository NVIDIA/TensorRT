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
#ifndef SAMPLE_NMT_TRT_UTIL_
#define SAMPLE_NMT_TRT_UTIL_

#include "NvInfer.h"
#include <vector>

namespace nmtSample
{
int inferTypeToBytes(nvinfer1::DataType t);

int getVolume(nvinfer1::Dims dims);

// Resize weights matrix to larger size
std::vector<float> resizeWeights(int rows, int cols, int rowsNew, int colsNew, const float* memory);

} // namespace nmtSample

#endif // SAMPLE_NMT_TRT_UTIL_
