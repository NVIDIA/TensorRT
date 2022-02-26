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

#include "common.h"
#include "trtUtil.h"
#include <functional>
#include <numeric>

namespace nmtSample
{
int32_t inferTypeToBytes(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kFLOAT: return sizeof(float); break;
    case nvinfer1::DataType::kHALF: return sizeof(int16_t); break;
    default: ASSERT(0); break;
    }
    return 0;
}

int32_t getVolume(nvinfer1::Dims dims)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int32_t>());
}

std::vector<float> resizeWeights(int32_t rows, int32_t cols, int32_t rowsNew, int32_t colsNew, const float* memory)
{
    std::vector<float> result(rowsNew * colsNew);
    for (int32_t row = 0; row < rows; row++)
    {
        for (int32_t col = 0; col < cols; col++)
        {
            result[row * colsNew + col] = memory[row * cols + col];
        }
    }
    return result;
}

} // namespace nmtSample
