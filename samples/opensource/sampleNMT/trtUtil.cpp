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

#include "trtUtil.h"

#include <cassert>
#include <functional>
#include <numeric>

namespace nmtSample
{
int inferTypeToBytes(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kFLOAT: return sizeof(float); break;
    case nvinfer1::DataType::kHALF: return sizeof(int16_t); break;
    default: assert(0); break;
    }
    return 0;
};

int getVolume(nvinfer1::Dims dims)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
}

std::vector<float> resizeWeights(int rows, int cols, int rowsNew, int colsNew, const float* memory)
{
    std::vector<float> result(rowsNew * colsNew);
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            result[row * colsNew + col] = memory[row * cols + col];
        }
    }
    return result;
}

} // namespace nmtSample
