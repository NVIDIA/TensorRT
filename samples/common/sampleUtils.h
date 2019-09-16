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

#ifndef TRT_SAMPLE_UTILS_H
#define TRT_SAMPLE_UTILS_H

#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <numeric>

#include "NvInfer.h"

namespace sample
{

inline void cudaCheck(cudaError_t ret, std::ostream& err = std::cerr)
{
    if (ret != cudaSuccess)
    {
        err << "Cuda failure: " << ret << std::endl;
        abort();
    }
}

template <typename T>
struct destroyer
{
    void operator()(T* t)
    {
        t->destroy();
    }
};

template <typename T>
using unique_ptr = std::unique_ptr<T, destroyer<T>>;

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

} // namespace sample

#endif // TRT_SAMPLE_UTILS_H
