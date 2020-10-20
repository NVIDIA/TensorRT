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

#ifndef SAMPLE_NMT_CUDA_ERROR_
#define SAMPLE_NMT_CUDA_ERROR_

#include <cassert>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(callstr)                                                                                            \
    {                                                                                                                  \
        cudaError_t error_code = callstr;                                                                              \
        if (error_code != cudaSuccess)                                                                                 \
        {                                                                                                              \
            std::cerr << "CUDA error " << error_code << ": \"" << cudaGetErrorString(error_code) << "\" at "           \
                      << __FILE__ << ":" << __LINE__ << std::endl;                                                     \
            assert(0);                                                                                                 \
        }                                                                                                              \
    }

#endif // SAMPLE_NMT_CUDA_ERROR_
