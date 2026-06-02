/*
* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "delayStreamKernel.h"

#include <tuple>

#ifndef TRT_SAFETY_INFERENCE_ONLY
namespace
{
__global__ void delayKernel(long long nanoSeconds)
{
    // It is supported with compute capability 7.0 or higher.
    __nanosleep(nanoSeconds);
}
} // namespace
#endif // TRT_SAFETY_INFERENCE_ONLY

namespace nvinfer1
{
cudaError_t delayStream(cudaStream_t stream, float timeInMsec) noexcept
{
#ifndef TRT_SAFETY_INFERENCE_ONLY
    auto nanoSeconds = static_cast<long long>(1000000 * timeInMsec);
    delayKernel<<<1, 1, 0, stream>>>(nanoSeconds);
    return cudaGetLastError();
#else
    // QNX SafeCUDA does not support PTX JIT or __cudaLaunchKernel; the delay is
    // optional timing-measurement padding and has no effect on inference correctness.
    std::ignore = stream;
    std::ignore = timeInMsec;
    return cudaSuccess;
#endif // TRT_SAFETY_INFERENCE_ONLY
}
} // namespace nvinfer1
