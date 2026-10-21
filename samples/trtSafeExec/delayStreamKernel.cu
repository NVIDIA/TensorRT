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

#include <limits>

namespace
{
__device__ __forceinline__ uint64_t readGlobalTimer()
{
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__global__ void delayKernel(uint64_t nanoSeconds)
{
    uint64_t const start{readGlobalTimer()};
    while (readGlobalTimer() - start < nanoSeconds)
    {
        // Busy-wait so that subsequent work can be submitted to the stream while this kernel is running.
    }
}
} // namespace

namespace nvinfer1
{
cudaError_t delayStream(cudaStream_t stream, std::chrono::duration<float, std::milli> duration) noexcept
{
    using FloatMilliseconds = std::chrono::duration<float, std::milli>;
    if (duration < FloatMilliseconds::zero())
    {
        return cudaErrorInvalidValue;
    }
    if (duration == FloatMilliseconds::zero())
    {
        return cudaSuccess;
    }
    constexpr double kNANOSECONDS_PER_MILLISECOND{1000000.0};
    auto const nanoSeconds = kNANOSECONDS_PER_MILLISECOND * static_cast<double>(duration.count());
    if (!(nanoSeconds < static_cast<double>(std::numeric_limits<uint64_t>::max())))
    {
        // This comparison also rejects NaN and infinity before the integer conversion.
        return cudaErrorInvalidValue;
    }
    delayKernel<<<1, 1, 0, stream>>>(static_cast<uint64_t>(nanoSeconds));
    return cudaGetLastError();
}
} // namespace nvinfer1
