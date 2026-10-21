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

#include "globalTimerKernel.h"

namespace
{
__global__ void readGlobalTimerKernel(uint64_t* timestamp)
{
    if (timestamp == nullptr)
    {
        return;
    }
    uint64_t ts;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));
    *timestamp = ts;
}
} // namespace

namespace sample
{
// NOTE: cudaGetLastError() only surfaces synchronous launch errors (invalid
// configuration, bad stream, etc.). Any asynchronous execution errors from
// the kernel itself — e.g. a dereference of a bad device pointer — will not
// be reported here; they become visible only on a subsequent synchronization
// point such as cudaEventRecord / cudaStreamSynchronize / cudaDeviceSynchronize.
cudaError_t launchGlobalTimerKernel(uint64_t* dTimestamp, cudaStream_t stream) noexcept
{
    readGlobalTimerKernel<<<1, 1, 0, stream>>>(dTimestamp);
    return cudaGetLastError();
}
} // namespace sample
