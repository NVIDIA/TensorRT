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

#ifndef TRT_GLOBAL_TIMER_KERNEL_H
#define TRT_GLOBAL_TIMER_KERNEL_H

#include <cstdint>
#include <cuda_runtime_api.h>

namespace sample
{
//! Launch a single-thread kernel that writes the current value of the PTX
//! %globaltimer register (GPU timer in ns) to \p dTimestamp on \p stream.
//!
//! Used as a replacement for cudaEventElapsedTime() when Confidential Compute
//! is enabled, where cudaEventElapsedTime() is documented to be unreliable.
//!
//! \param dTimestamp Device pointer to a uint64_t. Must be non-null and point
//!     to valid device memory reachable from \p stream.
//! \param stream CUDA stream on which to launch the kernel.
//! \return Result of \c cudaGetLastError() after the launch. This reports
//!     synchronous launch errors only; asynchronous execution errors will
//!     surface on a subsequent \c cudaEventRecord /
//!     \c cudaStreamSynchronize / similar.
[[nodiscard]] cudaError_t launchGlobalTimerKernel(uint64_t* dTimestamp, cudaStream_t stream) noexcept;
} // namespace sample

#endif // TRT_GLOBAL_TIMER_KERNEL_H
