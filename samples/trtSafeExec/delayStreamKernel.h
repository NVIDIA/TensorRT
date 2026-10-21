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

#ifndef DELAY_STREAM_KERNEL_H
#define DELAY_STREAM_KERNEL_H

#include <chrono>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace nvinfer1
{
//! \brief Launch a kernel on \p stream that busy-waits for \p duration, delaying subsequent work on the stream.
//!
//! \return cudaSuccess on success, cudaErrorInvalidValue if \p duration is negative or not representable as a
//!         nanosecond count, otherwise the error from the kernel launch.
cudaError_t delayStream(cudaStream_t stream, std::chrono::duration<float, std::milli> duration) noexcept;
} // namespace nvinfer1
#endif // DELAY_STREAM_KERNEL_H
