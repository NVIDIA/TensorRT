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

#include "transpose.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace nvinfer1::plugin
{

/// Transpose the last two dimensions of a 3D tensor [outer, rows, cols] -> [outer, cols, rows].
/// Each thread handles one element.
template <typename T>
__global__ void batchedTranspose2DKernel(
    T const* __restrict__ src, T* __restrict__ dst, int32_t outer, int32_t rows, int32_t cols)
{
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const total = static_cast<int64_t>(outer) * rows * cols;
    if (idx >= total)
    {
        return;
    }

    int64_t const matSize = static_cast<int64_t>(rows) * cols;
    int32_t const o = static_cast<int32_t>(idx / matSize);
    int32_t const rem = static_cast<int32_t>(idx % matSize);
    int32_t const r = rem / cols;
    int32_t const c = rem % cols;

    // src[o][r][c] -> dst[o][c][r]
    dst[static_cast<int64_t>(o) * matSize + static_cast<int64_t>(c) * rows + r] = src[idx];
}

template <typename T>
void launchBatchedTranspose2D(
    T const* src, T* dst, int32_t outer, int32_t rows, int32_t cols, cudaStream_t stream)
{
    int64_t const total = static_cast<int64_t>(outer) * rows * cols;
    if (total == 0)
    {
        return;
    }
    int32_t constexpr kBlockSize = 256;
    int32_t const numBlocks = static_cast<int32_t>((total + kBlockSize - 1) / kBlockSize);
    batchedTranspose2DKernel<T><<<numBlocks, kBlockSize, 0, stream>>>(src, dst, outer, rows, cols);
}

// Explicit instantiations.
template void launchBatchedTranspose2D<float>(float const*, float*, int32_t, int32_t, int32_t, cudaStream_t);
template void launchBatchedTranspose2D<half>(half const*, half*, int32_t, int32_t, int32_t, cudaStream_t);
template void launchBatchedTranspose2D<__nv_bfloat16>(
    __nv_bfloat16 const*, __nv_bfloat16*, int32_t, int32_t, int32_t, cudaStream_t);
template void launchBatchedTranspose2D<int32_t>(
    int32_t const*, int32_t*, int32_t, int32_t, int32_t, cudaStream_t);

} // namespace nvinfer1::plugin
