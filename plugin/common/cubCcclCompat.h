/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_CUB_CCCL_COMPAT_H
#define TRT_CUB_CCCL_COMPAT_H

#include <cuda.h>

// Include CUB header
#include <cub/cub.cuh>

#if CUDA_VERSION >= 13000
// Include CCCL headers for CUDA 13.0+
#include <cuda/std/functional>
#endif

namespace compat
{

// Min operation compatibility wrapper
__host__ __device__ __forceinline__ auto getCudaMinOp()
{
#if CUDA_VERSION >= 13000
    return cuda::minimum();
#else
    return cub::Min();
#endif
}

// Max operation compatibility wrapper
__host__ __device__ __forceinline__ auto getCudaMaxOp()
{
#if CUDA_VERSION >= 13000
    return cuda::maximum();
#else
    return cub::Max();
#endif
}

// Sum operation compatibility wrapper
__host__ __device__ __forceinline__ auto getCudaSumOp()
{
#if CUDA_VERSION >= 13000
    return cuda::std::plus<>();
#else
    return cub::Sum();
#endif
}

} // namespace compat

#endif // TRT_CUB_CCCL_COMPAT_H
