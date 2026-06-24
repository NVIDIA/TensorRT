/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda/std/functional>

namespace compat
{

// Min operation compatibility wrapper
__host__ __device__ __forceinline__ auto getCudaMinOp()
{
    return cuda::minimum();
}

// Max operation compatibility wrapper
__host__ __device__ __forceinline__ auto getCudaMaxOp()
{
    return cuda::maximum();
}

// Sum operation compatibility wrapper
__host__ __device__ __forceinline__ auto getCudaSumOp()
{
    return cuda::std::plus<>();
}

} // namespace compat

#endif // TRT_CUB_CCCL_COMPAT_H
