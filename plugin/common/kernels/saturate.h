/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef TRT_SATURATE_H
#define TRT_SATURATE_H

#include <array>

template <typename T_BBOX>
__device__ T_BBOX saturate(T_BBOX v)
{
    return max(min(v, T_BBOX(1)), T_BBOX(0));
}

template <>
inline __device__ __half saturate(__half v)
{
#if __CUDA_ARCH__ >= 800
    return __hmax(__hmin(v, __half(1)), __half(0));
#elif __CUDA_ARCH__ >= 530
    return __hge(v, __half(1)) ? __half(1) : (__hle(v, __half(0)) ? __half(0) : v);
#else
    return max(min(v, float(1)), float(0));
#endif
}

#endif // TRT_SATURATE_H
