/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _REDUCED_MATH_PLUGIN_H
#define _REDUCED_MATH_PLUGIN_H
#include <cstdint>
// Dynamically strength-reduced div and mod
//
// Ideas taken from Sean Baxter's MGPU library.
// These classes provide for reduced complexity division and modulus
// on integers, for the case where the same divisor or modulus will
// be used repeatedly.

namespace nvinfer1
{
namespace plugin
{
namespace detail
{

void findDivisor(int32_t denom, uint32_t& mul_coeff, uint32_t& shift_coeff);

__host__ __device__ __forceinline__ uint32_t umulhi(uint32_t x, uint32_t y)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 100
    return __umulhi(x, y);
#else
    uint64_t z = (uint64_t) x * (uint64_t) y;
    return (uint32_t) (z >> 32);
#endif
}

// This is a weird implementation that returns div_up(0,1)=0 but
// div_up(0,2)=1 (wrong) -- just do not use it with a=0.
__host__ __device__ inline int32_t div_up(int32_t a, int32_t b)
{
    return (a - 1) / b + 1;
}

} // end namespace detail

class ReducedDivisor
{
public:
    ReducedDivisor() {}
    __host__ __forceinline__ ReducedDivisor(int32_t _y)
        : y(_y)
    {
        detail::findDivisor(y, mul_coeff, shift_coeff);
    }
    __host__ __device__ __forceinline__ ReducedDivisor(uint32_t _mul_coeff, uint32_t _shift_coeff, int32_t _y)
        : mul_coeff(_mul_coeff)
        , shift_coeff(_shift_coeff)
        , y(_y)
    {
    }
    __host__ __device__ __forceinline__ int32_t div(int32_t x) const
    {
        // if dividing by 1, then findDivisor wouldn't have worked because
        // mul_coeff would have had to be 2^32, which can't be represented,
        // so we have to special case that one.
        return (y != 1) ? detail::umulhi((uint32_t) x, mul_coeff) >> shift_coeff : x;
    }
    __host__ __device__ __forceinline__ int32_t mod(int32_t x) const
    {
        return x - (div(x) * y);
    }
    __host__ __device__ __forceinline__ void divmod(int32_t x, int32_t& q, int32_t& mod) const
    {
        q = div(x);
        mod = x - (q * y);
    }
    __host__ __device__ __forceinline__ int32_t get() const
    {
        return y;
    }
    inline __host__ void get_mul_shift(uint32_t& mul, uint32_t& shift)
    {
        mul = mul_coeff;
        shift = shift_coeff;
    }

protected:
    uint32_t mul_coeff{};
    uint32_t shift_coeff{};
    int32_t y{};
};

} // namespace plugin

} // namespace nvinfer1
#endif /*_REDUCED_MATH_PLUGIN_H*/
