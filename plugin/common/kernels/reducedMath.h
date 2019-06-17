/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRT_REDUCED_MATH_H
#define TRT_REDUCED_MATH_H
/*
 * Dynamically strength-reduced div and mod
 * Ideas taken from Sean Baxter's MGPU library.
 * These classes provide for reduced complexity division and modulus
 * on integers, for the case where the same divisor or modulus will
 * be used repeatedly.
 */

namespace nvinfer1
{
namespace rt
{

namespace detail
{

void find_divisor(int denom, unsigned int& mul_coeff, unsigned int& shift_coeff);

__host__ __device__ __forceinline__ unsigned int umulhi(unsigned int x, unsigned int y)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 100
    return __umulhi(x, y);
#else
    unsigned long long z = (unsigned long long) x * (unsigned long long) y;
    return (unsigned int) (z >> 32);
#endif
}

/*
 * This is a weird implementation that returns div_up(0,1)=0 but
 * div_up(0,2)=1 (wrong) -- just do not use it with a=0.
 */
__host__ __device__ inline int div_up(int a, int b)
{
    return (a - 1) / b + 1;
}

} // end namespace detail

class reduced_divisor
{
public:
    reduced_divisor() = default;
    __host__ __forceinline__ reduced_divisor(int _y)
        : y(_y)
    {
        detail::find_divisor(y, mul_coeff, shift_coeff);
    }
    __host__ __device__ __forceinline__ reduced_divisor(unsigned _mul_coeff, unsigned _shift_coeff, int _y)
        : mul_coeff(_mul_coeff)
        , shift_coeff(_shift_coeff)
        , y(_y)
    {
    }
    __host__ __device__ __forceinline__ int div(int x) const
    {
        /*
         * if dividing by 1, then find_divisor wouldn't have worked because
         * mul_coeff would have had to be 2^32, which can't be represented,
         * so we have to special case that one.
         */
        return (y != 1) ? detail::umulhi((unsigned int) x, mul_coeff) >> shift_coeff : x;
    }
    __host__ __device__ __forceinline__ int mod(int x) const
    {
        return x - (div(x) * y);
    }
    __host__ __device__ __forceinline__ void divmod(int x, int& q, int& mod) const
    {
        q = div(x);
        mod = x - (q * y);
    }
    __host__ __device__ __forceinline__ int get() const
    {
        return y;
    }
    inline __host__ void get_mul_shift(unsigned& mul, unsigned& shift)
    {
        mul = mul_coeff;
        shift = shift_coeff;
    }

protected:
    unsigned int mul_coeff;
    unsigned int shift_coeff;
    int y;
};

} // namespace rt
} // namespace nvinfer1

#endif // TRT_REDUCED_MATH_H
