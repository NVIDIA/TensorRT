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
#include <cstdint>
namespace nvinfer1
{
namespace plugin
{
namespace detail
{

// Count leading zeros - start from most significant bit.
int clz(int x)
{
    for (int i = 31; i >= 0; --i)
    {
        if ((1U << i) & x)
        {
            return 31 - i;
        }
    }
    return 32;
}

#define CUDNN_IS_POW_2(x) (0 == ((x) & ((x) -1)))

int find_log_2(int x, bool round_up = false)
{
    int a = 31 - clz(x);
    if (round_up)
    {
        a += !CUDNN_IS_POW_2(x);
    }
    return a;
}

void findDivisor(int denom,
                  unsigned int& mul_coeff, unsigned int& shift_coeff)
{
    if (denom == 0)
    {
        return;
    }
    if (denom == 1)
    {
        // if dividing by 1, reduced math doesn't work because mul_coeff would
        // need to be 2^32, which doesn't fit into unsigned int.  the div()
        // routine handles this special case separately.
        mul_coeff = 0;
        shift_coeff = 0;
        return;
    }
    // To express the division N/D in terms of a multiplication, what we first
    // imagine is simply N*(1/D).  However, 1/D will always evaluate to 0 (for D>1),
    // so we need another way.  There's nothing that says we have to use exactly
    // the fraction 1/D; instead it could be any X/Y that reduces to 1/D (i.e.,
    // Y=X*D), or at least to "close enough" to it.  If we pick Y that is a power
    // of two, then the N*(X/Y) can be N*X followed by a right-shift by some amount.
    // The power of two we should pick should be at least 2^32, because in the
    // div() routine we'll use umulhi(), which returns only the upper 32 bits --
    // this being equivalent to a right-shift by 32.  But we might want a higher
    // power of two for better accuracy depending on the magnitude of the denominator.
    // Once we've picked Y, then X [our mul_coeff value] is simply Y/D, rounding up,
    // and we save shift_coeff as whatever further shift we have to do beyond
    // what the umulhi() implies.
    uint32_t p = 31 + find_log_2(denom, true);
    uint32_t m = ((1ull << p) + (uint32_t) denom - 1) / (uint32_t) denom;
    mul_coeff = m;
    shift_coeff = p - 32;
}

} // namespace detail

} // namespace plugin

} // namespace nvinfer1
