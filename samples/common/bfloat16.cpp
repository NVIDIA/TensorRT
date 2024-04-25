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

#include "bfloat16.h"
#include <cstring>

namespace sample
{

BFloat16::operator float() const
{
    static_assert(sizeof(uint32_t) == sizeof(float), "");
    float val{0.F};
    auto bits = static_cast<uint32_t>(mRep) << 16;
    std::memcpy(&val, &bits, sizeof(uint32_t));
    return val;
}

BFloat16::BFloat16(float x)
{
    static_assert(sizeof(uint32_t) == sizeof(float), "");
    uint32_t bits{0};
    std::memcpy(&bits, &x, sizeof(float));

    // FP32 format: 1 sign bit, 8 bit exponent, 23 bit mantissa
    // BF16 format: 1 sign bit, 8 bit exponent, 7 bit mantissa

    // Mask for exponent
    constexpr uint32_t exponent = 0xFFU << 23;

    // Check if exponent is all 1s (NaN or infinite)
    if ((bits & exponent) != exponent)
    {
        // x is finite - round to even
        bits += 0x7FFFU + (bits >> 16 & 1);
    }

    mRep = static_cast<uint16_t>(bits >> 16);
}

BFloat16 operator+(BFloat16 x, BFloat16 y)
{
    return BFloat16(static_cast<float>(x) + static_cast<float>(y));
}

} // namespace sample
