/*
 * SPDX-License-Identifier: Apache-2.0
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
