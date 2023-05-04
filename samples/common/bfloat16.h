/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace sample
{

//! Implements "Brain Floating Point": like an IEEE FP32,
//! but the significand is only 7 bits instead of 23 bits.
class BFloat16
{
public:
    BFloat16()
        : mRep(0)
    {
    }

    // Rounds to even if there is a tie.
    BFloat16(float x);

    operator float() const;

private:
    //! Value stored in BFloat16 representation.
    uint16_t mRep;
};
BFloat16 operator+(BFloat16 x, BFloat16 y);

} // namespace sample
