/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "bigInt.h"

namespace sample
{

BigInt::BigInt(std::string const& str)
{
    if (str.empty())
    {
        throw std::invalid_argument("Empty string");
    }

    BigInt const ten(10);
    for (char const c : str)
    {
        if (c < '0' || c > '9')
        {
            throw std::invalid_argument("Invalid decimal character in BigInt string");
        }

        auto [mulResult, mulOverflow] = multiplyWithOverflow(*this, ten);
        if (mulOverflow)
        {
            throw std::overflow_error("Number too large for BigInt");
        }

        auto [addResult, addOverflow] = addWithOverflow(mulResult, BigInt(static_cast<uint64_t>(c - '0')));
        if (addOverflow)
        {
            throw std::overflow_error("Number too large for BigInt");
        }

        *this = addResult;
    }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::pair<BigInt, bool> BigInt::multiplyWithOverflow(BigInt const& a, BigInt const& b) noexcept
{
    // Full multiplication into 2*kWordCount words
    std::array<WordType, kWordCount * 2> result{};

    for (uint64_t i = 0; i < kWordCount; ++i)
    {
        if (a.mWords[i] == 0)
        {
            continue;
        }

        WordType carry = 0;
        for (uint64_t j = 0; j < kWordCount; ++j)
        {
            uint64_t const k = i + j;
            if (k >= kWordCount * 2)
            {
                break;
            }

            // 64x64 → 128-bit multiply using four 32-bit half-words.
            // Split: a = aHi*2^32 + aLo, b = bHi*2^32 + bLo
            // Product = aHi*bHi*2^64 + (aHi*bLo + aLo*bHi)*2^32 + aLo*bLo
            uint64_t const aLo = a.mWords[i] & 0xFFFFFFFF;
            uint64_t const aHi = a.mWords[i] >> 32;
            uint64_t const bLo = b.mWords[j] & 0xFFFFFFFF;
            uint64_t const bHi = b.mWords[j] >> 32;

            uint64_t const p0 = aLo * bLo;
            uint64_t const p1 = aLo * bHi;
            uint64_t const p2 = aHi * bLo;
            uint64_t const p3 = aHi * bHi;

            // Combine: prodLo = lower 64 bits, prodHi = upper 64 bits
            uint64_t const mid = p1 + (p0 >> 32);
            uint64_t const midCarry = (mid < p1) ? (uint64_t{1} << 32) : 0;
            uint64_t const mid2 = mid + p2;
            uint64_t const midCarry2 = (mid2 < mid) ? (uint64_t{1} << 32) : 0;

            uint64_t prodLo = (mid2 << 32) | (p0 & 0xFFFFFFFF);
            uint64_t prodHi = p3 + (mid2 >> 32) + midCarry + midCarry2;

            // Add result[k] and carry to the 128-bit product.
            prodLo += result[k];
            if (prodLo < result[k])
            {
                ++prodHi;
            }
            prodLo += carry;
            if (prodLo < carry)
            {
                ++prodHi;
            }
            result[k] = prodLo;
            carry = prodHi;
        }
        if (i + kWordCount < kWordCount * 2)
        {
            result[i + kWordCount] += carry;
        }
    }

    // Check for overflow (any non-zero word in upper half)
    bool overflow = false;
    for (uint64_t i = kWordCount; i < kWordCount * 2; ++i)
    {
        if (result[i] != 0)
        {
            overflow = true;
            break;
        }
    }

    // Copy lower half to result
    BigInt low;
    for (uint64_t i = 0; i < kWordCount; ++i)
    {
        low.mWords[i] = result[i];
    }

    return {low, overflow};
}

std::pair<BigInt, BigInt> BigInt::divideWithRemainder(BigInt const& dividend, BigInt const& divisor)
{
    if (divisor.isZero())
    {
        throw std::domain_error("Division by zero in BigInt");
    }

    if (dividend < divisor)
    {
        return {BigInt(), dividend};
    }

    if (dividend == divisor)
    {
        return {BigInt(1), BigInt()};
    }

    // Binary long division algorithm
    BigInt quotient;
    BigInt remainder;

    int32_t const highBit = dividend.getHighestSetBit();

    for (int32_t i = highBit; i >= 0; --i)
    {
        remainder <<= 1;
        if (dividend.getBit(i))
        {
            remainder.setBit(0, true);
        }

        if (remainder >= divisor)
        {
            remainder -= divisor;
            quotient.setBit(i, true);
        }
    }

    return {quotient, remainder};
}

std::string BigInt::toString() const
{
    if (isZero())
    {
        return "0";
    }

    std::string result;
    BigInt tmp = *this;
    BigInt const ten(10);

    while (!tmp.isZero())
    {
        auto [q, r] = divideWithRemainder(tmp, ten);
        result += static_cast<char>('0' + r.mWords[0]);
        tmp = q;
    }

    std::reverse(result.begin(), result.end());
    return result;
}

} // namespace sample
