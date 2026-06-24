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

#ifndef TRT_SAMPLE_BIG_INT_H
#define TRT_SAMPLE_BIG_INT_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace sample
{

//!
//! \class BigInt
//! \brief A class for arbitrary-precision unsigned integers (8192 bits).
//!
//! This class provides support for very large unsigned integers, primarily used
//! for counting and indexing in build path expression expansion where the number
//! of combinations can be astronomically large (e.g., 2^1000 combinations).
//!
//! Key operations:
//! - Construction from uint64_t or decimal string
//! - Comparison operators for loop termination
//! - Increment operator for loop counting
//! - Division/modulo for mixed-radix index decomposition
//! - String conversion for display
//!
class BigInt
{
public:
    //! Number of bits in the integer (8192 = 128 * 64)
    static constexpr uint64_t kBitCount = 8192;
    //! Number of 64-bit words
    static constexpr uint64_t kWordCount = kBitCount / 64; // 128 words
    using WordType = uint64_t;

    //! \brief Default constructor. Initializes to zero.
    constexpr BigInt() noexcept = default;

    //! \brief Construct from a 64-bit unsigned integer.
    //! \param[in] value The initial value.
    constexpr BigInt(uint64_t value) noexcept
    {
        mWords[0] = value;
    }

    //! \brief Construct from a decimal string.
    //! \param[in] str The decimal string representation.
    //! \throws std::invalid_argument If the string is empty or contains invalid characters.
    //! \throws std::overflow_error If the number is too large.
    explicit BigInt(std::string const& str);

    // Default copy and move operations
    constexpr BigInt(BigInt const&) noexcept = default;
    constexpr BigInt& operator=(BigInt const&) noexcept = default;
    constexpr BigInt(BigInt&&) noexcept = default;
    constexpr BigInt& operator=(BigInt&&) noexcept = default;

    //! \brief Check if the value is zero.
    //! \return True if zero.
    constexpr bool isZero() const noexcept
    {
        for (uint64_t i = 0; i < kWordCount; ++i)
        {
            if (mWords[i] != 0)
            {
                return false;
            }
        }
        return true;
    }

    //! \brief Get the bit value at a specific position.
    //! \param[in] pos The bit position (0 = LSB).
    //! \return The bit value.
    constexpr bool getBit(uint64_t pos) const noexcept
    {
        if (pos >= kBitCount)
        {
            return false;
        }
        uint64_t const wordIdx = pos / 64;
        uint64_t const bitIdx = pos % 64;
        return (mWords[wordIdx] >> bitIdx) & 1;
    }

    //! \brief Set the bit value at a specific position.
    //! \param[in] pos The bit position (0 = LSB).
    //! \param[in] value The bit value to set.
    constexpr void setBit(uint64_t pos, bool value = true) noexcept
    {
        if (pos >= kBitCount)
        {
            return;
        }
        uint64_t const wordIdx = pos / 64;
        uint64_t const bitIdx = pos % 64;
        if (value)
        {
            mWords[wordIdx] |= (WordType{1} << bitIdx);
        }
        else
        {
            mWords[wordIdx] &= ~(WordType{1} << bitIdx);
        }
    }

    //! \brief Get the position of the highest set bit.
    //! \return The position (0-indexed), or -1 if zero.
    constexpr int32_t getHighestSetBit() const noexcept
    {
        for (int32_t i = kWordCount - 1; i >= 0; --i)
        {
            if (mWords[i] != 0)
            {
                // Count leading zeros portably (no compiler intrinsics).
                uint64_t w = mWords[i];
                int32_t bit = 63;
                while (bit > 0 && (w & (uint64_t{1} << bit)) == 0)
                {
                    --bit;
                }
                return i * 64 + bit;
            }
        }
        return -1;
    }

    // ========================================================================
    // Comparison operators
    // ========================================================================

    constexpr bool operator==(BigInt const& other) const noexcept
    {
        // Manual element-by-element comparison (std::array::operator== is not constexpr in C++17)
        for (uint64_t i = 0; i < kWordCount; ++i)
        {
            if (mWords[i] != other.mWords[i])
            {
                return false;
            }
        }
        return true;
    }

    constexpr bool operator!=(BigInt const& other) const noexcept
    {
        return !(*this == other);
    }

    //! \brief Less-than comparison.
    //! Compares from most significant word down.
    constexpr bool operator<(BigInt const& other) const noexcept
    {
        for (int32_t i = kWordCount - 1; i >= 0; --i)
        {
            if (mWords[i] < other.mWords[i])
            {
                return true;
            }
            if (mWords[i] > other.mWords[i])
            {
                return false;
            }
        }
        return false;
    }

    constexpr bool operator<=(BigInt const& other) const noexcept
    {
        return !(other < *this);
    }

    constexpr bool operator>(BigInt const& other) const noexcept
    {
        return other < *this;
    }

    constexpr bool operator>=(BigInt const& other) const noexcept
    {
        return !(*this < other);
    }

    // ========================================================================
    // Arithmetic operators
    // ========================================================================

    //! \brief Add with overflow detection.
    //! \return Pair of (result, overflow_flag).
    static constexpr std::pair<BigInt, bool> addWithOverflow(BigInt const& a, BigInt const& b) noexcept
    {
        BigInt result;
        uint64_t carry = 0;
        for (uint64_t i = 0; i < kWordCount; ++i)
        {
            // Add with carry using plain uint64_t. Overflow is detected by comparing
            // the result against the operand: if sum < a then overflow occurred.
            uint64_t sum = a.mWords[i] + b.mWords[i];
            uint64_t c1 = (sum < a.mWords[i]) ? 1U : 0U;
            uint64_t sum2 = sum + carry;
            uint64_t c2 = (sum2 < sum) ? 1U : 0U;
            result.mWords[i] = sum2;
            carry = c1 + c2;
        }
        return {result, carry != 0};
    }

    //! \brief Subtract with underflow detection.
    //! \return Pair of (result, underflow_flag).
    static constexpr std::pair<BigInt, bool> subWithUnderflow(BigInt const& a, BigInt const& b) noexcept
    {
        BigInt result;
        uint64_t borrow = 0;
        for (uint64_t i = 0; i < kWordCount; ++i)
        {
            // Subtract with borrow using plain uint64_t.
            // Borrow is detected by: if a < b+borrow, then we borrowed from the next word.
            uint64_t sub = a.mWords[i] - b.mWords[i];
            uint64_t b1 = (a.mWords[i] < b.mWords[i]) ? 1U : 0U;
            uint64_t sub2 = sub - borrow;
            uint64_t b2 = (sub < borrow) ? 1U : 0U;
            result.mWords[i] = sub2;
            borrow = b1 + b2;
        }
        return {result, borrow != 0};
    }

    //! \brief Multiply with overflow detection.
    //! \return Pair of (result, overflow_flag).
    static std::pair<BigInt, bool> multiplyWithOverflow(BigInt const& a, BigInt const& b) noexcept;

    //! \brief Divide with remainder.
    //! \param[in] dividend The dividend.
    //! \param[in] divisor The divisor.
    //! \return Pair of (quotient, remainder).
    //! \throws std::domain_error If divisor is zero.
    static std::pair<BigInt, BigInt> divideWithRemainder(BigInt const& dividend, BigInt const& divisor);

    constexpr BigInt operator+(BigInt const& other) const noexcept
    {
        return addWithOverflow(*this, other).first;
    }

    constexpr BigInt& operator+=(BigInt const& other) noexcept
    {
        *this = *this + other;
        return *this;
    }

    constexpr BigInt operator-(BigInt const& other) const noexcept
    {
        return subWithUnderflow(*this, other).first;
    }

    constexpr BigInt& operator-=(BigInt const& other) noexcept
    {
        *this = *this - other;
        return *this;
    }

    BigInt operator*(BigInt const& other) const noexcept
    {
        return multiplyWithOverflow(*this, other).first;
    }

    BigInt operator/(BigInt const& other) const
    {
        return divideWithRemainder(*this, other).first;
    }

    BigInt operator%(BigInt const& other) const
    {
        return divideWithRemainder(*this, other).second;
    }

    // ========================================================================
    // Shift operators (needed for division algorithm)
    // ========================================================================

    constexpr BigInt operator<<(uint64_t shift) const noexcept
    {
        if (shift >= kBitCount)
        {
            return BigInt();
        }
        if (shift == 0)
        {
            return *this;
        }

        BigInt result;
        uint64_t const wordShift = shift / 64;
        uint64_t const bitShift = shift % 64;

        if (bitShift == 0)
        {
            for (uint64_t i = wordShift; i < kWordCount; ++i)
            {
                result.mWords[i] = mWords[i - wordShift];
            }
        }
        else
        {
            for (uint64_t i = wordShift; i < kWordCount; ++i)
            {
                result.mWords[i] = mWords[i - wordShift] << bitShift;
                if (i > wordShift)
                {
                    result.mWords[i] |= mWords[i - wordShift - 1] >> (64 - bitShift);
                }
            }
        }
        return result;
    }

    constexpr BigInt& operator<<=(uint64_t shift) noexcept
    {
        *this = *this << shift;
        return *this;
    }

    // ========================================================================
    // Increment/Decrement operators
    // ========================================================================

    //! \brief Pre-increment operator.
    //! Handles carry propagation across words.
    constexpr BigInt& operator++() noexcept
    {
        for (uint64_t i = 0; i < kWordCount; ++i)
        {
            if (++mWords[i] != 0)
            {
                break; // No carry, done
            }
            // Carry propagates to next word
        }
        return *this;
    }

    constexpr BigInt operator++(int32_t) noexcept
    {
        BigInt tmp = *this;
        ++(*this);
        return tmp;
    }

    //! \brief Pre-decrement operator.
    //! Handles borrow propagation across words.
    constexpr BigInt& operator--() noexcept
    {
        for (uint64_t i = 0; i < kWordCount; ++i)
        {
            if (mWords[i]-- != 0)
            {
                break; // No borrow, done
            }
            // Borrow propagates to next word
        }
        return *this;
    }

    constexpr BigInt operator--(int32_t) noexcept
    {
        BigInt tmp = *this;
        --(*this);
        return tmp;
    }

    // ========================================================================
    // String conversion
    // ========================================================================

    //! \brief Convert to decimal string representation.
    //! \return The decimal string.
    std::string toString() const;

    //! \brief Get the lowest 64 bits as uint64_t.
    //! Useful when the value is known to fit in 64 bits.
    constexpr uint64_t toUint64() const noexcept
    {
        return mWords[0];
    }

private:
    std::array<WordType, kWordCount> mWords{};
};

} // namespace sample

#endif // TRT_SAMPLE_BIG_INT_H
