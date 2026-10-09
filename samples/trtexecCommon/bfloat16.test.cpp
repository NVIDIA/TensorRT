/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

using sample::BFloat16;
using NLF32 = std::numeric_limits<float>;

TEST(BFloat16, Type)
{
    static_assert(sizeof(BFloat16) == sizeof(uint16_t), "BFloat16 should be 16 bits!");
    static_assert(alignof(BFloat16) == alignof(uint16_t), "BFloat16 should be 16 bit aligned!");
    EXPECT_EQ(BFloat16{}.operator float(), 0.0F);
}

TEST(BFloat16, Constructors)
{
    EXPECT_EQ(BFloat16{}, 0.0F);
    EXPECT_EQ(BFloat16{1.0F}, 1.0F);
    EXPECT_EQ(BFloat16{-1.0F}, -1.0F);
    EXPECT_EQ(BFloat16{0.0F}, 0.0F);
    EXPECT_EQ(BFloat16{0.5F}, 0.5F);
    // Preserve sign bit, even for zero.
    EXPECT_EQ(std::signbit(static_cast<float>(BFloat16{-0.0F})), std::signbit(-1.0F));
    EXPECT_EQ(std::signbit(static_cast<float>(BFloat16{0.0F})), std::signbit(1.0F));
}

TEST(BFloat16, UnaryMinus)
{
    BFloat16 const bf16Pos = 2.5F;
    BFloat16 const bf16Neg = -bf16Pos;
    EXPECT_EQ(bf16Neg, -2.5F);

    BFloat16 const bf16NegInput = -3.0F;
    BFloat16 const bf16PosResult = -bf16NegInput;
    EXPECT_EQ(bf16PosResult, 3.0F);

    BFloat16 const bf16Zero = 0.0F;
    BFloat16 const bf16NegZero = -bf16Zero;
    EXPECT_EQ(bf16NegZero, 0.0F);
}

TEST(BFloat16, Addition)
{
    BFloat16 const a = 1.5F;
    BFloat16 const b = 2.5F;
    EXPECT_EQ(a + b, 4.0F);

    BFloat16 const c = -1.0F;
    BFloat16 const d = 3.0F;
    EXPECT_EQ(c + d, 2.0F);

    BFloat16 const e = 0.0F;
    BFloat16 const f = 5.0F;
    EXPECT_EQ(e + f, 5.0F);
}

TEST(BFloat16, FloatConversion)
{
    EXPECT_EQ(static_cast<float>(BFloat16{3.14159F}), 3.140625F);
    EXPECT_EQ(static_cast<float>(BFloat16{1000.0F}), 1000.0F);
    EXPECT_EQ(static_cast<float>(BFloat16{0.001F}), 0.0009994507F);
    // Out-of-bounds conversion rounds to infinity.
    EXPECT_TRUE(std::isinf(static_cast<float>(BFloat16{NLF32::max()})));
}

TEST(BFloat16, StreamOutput)
{
    auto toStr = [](auto const& value) {
        std::stringstream ss;
        ss << value;
        return ss.str();
    };
    using namespace std::string_view_literals;
    EXPECT_EQ(toStr(BFloat16(2.718F)), "2.71875"sv);
    EXPECT_EQ(toStr(BFloat16(0.0F)), "0"sv);
    // BFloat16 should match float stringification for special values.
    EXPECT_EQ(toStr(BFloat16{NLF32::infinity()}), std::to_string(NLF32::infinity()));
    EXPECT_EQ(toStr(-BFloat16{NLF32::infinity()}), std::to_string(-NLF32::infinity()));
    EXPECT_EQ(toStr(BFloat16{NLF32::quiet_NaN()}), std::to_string(NLF32::quiet_NaN()));
}

TEST(BFloat16, NumericLimits)
{
    static_assert(!std::numeric_limits<BFloat16>::is_specialized);
}

TEST(BFloat16, TypeTraits)
{
    static_assert(!std::is_arithmetic_v<BFloat16>);
    static_assert(!std::is_scalar_v<BFloat16>);
}

TEST(BFloat16, NaN)
{
    EXPECT_TRUE(std::isnan(static_cast<float>(BFloat16{NLF32::quiet_NaN()})));
    EXPECT_TRUE(std::isnan(BFloat16{} / BFloat16{}));
    EXPECT_TRUE(std::isnan(static_cast<float>(BFloat16{BFloat16{} / BFloat16{}})));
    EXPECT_TRUE(std::isnan(BFloat16{NLF32::quiet_NaN()} / BFloat16{}));
    EXPECT_TRUE(std::isnan(BFloat16{NLF32::infinity()} - BFloat16{NLF32::infinity()}));
}

TEST(BFloat16, EdgeCases)
{
    auto const bf16Large = BFloat16(1e38F);
    EXPECT_FALSE(std::isinf(static_cast<float>(bf16Large)));
    EXPECT_LT(BFloat16(1e30F), bf16Large);

    auto const bf16Small = BFloat16(1e-38F);
    EXPECT_LT(0.0F, static_cast<float>(bf16Small));
    EXPECT_LT(bf16Small, BFloat16(1e-30F));
}

TEST(BFloat16, PrecisionAndRounding)
{
    // BFloat16 has lower precision than float; values are rounded (to even).
    constexpr float kPRECISE_VALUE = 1.0F + 1e-7F;
    EXPECT_NEAR(static_cast<float>(BFloat16{kPRECISE_VALUE}), kPRECISE_VALUE, 1e-3F);
}
