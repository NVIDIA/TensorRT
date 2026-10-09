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

#include "half.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

using half_float::half;
using NLF32 = std::numeric_limits<float>;
using NLHalf = std::numeric_limits<half>;

TEST(Half, Type)
{
    static_assert(sizeof(half) == sizeof(uint16_t), "half should be 16 bits!");
    static_assert(alignof(half) == alignof(uint16_t), "half should be 16 bit aligned!");
    EXPECT_EQ(half{}.operator float(), 0.0F);
}

TEST(Half, Constructors)
{
    EXPECT_EQ(half{}, 0.0F);
    EXPECT_EQ(half{1.0F}, 1.0F);
    EXPECT_EQ(half{-1.0F}, -1.0F);
    EXPECT_EQ(half{0.0F}, 0.0F);
    EXPECT_EQ(half{0.5F}, 0.5F);
    // Preserve sign bit, even for zero.
    EXPECT_EQ(std::signbit(static_cast<float>(half{-0.0F})), std::signbit(-1.0F));
    EXPECT_EQ(std::signbit(static_cast<float>(-half{0.0F})), std::signbit(-1.0F));
    EXPECT_EQ(std::signbit(static_cast<float>(half{0.0F})), std::signbit(1.0F));
}

TEST(Half, UnaryMinus)
{
    half const pos(2.5F);
    half const neg = -pos;
    EXPECT_EQ(neg, -2.5F);

    half const negInput(-3.0F);
    half const posResult = -negInput;
    EXPECT_EQ(posResult, 3.0F);

    half const zero(0.0F);
    half const negZero = -zero;
    EXPECT_EQ(negZero, 0.0F);
}

TEST(Half, Addition)
{
    half const a(1.5F);
    half const b(2.5F);
    EXPECT_EQ(a + b, 4.0F);

    half const c(-1.0F);
    half const d(3.0F);
    EXPECT_EQ(c + d, 2.0F);

    half const e(0.0F);
    half const f(5.0F);
    EXPECT_EQ(e + f, 5.0F);
}

TEST(Half, FloatConversion)
{
    EXPECT_EQ(static_cast<float>(half{3.14159F}), 3.140625F);
    EXPECT_EQ(static_cast<float>(half{1000.0F}), 1000.0F);
    EXPECT_EQ(static_cast<float>(half{0.001F}), 0.0010004043F);
    // Out-of-bounds conversion rounds to infinity.
    EXPECT_TRUE(std::isinf(static_cast<float>(half{NLF32::max()})));
}

TEST(Half, StreamOutput)
{
    auto toStr = [](auto const& value) {
        std::stringstream ss;
        ss << value;
        return ss.str();
    };
    using namespace std::string_view_literals;
    EXPECT_EQ(toStr(half(2.718F)), "2.71875"sv);
    EXPECT_EQ(toStr(half(0.0F)), "0"sv);
    // half should match float stringification for special values.
    EXPECT_EQ(toStr(half{NLF32::infinity()}), std::to_string(NLF32::infinity()));
    EXPECT_EQ(toStr(-half{NLF32::infinity()}), std::to_string(-NLF32::infinity()));
    EXPECT_EQ(toStr(half{NLF32::quiet_NaN()}), std::to_string(NLF32::quiet_NaN()));
}

TEST(Half, NumericLimits)
{
    static_assert(std::numeric_limits<half>::is_specialized);
    EXPECT_FALSE(std::numeric_limits<half>::is_integer);
    EXPECT_TRUE(std::numeric_limits<half>::has_infinity);

    constexpr auto kINF = NLHalf::infinity();
    EXPECT_TRUE(isinf(kINF));
    EXPECT_LT(0.0F, kINF);
    EXPECT_TRUE(isinf(-kINF));
    EXPECT_LT(-kINF, 0.0F);

    constexpr auto kMAX_VAL = NLHalf::max();
    EXPECT_FALSE(isinf(kMAX_VAL));
    EXPECT_EQ(kMAX_VAL, 65504.0F);
    EXPECT_EQ(half(2.0F * 65505.0F), NLHalf::infinity());

    constexpr auto kNAN = NLHalf::quiet_NaN();
    EXPECT_TRUE(isnan(kNAN));
    EXPECT_NE(kNAN, kNAN); // NaN is not equal to itself.
}

TEST(Half, TypeTraits)
{
    static_assert(!std::is_arithmetic_v<half>);
    static_assert(!std::is_scalar_v<half>);
}

TEST(Half, NaN)
{
    EXPECT_TRUE(isnan(half{NLF32::quiet_NaN()}));
    EXPECT_TRUE(isnan(half{} / half{}));
    EXPECT_TRUE(isnan(half{half{} / half{}}));
    EXPECT_TRUE(isnan(half{NLF32::quiet_NaN()} / half{}));
    EXPECT_TRUE(isnan(half{NLF32::infinity()} - half{NLF32::infinity()}));
}

TEST(Half, EdgeCases)
{
    auto const f16Large = half(1e38F);
    EXPECT_TRUE(std::isinf(static_cast<float>(f16Large)));
    EXPECT_LT(NLHalf::max(), f16Large);

    auto const f16Small = half(0x1p-24F); // smallest subnormal
    EXPECT_LT(0.0F, f16Small);
    // Note: this library uses HALF_ROUND_TIES_TO_EVEN=0 (ties away from zero) by default.
    // 0x1p-24 / 2 = 0x1p-25, which is equidistant between 0 and 0x1p-24. Ties-away-from-zero
    // rounds up to 0x1p-24 rather than down to 0 as IEEE round-to-nearest-even would.
    EXPECT_EQ(f16Small, half(f16Small / 2.0F));
}

TEST(Half, PrecisionAndRounding)
{
    constexpr float kPRECISE_VALUE = 1.0F + 1e-7F;
    EXPECT_NEAR(static_cast<float>(half{kPRECISE_VALUE}), kPRECISE_VALUE, 1e-3F);
}
