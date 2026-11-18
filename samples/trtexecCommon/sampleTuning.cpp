/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "sampleTuning.h"
#include "common.h"
#include "nlohmann/json.hpp"
#include "sampleUtils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_set>

namespace sample
{

std::vector<std::string> splitPipeDelimited(std::string const& str)
{
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, '|'))
    {
        uint64_t const start = token.find_first_not_of(" \t");
        uint64_t const end = token.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos)
        {
            result.push_back(token.substr(start, end - start + 1));
        }
    }
    return result;
}

// ============================================================================
// BuildRouteKnobDatabase Implementation
// ============================================================================

bool BuildRouteKnobDatabase::loadFromJsonString(std::string const& jsonStr)
{
    mKnobs.clear();
    mKnobOrder.clear();
    mTunerVersion = "unknown";

    if (jsonStr.empty())
    {
        return false;
    }

    try
    {
        // Parse JSON using nlohmann/json
        nlohmann::json const root = nlohmann::json::parse(jsonStr);

        // Extract tuner version from the top-level JSON object.
        if (root.contains("tuner_version") && root["tuner_version"].is_string())
        {
            mTunerVersion = root["tuner_version"].get<std::string>();
        }

        // Check for "tuner_options" array
        if (!root.contains("tuner_options") || !root["tuner_options"].is_array())
        {
            return false;
        }

        // Iterate over tuner_options array
        for (auto const& item : root["tuner_options"])
        {
            if (!item.is_object())
            {
                continue;
            }

            BuildRouteKnobDef knob;

            // Extract fields from JSON object
            if (item.contains("option") && item["option"].is_string())
            {
                knob.mOption = item["option"].get<std::string>();
            }
            if (item.contains("allowed_values") && item["allowed_values"].is_string())
            {
                knob.mAllowedValues = item["allowed_values"].get<std::string>();
            }
            if (item.contains("default_value") && item["default_value"].is_string())
            {
                knob.mDefaultValue = item["default_value"].get<std::string>();
            }
            if (item.contains("help") && item["help"].is_string())
            {
                knob.mHelp = item["help"].get<std::string>();
            }

            // Parse allowed values and add to database
            if (!knob.mOption.empty())
            {
                knob.mValues = parseAllowedValues(knob.mAllowedValues);
                knob.mIsBounded = !knob.mValues.empty();
                mKnobOrder.push_back(knob.mOption);
                mKnobs[knob.mOption] = std::move(knob);
            }
        }
    }
    catch (nlohmann::json::exception const&)
    {
        // JSON parsing failed
        return false;
    }

    return !mKnobs.empty();
}

std::string BuildRouteKnobDatabase::buildDefaultPath() const
{
    // Build a space-separated string of "knob=default_value" pairs,
    // using the insertion order preserved from the original JSON.
    std::string result;
    for (auto const& name : mKnobOrder)
    {
        auto it = mKnobs.find(name);
        if (it == mKnobs.end())
        {
            continue;
        }
        if (!result.empty())
        {
            result += " ";
        }
        result += it->second.mOption + "=" + it->second.mDefaultValue;
    }
    return result;
}

bool BuildRouteKnobDatabase::hasKnob(std::string const& knobName) const
{
    return mKnobs.contains(knobName);
}

BuildRouteKnobDef const* BuildRouteKnobDatabase::getKnob(std::string const& knobName) const
{
    auto const it = mKnobs.find(knobName);
    return it != mKnobs.end() ? &it->second : nullptr;
}

bool BuildRouteKnobDatabase::validateValues(std::string const& knobName, std::vector<std::string> const& values) const
{
    BuildRouteKnobDef const* knob = getKnob(knobName);
    if (knob == nullptr)
    {
        return false;
    }
    if (!knob->mIsBounded)
    {
        return false;
    }

    std::set<std::string> const allowed(knob->mValues.begin(), knob->mValues.end());
    return std::ranges::all_of(values, [&allowed](auto const& v) { return allowed.contains(v); });
}

bool BuildRouteKnobDatabase::isBounded(std::string const& knobName) const
{
    BuildRouteKnobDef const* knob = getKnob(knobName);
    return knob != nullptr && knob->mIsBounded;
}

std::string BuildRouteKnobDatabase::getDefaultValue(std::string const& knobName) const
{
    BuildRouteKnobDef const* knob = getKnob(knobName);
    return knob != nullptr ? knob->mDefaultValue : std::string();
}

std::vector<std::string> BuildRouteKnobDatabase::parseAllowedValues(std::string const& allowedStr)
{
    // Find the bracket pattern: -knob=[val1|val2|val3]
    uint64_t const bracketStart = allowedStr.find('[');
    uint64_t const bracketEnd = allowedStr.find(']');

    if (bracketStart == std::string::npos || bracketEnd == std::string::npos || bracketEnd <= bracketStart)
    {
        return {}; // No brackets: unbounded or unknown format (e.g., "=int32_t")
    }

    std::string const valuesStr = allowedStr.substr(bracketStart + 1, bracketEnd - bracketStart - 1);

    // Range patterns like "..." indicate unbounded
    if (valuesStr.find("...") != std::string::npos)
    {
        return {};
    }

    return splitPipeDelimited(valuesStr);
}

// ============================================================================
// BuildRouteExprParser Implementation
// ============================================================================

BuildRouteExprParser::BuildRouteExprParser(BuildRouteKnobDatabase const& db)
    : mDb(db)
{
}

std::string const& BuildRouteExprParser::getError() const noexcept
{
    return mError;
}

std::optional<std::vector<BuildRouteParsedExpr>> BuildRouteExprParser::parse(std::string const& input) const
{
    mError.clear();

    if (input.empty())
    {
        mError = "Empty input";
        return std::nullopt;
    }

    std::vector<std::string> const tokens = tokenize(input);
    if (tokens.empty())
    {
        mError = "No expressions found";
        return std::nullopt;
    }

    std::vector<BuildRouteParsedExpr> result;
    result.reserve(tokens.size());

    for (auto const& token : tokens)
    {
        auto expr = parseExpr(token);
        if (!expr)
        {
            return std::nullopt;
        }
        result.push_back(std::move(*expr));
    }

    return result;
}

std::vector<std::string> BuildRouteExprParser::tokenize(std::string const& input) const
{
    // Split input by spaces, but keep bracketed content together
    // E.g., "-opt1=[a|b] -opt2=c" -> ["-opt1=[a|b]", "-opt2=c"]

    std::vector<std::string> tokens;
    std::string current;
    int32_t bracketDepth = 0;

    for (char const c : input)
    {
        if (c == '[')
        {
            ++bracketDepth;
            current += c;
        }
        else if (c == ']')
        {
            --bracketDepth;
            current += c;
        }
        else if (c == ' ' && bracketDepth == 0)
        {
            // Split point - space outside brackets
            if (!current.empty())
            {
                tokens.push_back(std::move(current));
                current.clear();
            }
        }
        else
        {
            current += c;
        }
    }

    // Add final token if any
    if (!current.empty())
    {
        tokens.push_back(std::move(current));
    }

    return tokens;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::optional<BuildRouteParsedExpr> BuildRouteExprParser::parseExpr(std::string const& expr) const
{
    BuildRouteParsedExpr result;

    // Find the '=' separator
    uint64_t const eqPos = expr.find('=');
    if (eqPos == std::string::npos)
    {
        mError = "Invalid expression (no '='): " + expr;
        return std::nullopt;
    }

    // Extract knob name (before '=')
    result.mKnobName = expr.substr(0, eqPos);

    // Trim whitespace from knob name
    uint64_t start = result.mKnobName.find_first_not_of(" \t");
    uint64_t end = result.mKnobName.find_last_not_of(" \t");
    if (start != std::string::npos && end != std::string::npos)
    {
        result.mKnobName = result.mKnobName.substr(start, end - start + 1);
    }

    // Validate knob exists in database
    if (!mDb.hasKnob(result.mKnobName))
    {
        mError = "Unknown knob: " + result.mKnobName;
        return std::nullopt;
    }

    // Extract value part (after '=')
    std::string valueStr = expr.substr(eqPos + 1);

    // Check if this is a bracketed value list or a fixed value
    uint64_t const bracketStart = valueStr.find('[');
    uint64_t const bracketEnd = valueStr.find(']');

    if (bracketStart != std::string::npos && bracketEnd != std::string::npos && bracketEnd > bracketStart)
    {
        // Bracketed value list: -knob=[val1|val2|val3]

        // Validate: nothing should appear before the opening bracket
        std::string const beforeBracket = valueStr.substr(0, bracketStart);
        uint64_t nonSpace = beforeBracket.find_first_not_of(" \t");
        if (nonSpace != std::string::npos)
        {
            mError = "Invalid expression format (unexpected content before '['): " + expr;
            return std::nullopt;
        }

        // Validate: nothing should appear after the closing bracket
        std::string const afterBracket = valueStr.substr(bracketEnd + 1);
        nonSpace = afterBracket.find_first_not_of(" \t");
        if (nonSpace != std::string::npos)
        {
            mError = "Invalid expression format (unexpected content after ']'): " + expr;
            return std::nullopt;
        }

        // Extract and parse values inside brackets
        std::string const valuesInBrackets = valueStr.substr(bracketStart + 1, bracketEnd - bracketStart - 1);

        // Check for unbounded patterns
        if (valuesInBrackets.find("...") != std::string::npos)
        {
            mError = "Unbounded expression not allowed: " + expr;
            return std::nullopt;
        }

        // Split by '|' and trim each value
        result.mValues = splitPipeDelimited(valuesInBrackets);

        if (result.mValues.empty())
        {
            mError = "Empty value list in expression: " + expr;
            return std::nullopt;
        }

        // Check for duplicates
        std::unordered_set<std::string> seenValues;
        for (auto const& val : result.mValues)
        {
            auto [iter, inserted] = seenValues.insert(val);
            if (!inserted)
            {
                mError = "Duplicate value '" + val + "' in expression: " + expr;
                return std::nullopt;
            }
        }

        // Validate values against the knob's allowed values
        if (!mDb.validateValues(result.mKnobName, result.mValues))
        {
            if (!mDb.isBounded(result.mKnobName))
            {
                mError = "Knob has unbounded values (int32_t): " + result.mKnobName;
                return std::nullopt;
            }
            mError = "Invalid value(s) for knob: " + result.mKnobName;
            return std::nullopt;
        }

        result.mIsFixed = false;
    }
    else
    {
        // Fixed value: -knob=value

        // Trim whitespace from value
        start = valueStr.find_first_not_of(" \t");
        end = valueStr.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos)
        {
            valueStr = valueStr.substr(start, end - start + 1);
        }

        // Check for unbounded type keyword
        if (valueStr == "int32_t")
        {
            mError = "Unbounded expression 'int32_t' not allowed: " + expr;
            return std::nullopt;
        }

        result.mValues.push_back(valueStr);
        result.mIsFixed = true;

        // Validate fixed value if knob is bounded
        if (mDb.isBounded(result.mKnobName))
        {
            if (!mDb.validateValues(result.mKnobName, result.mValues))
            {
                mError = "Invalid value for knob: " + result.mKnobName + "=" + valueStr;
                return std::nullopt;
            }
        }
    }

    return result;
}

// ============================================================================
// TuningContext Implementation
// ============================================================================

std::optional<BigInt> TuningContext::count() const
{
    return searchStrategy->count(*this);
}

std::optional<std::string> TuningContext::getBuildRouteAtIndex(BigInt const& index)
{
    return searchStrategy->getBuildRouteAtIndex(*this, index);
}

namespace
{
//! \brief Count the Cartesian product without materializing routes.
//! Each expression contributes its value-list size; fixed arguments contribute one. An empty
//! expression list represents no routes. BigInt preserves products larger than native integers.
BigInt countBuildRoutes(std::vector<BuildRouteParsedExpr> const& exprs)
{
    BigInt total{exprs.empty() ? 0U : 1U};
    for (auto const& expr : exprs)
    {
        total = total * BigInt{expr.mValues.size()};
    }
    return total;
}

//! \brief Preserve expression order when formatting one set of selected values.
//! values must contain one entry per expression. Spaces separate knob=value tokens because each
//! token is a separate compiler option; no other combinations are generated or stored here.
std::string formatBuildRoute(std::vector<BuildRouteParsedExpr> const& exprs, std::vector<std::string> const& values)
{
    std::string route;
    for (size_t i = 0; i < exprs.size(); ++i)
    {
        if (i != 0)
        {
            route += " ";
        }
        route += exprs[i].mKnobName + "=" + values[i];
    }
    return route;
}

//! \brief Use explicit fixed arguments in preference to knob defaults.
//! The returned vector parallels parsedExprs. Variable knobs use their database defaults even if
//! those defaults are absent from their candidate lists; fixed knobs retain the requested value.
std::vector<std::string> baselineValues(TuningContext const& ctx)
{
    auto values = ctx.defaultValues;
    for (size_t i = 0; i < ctx.parsedExprs.size(); ++i)
    {
        if (ctx.parsedExprs[i].mIsFixed)
        {
            values[i] = ctx.parsedExprs[i].mValues.front();
        }
    }
    return values;
}

//! \brief Decode one mixed-radix index, with the rightmost knob changing fastest.
//! For value lists [on,off] and [0,1,2], index 4 yields off/1: 4 % 3 selects 1 in the last list,
//! and (4 / 3) % 2 selects off in the first. Both full and mixed searches reuse this arithmetic.
//! Return nullopt when index is outside the product; working storage holds only one route's values.
std::optional<std::string> getExhaustiveBuildRoute(std::vector<BuildRouteParsedExpr> const& exprs, BigInt const& index)
{
    if (index >= countBuildRoutes(exprs))
    {
        return std::nullopt;
    }
    std::vector<std::string> values(exprs.size());
    BigInt remaining = index;
    for (size_t i = exprs.size(); i > 0; --i)
    {
        auto const& candidates = exprs[i - 1].mValues;
        BigInt const base{candidates.size()};
        values[i - 1] = candidates[(remaining % base).toUint64()];
        remaining = remaining / base;
    }
    return formatBuildRoute(exprs, values);
}

//! \brief Keep the fastest value per knob, breaking equal-time ties deterministically.
//! The caller supplies eligible improvements. Rank by GPU time, expression index, and value text,
//! then retain at most maxKnobs distinct indices. Taking the vector by value lets streaming callers
//! move their bounded working selection into this helper instead of copying a complete history.
std::vector<MixedSearchKnobResult> selectTopPositiveKnobs(
    std::vector<MixedSearchKnobResult> positiveKnobs, uint32_t maxKnobs)
{
    std::sort(positiveKnobs.begin(), positiveKnobs.end(), [](auto const& lhs, auto const& rhs) {
        return std::tie(lhs.gpuTimeMs, lhs.knobIndex, lhs.value) < std::tie(rhs.gpuTimeMs, rhs.knobIndex, rhs.value);
    });
    std::set<int32_t> selectedIndices;
    std::vector<MixedSearchKnobResult> selected;
    for (auto const& knob : positiveKnobs)
    {
        if (selected.size() == maxKnobs)
        {
            break;
        }
        if (selectedIndices.insert(knob.knobIndex).second)
        {
            selected.push_back(knob);
        }
    }
    return selected;
}
} // namespace

std::optional<BigInt> FullSearch::count(TuningContext const& ctx) const
{
    return countBuildRoutes(ctx.parsedExprs);
}

std::optional<std::string> FullSearch::getBuildRouteAtIndex(TuningContext const& ctx, BigInt const& index)
{
    return getExhaustiveBuildRoute(ctx.parsedExprs, index);
}

std::optional<BigInt> FastSearch::count(TuningContext const& ctx) const
{
    BigInt total{ctx.parsedExprs.empty() ? 0U : 1U};
    for (size_t i = 0; i < ctx.parsedExprs.size(); ++i)
    {
        auto const& expr = ctx.parsedExprs[i];
        if (!expr.mIsFixed)
        {
            for (auto const& value : expr.mValues)
            {
                if (value != ctx.defaultValues[i])
                {
                    ++total;
                }
            }
        }
    }
    return total;
}

std::optional<std::string> FastSearch::getBuildRouteAtIndex(TuningContext const& ctx, BigInt const& index)
{
    if (index >= *count(ctx))
    {
        return std::nullopt;
    }
    auto values = baselineValues(ctx);
    if (auto const knob = identifyVariedKnob(ctx, index))
    {
        values[knob->first] = knob->second;
    }
    return formatBuildRoute(ctx.parsedExprs, values);
}

std::vector<MixedSearchKnobResult> readPositiveKnobsFromCache(
    TuningContext const& ctx, std::string const& cacheFile, uint32_t maxKnobs)
{
    std::ifstream file(cacheFile);
    std::string header;
    if (!std::getline(file, header))
    {
        throw std::runtime_error("Cannot read individual measurements from tuning cache: " + cacheFile);
    }
    std::vector<MixedSearchKnobResult> positiveKnobs;
    double baselineGpuTimeMs = std::numeric_limits<double>::infinity();
    BigInt const scanCount = *FastSearch{}.count(ctx);
    for (BigInt i{0}; i < scanCount; ++i)
    {
        auto const row = readCachedIterationResult(file);
        if (!row)
        {
            throw std::runtime_error("Incomplete individual measurements in tuning cache: " + cacheFile);
        }
        if (i.isZero())
        {
            baselineGpuTimeMs = row->succeeded() ? row->gpuTimeMs : std::numeric_limits<double>::infinity();
        }
        else
        {
            collectPositiveKnobFromResult(!row->succeeded(), row->gpuTimeMs, baselineGpuTimeMs, i, ctx, positiveKnobs);
            // The scan can be large too; retain at most maxKnobs candidates after every record.
            positiveKnobs = selectTopPositiveKnobs(std::move(positiveKnobs), maxKnobs);
        }
    }
    return positiveKnobs;
}

MixedSearch::MixedSearch(std::string cacheFile)
    : mCacheFile(std::move(cacheFile))
{
}

std::optional<BigInt> MixedSearch::count(TuningContext const& ctx) const
{
    if (!mSelectedExprs)
    {
        // The total depends on the knob selection loaded by the first combination lookup.
        return std::nullopt;
    }
    return *mFastSearch.count(ctx) + countBuildRoutes(*mSelectedExprs);
}

std::optional<std::string> MixedSearch::getBuildRouteAtIndex(TuningContext const& ctx, BigInt const& index)
{
    BigInt const scanCount = *mFastSearch.count(ctx);
    if (index < scanCount)
    {
        return mFastSearch.getBuildRouteAtIndex(ctx, index);
    }
    if (!mSelectedExprs)
    {
        // Restore the same domains for a direct resume as for a fresh run reaching combinations.
        // Retain value lists only; their Cartesian product may be too large to expand.
        auto const positiveKnobs = readPositiveKnobsFromCache(ctx, mCacheFile, kMAX_POSITIVE_KNOBS);
        std::vector<BuildRouteParsedExpr> exprs;
        if (positiveKnobs.size() > 1)
        {
            exprs = ctx.parsedExprs;
            auto const defaults = baselineValues(ctx);
            for (size_t i = 0; i < exprs.size(); ++i)
            {
                bool const selected = std::any_of(positiveKnobs.begin(), positiveKnobs.end(),
                    [i](auto const& knob) { return static_cast<size_t>(knob.knobIndex) == i; });
                if (!selected)
                {
                    exprs[i].mValues = {defaults[i]};
                    exprs[i].mIsFixed = true;
                }
            }
        }
        mSelectedExprs = std::move(exprs);
    }
    return getExhaustiveBuildRoute(*mSelectedExprs, index - scanCount);
}

MixedFastSearch::MixedFastSearch(std::string cacheFile)
    : mCacheFile(std::move(cacheFile))
{
}

std::optional<BigInt> MixedFastSearch::count(TuningContext const& ctx) const
{
    if (!mPositiveKnobs)
    {
        // The total depends on the knob selection loaded by the first combination lookup.
        return std::nullopt;
    }
    auto const selected = mPositiveKnobs->size();
    return *mFastSearch.count(ctx) + BigInt{(uint64_t{1} << selected) - 1 - selected};
}

std::optional<std::string> MixedFastSearch::getBuildRouteAtIndex(TuningContext const& ctx, BigInt const& index)
{
    BigInt const scanCount = *mFastSearch.count(ctx);
    if (index < scanCount)
    {
        return mFastSearch.getBuildRouteAtIndex(ctx, index);
    }
    if (!mPositiveKnobs)
    {
        // A direct lookup after continuation may be the first call that needs this selection.
        // Load it once from the completed individual trials; later combination timings do not alter it.
        mPositiveKnobs = readPositiveKnobsFromCache(ctx, mCacheFile, kMAX_POSITIVE_KNOBS);
    }
    if (index >= *count(ctx))
    {
        return std::nullopt;
    }
    BigInt remaining = index - scanCount;
    auto const& selected = *mPositiveKnobs;
    // selected is fastest first, so its first entry occupies the highest bit. Descending masks
    // put all knobs first; for ranks C, B, A the multi-knob masks yield ABC, BC, AC, AB.
    for (uint32_t mask = (uint32_t{1} << selected.size()) - 1; mask != 0; --mask)
    {
        if ((mask & (mask - 1)) == 0)
        {
            continue; // Individual values have already been measured.
        }
        if (!remaining.isZero())
        {
            --remaining;
            continue;
        }
        auto values = baselineValues(ctx);
        for (size_t i = 0; i < selected.size(); ++i)
        {
            if ((mask & (uint32_t{1} << (selected.size() - 1 - i))) != 0)
            {
                values[selected[i].knobIndex] = selected[i].value;
            }
        }
        return formatBuildRoute(ctx.parsedExprs, values);
    }
    return std::nullopt;
}

std::unique_ptr<ISearchStrategy> createSearchStrategy(TuningSearchAlgorithm algorithm, std::string const& cacheFile)
{
    switch (algorithm)
    {
    case TuningSearchAlgorithm::kEXHAUSTIVE: return std::make_unique<FullSearch>();
    case TuningSearchAlgorithm::kFAST: return std::make_unique<FastSearch>();
    case TuningSearchAlgorithm::kMIXED: return std::make_unique<MixedSearch>(cacheFile);
    case TuningSearchAlgorithm::kMIXED_FAST: return std::make_unique<MixedFastSearch>(cacheFile);
    }
    throw std::invalid_argument("Unsupported tuning search algorithm");
}

void collectPositiveKnobFromResult(bool crashed, double gpuTimeMs, double baselineGpuTimeMs, BigInt const& index,
    TuningContext const& ctx, std::vector<MixedSearchKnobResult>& positiveKnobs)
{
    if (!crashed && std::isfinite(gpuTimeMs) && std::isfinite(baselineGpuTimeMs) && gpuTimeMs > 0.0
        && gpuTimeMs < baselineGpuTimeMs)
    {
        auto knob = identifyVariedKnob(ctx, index);
        if (knob)
        {
            positiveKnobs.push_back({knob->first, knob->second, gpuTimeMs});
        }
    }
}

std::optional<std::pair<int32_t, std::string>> identifyVariedKnob(TuningContext const& ctx, BigInt const& index)
{
    // In fast/mixed mode, index 0 is baseline. Indices 1..N are one-off variations
    // iterating knobs right-to-left, skipping default values. This reverses that mapping.
    if (index == BigInt(0))
    {
        return std::nullopt; // Baseline, no knob varied
    }

    BigInt remaining = index;
    --remaining;
    for (int64_t i = static_cast<int64_t>(ctx.parsedExprs.size()) - 1; i >= 0; --i)
    {
        if (ctx.parsedExprs[i].mIsFixed)
        {
            continue;
        }
        for (auto const& val : ctx.parsedExprs[i].mValues)
        {
            if (val == ctx.defaultValues[i])
            {
                continue;
            }
            if (remaining == BigInt(0))
            {
                return std::make_pair(static_cast<int32_t>(i), val);
            }
            --remaining;
        }
    }
    return std::nullopt; // Index out of range
}

bool isTuningOnlyArg(char const* arg)
{
    // Tuning-only flags that the parent interprets and that must not appear on the child argv.
    // The child runs a plain single-route trtexec build, so the parent strips these and
    // re-injects the canonical child trio (--setBuildRoute, --saveEngine, --tuningResultFile).
    static constexpr char const* kTUNING_STRIP_PREFIXES[] = {
        "--tuneBuildRoutes",
        "--tuneBuildRouteFile",
        "--tuningSearch",
        "--tuningCacheFile",
        "--tuningTimeOut",
        "--saveAllEngines",
        "--continue",
        "--dryRun",
        // The parent will inject its own --setBuildRoute, --saveEngine, --tuningResultFile.
        "--setBuildRoute",
        "--saveEngine",
        "--tuningResultFile",
    };
    return std::any_of(std::begin(kTUNING_STRIP_PREFIXES), std::end(kTUNING_STRIP_PREFIXES), [arg](char const* prefix) {
        auto const len = std::strlen(prefix);
        return std::strncmp(arg, prefix, len) == 0 && (arg[len] == '\0' || arg[len] == '=');
    });
}

std::vector<char*> buildTuningChildArgv(int32_t argc, char** argv, std::string const& route,
    std::string const& enginePath, std::string const& resultJsonPath, std::vector<std::string>& storage)
{
    storage.clear();
    storage.reserve(argc + 3);
    // Always include argv[0] (the trtexec executable path) verbatim.
    storage.emplace_back(argv[0]);
    for (int32_t i = 1; i < argc; ++i)
    {
        if (argv[i] != nullptr && !isTuningOnlyArg(argv[i]))
        {
            storage.emplace_back(argv[i]);
        }
    }
    storage.emplace_back("--setBuildRoute=" + route);
    storage.emplace_back("--saveEngine=" + enginePath);
    storage.emplace_back("--tuningResultFile=" + resultJsonPath);

    std::vector<char*> out;
    out.reserve(storage.size() + 1);
    for (auto& s : storage)
    {
        out.emplace_back(s.data());
    }
    out.emplace_back(nullptr);
    return out;
}

} // namespace sample
