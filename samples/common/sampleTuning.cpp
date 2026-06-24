/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <cstring>
#include <set>
#include <sstream>
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
    return mKnobs.find(knobName) != mKnobs.end();
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

BigInt TuningContext::count() const
{
    if (parsedExprs.empty())
    {
        return BigInt(0);
    }

    switch (searchAlgorithm)
    {
    case TuningSearchAlgorithm::kEXHAUSTIVE:
    {
        // Product of all value list sizes
        BigInt total(1);
        for (auto const& expr : parsedExprs)
        {
            total = total * BigInt(expr.mValues.size());
        }
        return total;
    }
    case TuningSearchAlgorithm::kFAST:
    case TuningSearchAlgorithm::kMIXED:
    {
        // 1 (baseline) + sum of non-default values per variable knob.
        // For mixed mode, this returns the phase 1 (fast scan) count only.
        // Phase 2 count is determined dynamically after phase 1 completes.
        ASSERT(parsedExprs.size() == defaultValues.size());
        BigInt total(1);
        for (uint64_t i = 0; i < parsedExprs.size(); ++i)
        {
            if (parsedExprs[i].mIsFixed)
            {
                continue;
            }
            for (auto const& val : parsedExprs[i].mValues)
            {
                if (val != defaultValues[i])
                {
                    ++total;
                }
            }
        }
        return total;
    }
    default:
    {
        throw std::invalid_argument("Unsupported tuning search algorithm");
    }
    }
}

std::string TuningContext::getPathAtIndex(BigInt const& index) const
{
    // Helper lambda: build a space-separated knob=value string from per-knob value selections.
    // Spaces are required because the Myelin compiler parses each knob as a separate option.
    // Used by both exhaustive and fast modes.
    auto buildString = [this](auto const& getValueAtPosition) -> std::string {
        std::string result;
        for (uint64_t j = 0; j < parsedExprs.size(); ++j)
        {
            if (j > 0)
            {
                result += " ";
            }
            result += parsedExprs[j].mKnobName + "=" + getValueAtPosition(j);
        }
        return result;
    };

    switch (searchAlgorithm)
    {
    case TuningSearchAlgorithm::kEXHAUSTIVE:
    {
        // Reverse mixed-radix decomposition: decompose index into per-knob value indices.
        // Work from right to left (least significant to most significant).
        BigInt const total = count();
        if (index >= total)
        {
            throw std::out_of_range("Index " + index.toString() + " is out of range [0, " + total.toString() + ")");
        }

        std::vector<uint64_t> valueIndices(parsedExprs.size());
        BigInt current = index;
        for (int64_t i = static_cast<int64_t>(parsedExprs.size()) - 1; i >= 0; --i)
        {
            BigInt const base(parsedExprs[i].mValues.size());
            valueIndices[i] = (current % base).toUint64();
            current = current / base;
        }

        return buildString([&](uint64_t j) -> std::string const& { return parsedExprs[j].mValues[valueIndices[j]]; });
    }
    case TuningSearchAlgorithm::kFAST:
    case TuningSearchAlgorithm::kMIXED:
    {
        // Fast/mixed mode: index 0 is the baseline (all defaults), index 1..N are one-off variations
        // iterating from last knob to first, skipping default values.
        // For mixed mode, this handles phase 1 only. Phase 2 uses a separate TuningContext.
        ASSERT(parsedExprs.size() == defaultValues.size());

        auto baselineValue = [this](uint64_t j) -> std::string const& {
            return parsedExprs[j].mIsFixed ? parsedExprs[j].mValues[0] : defaultValues[j];
        };

        // Index 0: pure baseline
        if (index == BigInt(0))
        {
            return buildString(baselineValue);
        }

        // Index > 0: find which knob is varied and to what value
        auto knob = identifyVariedKnob(*this, index);
        if (!knob)
        {
            throw std::out_of_range("Index " + index.toString() + " is out of range for fast expansion");
        }
        return buildString([&](uint64_t j) -> std::string const& {
            return static_cast<int64_t>(j) == knob->first ? knob->second : baselineValue(j);
        });
    }
    default:
    {
        throw std::invalid_argument("Unsupported tuning search algorithm");
    }
    }
}

// ============================================================================
// Mixed Search Phase 2 Context Builder
// ============================================================================

TuningContext buildMixedPhase2Context(
    TuningContext const& phase1Context, std::vector<MixedSearchKnobResult> const& positiveKnobs)
{
    // Build a set of knob indices that are "positive" (showed improvement in phase 1).
    // For these knobs, we keep their full value lists. All other knobs become fixed to baseline.
    std::set<int32_t> positiveIndices;
    for (auto const& knob : positiveKnobs)
    {
        positiveIndices.insert(knob.knobIndex);
    }

    TuningContext phase2;
    phase2.searchAlgorithm = TuningSearchAlgorithm::kEXHAUSTIVE;
    phase2.tunerVersion = phase1Context.tunerVersion;
    phase2.defaultBuildRoute = phase1Context.defaultBuildRoute;

    for (uint64_t i = 0; i < phase1Context.parsedExprs.size(); ++i)
    {
        BuildRouteParsedExpr expr;
        expr.mKnobName = phase1Context.parsedExprs[i].mKnobName;

        if (positiveIndices.count(static_cast<int32_t>(i)) > 0 && !phase1Context.parsedExprs[i].mIsFixed)
        {
            // Positive knob: keep full value list for exhaustive expansion.
            expr.mValues = phase1Context.parsedExprs[i].mValues;
            expr.mIsFixed = false;
        }
        else
        {
            // Non-positive or fixed knob: lock to baseline default value.
            expr.mValues = {phase1Context.parsedExprs[i].mIsFixed ? phase1Context.parsedExprs[i].mValues[0]
                                                                  : phase1Context.defaultValues[i]};
            expr.mIsFixed = true;
        }

        phase2.parsedExprs.push_back(std::move(expr));
        phase2.defaultValues.push_back(phase1Context.defaultValues[i]);
    }

    phase2.totalCount = phase2.count();
    return phase2;
}

void collectPositiveKnobFromResult(bool crashed, double gpuTimeMs, double baselineGpuTimeMs, BigInt const& index,
    TuningContext const& ctx, std::vector<MixedSearchKnobResult>& positiveKnobs)
{
    if (!crashed && gpuTimeMs > 0.0 && gpuTimeMs < baselineGpuTimeMs)
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
