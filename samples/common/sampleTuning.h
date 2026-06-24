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

#ifndef TRT_SAMPLE_TUNING_H
#define TRT_SAMPLE_TUNING_H

#include "bigInt.h"
#include "logger.h"
#include "sampleOptions.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace sample
{

//! \brief Split a pipe-delimited string into trimmed tokens.
//!
//! Given a string like "val1|val2|val3", returns {"val1", "val2", "val3"}.
//! Each token is trimmed of leading/trailing whitespace. Empty tokens
//! (after trimming) are skipped.
//!
//! \param[in] str The pipe-delimited string to split.
//! \return Vector of trimmed token strings.
std::vector<std::string> splitPipeDelimited(std::string const& str);

// ============================================================================
// Build Route Expression Parser and Expander
// ============================================================================
//
// These classes implement parsing and expansion of build route expressions
// for the --tuneBuildRoutes feature in trtexec. The expression syntax supports:
//   - Variable values: "-knob=[val1|val2|val3]"
//   - Fixed values: "-knob=fixed_value"
//
// Two expansion modes are supported:
//   - Full (exhaustive): All combinations enumerated (exponential in number of knobs)
//   - Fast (one-off): Baseline (all defaults) + one-knob-different variations (linear)
//
// Example:
//   Expression: "-opt1=[on|off] -opt2=[0|1|2]"
//   Full expansion (6 combinations):
//     [0]: -opt1=on -opt2=0
//     [1]: -opt1=on -opt2=1
//     [2]: -opt1=on -opt2=2
//     [3]: -opt1=off -opt2=0
//     [4]: -opt1=off -opt2=1
//     [5]: -opt1=off -opt2=2
//   Fast expansion (with defaults opt1=off, opt2=2):
//     [0]: -opt1=off -opt2=2  (baseline)
//     [1]: -opt1=off -opt2=0  (opt2 changed)
//     [2]: -opt1=off -opt2=1  (opt2 changed)
//     [3]: -opt1=on -opt2=2   (opt1 changed)
// ============================================================================

//! \struct BuildRouteKnobDef
//! \brief Represents a single knob/option definition from the JSON configuration.
struct BuildRouteKnobDef
{
    std::string mOption;              //!< Option name, e.g., "-kgen:tiling"
    std::string mAllowedValues;       //!< Allowed values string, e.g., "-kgen:tiling=[0|1|2]"
    std::string mDefaultValue;        //!< Default value
    std::string mHelp;                //!< Help text
    std::vector<std::string> mValues; //!< Parsed allowed values (empty if unbounded)
    bool mIsBounded{false};           //!< True if values form a closed set
};

//! \struct BuildRouteParsedExpr
//! \brief Represents a parsed expression from user input.
struct BuildRouteParsedExpr
{
    std::string mKnobName;            //!< Knob name, e.g., "-kgen:tiling"
    std::vector<std::string> mValues; //!< Values to expand, e.g., ["0", "1", "2"]
    bool mIsFixed{false};             //!< True if this is a fixed value (no expansion)
};

//! \class BuildRouteKnobDatabase
//! \brief Database of knob definitions loaded from getAllBuildRoutes() JSON output.
//!
//! This class parses the JSON output from IBuilderConfig::getAllBuildRoutes()
//! and provides lookup and validation functions for knob definitions.
class BuildRouteKnobDatabase
{
public:
    //! \brief Default constructor.
    BuildRouteKnobDatabase() = default;

    //! \brief Destructor.
    ~BuildRouteKnobDatabase() = default;

    // Non-copyable, movable
    BuildRouteKnobDatabase(BuildRouteKnobDatabase const&) = delete;
    BuildRouteKnobDatabase& operator=(BuildRouteKnobDatabase const&) = delete;
    BuildRouteKnobDatabase(BuildRouteKnobDatabase&&) = default;
    BuildRouteKnobDatabase& operator=(BuildRouteKnobDatabase&&) = default;

    //! \brief Load knob definitions from a JSON string.
    //! \param[in] jsonStr JSON string from getAllBuildRoutes().
    //! \return True if loading succeeded, false otherwise.
    bool loadFromJsonString(std::string const& jsonStr);

    //! \brief Check if a knob exists in the database.
    //! \param[in] knobName The knob name to check.
    //! \return True if the knob exists.
    bool hasKnob(std::string const& knobName) const;

    //! \brief Get the knob definition for a given knob name.
    //! \param[in] knobName The knob name to look up.
    //! \return Pointer to the KnobDef, or nullptr if not found.
    BuildRouteKnobDef const* getKnob(std::string const& knobName) const;

    //! \brief Validate that all values are allowed for a knob.
    //! \param[in] knobName The knob name.
    //! \param[in] values The values to validate.
    //! \return True if all values are valid.
    bool validateValues(std::string const& knobName, std::vector<std::string> const& values) const;

    //! \brief Check if a knob has bounded (finite) values.
    //! \param[in] knobName The knob name.
    //! \return True if the knob has bounded values.
    bool isBounded(std::string const& knobName) const;

    //! \brief Get the default value for a knob.
    //! \param[in] knobName The knob name.
    //! \return The default value string, or empty string if not found.
    std::string getDefaultValue(std::string const& knobName) const;

    //! \brief Get the tuner version string extracted from the JSON.
    //! \return Tuner version (e.g. "2.17.83"), or "unknown" if not available.
    std::string const& getTunerVersion() const
    {
        return mTunerVersion;
    }

    //! \brief Build the default build route string from all knob defaults.
    //! Returns a space-separated string of "knob=default_value" pairs,
    //! preserving the insertion order from the original JSON.
    //! Example: "-conv_use_long_w=on -reshape_ppg=on -transpose_ppg=on ..."
    std::string buildDefaultPath() const;

private:
    //! \brief Parse allowed values from an allowed_values string.
    //! \param[in] allowedStr The allowed_values string from JSON.
    //! \return Vector of parsed values (empty if unbounded).
    static std::vector<std::string> parseAllowedValues(std::string const& allowedStr);

    std::unordered_map<std::string, BuildRouteKnobDef> mKnobs; //!< Map of knob names to definitions.
    std::vector<std::string> mKnobOrder;                       //!< Insertion order of knob names.
    std::string mTunerVersion{"unknown"};                      //!< Tuner version from JSON.
};

//! \class BuildRouteExprParser
//! \brief Parser for build route expression strings.
//!
//! Parses expressions like "-opt1=[a|b] -opt2=fixed -opt3=[0|1|2]"
//! into a list of BuildRouteParsedExpr structures for expansion.
class BuildRouteExprParser
{
public:
    //! \brief Construct a parser with a reference to a knob database.
    //! \param[in] db Reference to the BuildRouteKnobDatabase for validation.
    explicit BuildRouteExprParser(BuildRouteKnobDatabase const& db);

    //! \brief Destructor.
    ~BuildRouteExprParser() = default;

    // Non-copyable, non-movable (holds reference)
    BuildRouteExprParser(BuildRouteExprParser const&) = delete;
    BuildRouteExprParser& operator=(BuildRouteExprParser const&) = delete;
    BuildRouteExprParser(BuildRouteExprParser&&) = delete;
    BuildRouteExprParser& operator=(BuildRouteExprParser&&) = delete;

    //! \brief Parse an input string into a list of expressions.
    //! \param[in] input The input string containing expressions.
    //! \return Optional vector of BuildRouteParsedExpr, or nullopt on error.
    std::optional<std::vector<BuildRouteParsedExpr>> parse(std::string const& input) const;

    //! \brief Get the last error message.
    //! \return The error message string.
    std::string const& getError() const noexcept;

private:
    //! \brief Parse a single expression.
    //! \param[in] expr The expression string.
    //! \return Optional BuildRouteParsedExpr, or nullopt on error.
    std::optional<BuildRouteParsedExpr> parseExpr(std::string const& expr) const;

    //! \brief Tokenize input into individual expressions.
    //! \param[in] input The input string.
    //! \return Vector of token strings.
    std::vector<std::string> tokenize(std::string const& input) const;

    BuildRouteKnobDatabase const& mDb; //!< Reference to the knob database.
    mutable std::string mError;        //!< Last error message.
};

//! \struct TuningContext
//! \brief Context for build route tuning when --tuneBuildRoutes is specified.
//!
//! Uses lazy expansion: build route strings are computed on-demand from
//! the parsed expressions and an index, without storing all strings in memory.
//! This supports arbitrarily large expansion spaces (full mode with BigInt count).
struct TuningContext
{
    //! Parsed expressions from --tuneBuildRoutes.
    std::vector<BuildRouteParsedExpr> parsedExprs;

    //! Default values for each knob (from knob database, needed for fast mode).
    std::vector<std::string> defaultValues;

    //! Total number of configurations.
    BigInt totalCount;

    //! Search algorithm (determines expansion strategy).
    TuningSearchAlgorithm searchAlgorithm{TuningSearchAlgorithm::kFAST};

    //! Tuner version string from the knob database JSON (e.g. "2.17.83").
    std::string tunerVersion{"unknown"};

    //! Default build route string (all knobs at their default values).
    std::string defaultBuildRoute;

    //! \brief Compute total configuration count based on searchAlgorithm.
    //! - Exhaustive: product of all value list sizes.
    //! - Fast: 1 (baseline) + sum of non-default values per variable knob.
    BigInt count() const;

    //! \brief Get the build route string for a given index (lazy).
    //! Throws std::out_of_range if index is out of range.
    //! - Exhaustive: reverse mixed-radix decomposition.
    //! - Fast: baseline at index 0, one-off variations at index 1..N.
    std::string getPathAtIndex(BigInt const& index) const;
};

// ============================================================================
// Mixed Search Algorithm Support
// ============================================================================
//
// Mixed search runs in two phases:
//   Phase 1 (fast scan): baseline + one-off variations (same as --tuningSearch=fast).
//     After each one-off iteration, if it passed accuracy, didn't crash, and was
//     faster than the baseline, the knob is recorded as a "positive" knob.
//   Phase 2 (full combinatorial): if >1 positive knobs found, run 2^n combinations
//     of the top-N positive knobs. Other knobs stay at baseline defaults.
//
// Total iterations: fast_count + (2^n if n > 1, else 0).

//! Maximum number of positive knobs to carry into phase 2 of mixed search.
constexpr int32_t kMixedSearchMaxPositiveKnobs = 10;

//! \struct MixedSearchKnobResult
//! \brief Records a single knob variation that improved performance in phase 1.
//! Used to select which knobs enter phase 2 of mixed search.
struct MixedSearchKnobResult
{
    int32_t knobIndex; //!< Index into TuningContext::parsedExprs
    std::string value; //!< The non-default value that was tested
    double gpuTimeMs;  //!< GPU time achieved (lower is better)
};

//! \brief Build a TuningContext for phase 2 of mixed search.
//!
//! Constructs a new exhaustive-mode TuningContext where only the positive knobs
//! are variable (with their full value lists from phase 1) and all other knobs
//! are fixed to their baseline default values.
//!
//! \param phase1Context The original TuningContext from phase 1 (fast mode).
//! \param positiveKnobs The knobs that showed improvement in phase 1.
//! \return A new TuningContext configured for exhaustive search over positive knobs.
TuningContext buildMixedPhase2Context(
    TuningContext const& phase1Context, std::vector<MixedSearchKnobResult> const& positiveKnobs);

//! \brief Identify which knob was varied at a given fast-mode iteration index.
//!
//! In fast mode, index 0 is the baseline (all defaults). Indices 1..N are one-off
//! variations, iterating knobs right-to-left and skipping default values.
//! This function reverses that mapping: given a one-off index (must be > 0),
//! returns the (knobIndex, value) pair, or nullopt if the index is out of range.
//!
//! Example: with 2 binary knobs (defaults "on", "on"), values [on|off] each:
//!   index 1 → knob 1 changed to "off"  → returns {1, "off"}
//!   index 2 → knob 0 changed to "off"  → returns {0, "off"}
//!
//! \param ctx   The TuningContext (must be fast or mixed mode).
//! \param index The one-off iteration index (must be > 0).
//! \return (knobIndex, value) pair, or nullopt if index is invalid.
std::optional<std::pair<int32_t, std::string>> identifyVariedKnob(TuningContext const& ctx, BigInt const& index);

//! \brief Collect a positive knob from an iteration result (used in mixed search).
//!
//! If the iteration did not crash, achieved positive GPU time, and was faster than the
//! baseline, identifies which knob was varied (using identifyVariedKnob) and appends
//! it to the positiveKnobs vector. Called from both the live tuning loop (phase 1) and
//! from the --continue cache reconstruction path.
//!
//! Example: if iteration 3 changed knob 1 to "off" and achieved 1.2ms vs baseline 1.5ms,
//! this appends {knobIndex=1, value="off", gpuTimeMs=1.2} to positiveKnobs.
//!
//! \param[in] crashed          Whether the iteration crashed.
//! \param[in] gpuTimeMs        GPU time achieved by this iteration.
//! \param[in] baselineGpuTimeMs Baseline GPU time (index 0).
//! \param[in] index            The iteration index (must be > 0 for one-off variations).
//! \param[in] ctx              The TuningContext (fast/mixed mode).
//! \param[in,out] positiveKnobs Vector to append the positive knob result to.
void collectPositiveKnobFromResult(bool crashed, double gpuTimeMs, double baselineGpuTimeMs, BigInt const& index,
    TuningContext const& ctx, std::vector<MixedSearchKnobResult>& positiveKnobs);

// Exit codes for child process to distinguish failure modes (used when --tuneBuildRoutes expands to multiple configs)
constexpr int32_t kChildExitSuccess = 0;
constexpr int32_t kChildExitFailure = 1;            // runOnceBuildAndInfer returned failure
constexpr int32_t kChildExitStdException = 100;     // Caught std::exception
constexpr int32_t kChildExitUnknownException = 101; // Caught unknown exception

//! \struct TuningIterationResult
//! \brief Captures results from a single tuning iteration for the tuning cache file.
struct TuningIterationResult
{
    double gpuTimeMs{0.0};                                      //!< Mean GPU compute time in milliseconds
    bool accuracyFailed{false};                                 //!< True if accuracy exceeded the threshold
    std::unordered_map<std::string, double> accuracyLossValues; //!< Per-tensor accuracy values
};

//! \brief True if `arg` is a tuning-only parent flag that should be stripped from
//! the child argv. Matches both bare flag and flag=value forms. Tuning-only means
//! the parent loop interprets it; the child runs a plain single-route build.
[[nodiscard]] bool isTuningOnlyArg(char const* arg);

//! \brief Build a child argv for one tuning iteration: copies argv with tuning-only
//! flags removed and appends `--setBuildRoute=<route>`, `--saveEngine=<enginePath>`,
//! `--tuningResultFile=<resultJsonPath>`. String storage is owned by `storage` so
//! the returned `char*` pointers stay valid until the caller's execvp() completes.
//! The returned vector is nullptr-terminated, ready for execvp().
[[nodiscard]] std::vector<char*> buildTuningChildArgv(int32_t argc, char** argv, std::string const& route,
    std::string const& enginePath, std::string const& resultJsonPath, std::vector<std::string>& storage);

} // namespace sample

#endif // TRT_SAMPLE_TUNING_H
