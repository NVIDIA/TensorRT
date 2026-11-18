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

class ISearchStrategy;

//! \struct TuningContext
//! \brief Context for build route tuning when --tuneBuildRoutes is specified.
//!
//! Uses lazy expansion: build route strings are computed on-demand from
//! the parsed expressions and an index, without storing all strings in memory.
//! This supports arbitrarily large expansion spaces (full mode with BigInt count).
//! The context owns its search strategy. Keep the parsed expressions and defaults unchanged
//! while using that strategy, since mixed searches retain a selection derived from those inputs.
struct TuningContext
{
    //! Parsed expressions from --tuneBuildRoutes.
    std::vector<BuildRouteParsedExpr> parsedExprs;

    //! Default values for each knob (from knob database, needed for fast mode).
    std::vector<std::string> defaultValues;

    //! Zero-based index of the next unfinished iteration. A fresh run starts at zero; continuation
    //! initializes it from the completed cache-record count. Only the execution loop advances it.
    BigInt globalIndex{0};

    //! Enumeration policy selected by buildTuningContext(); must be set before calling either method.
    std::unique_ptr<ISearchStrategy> searchStrategy;

    //! Tuner version string from the knob database JSON (e.g. "2.17.83").
    std::string tunerVersion{"unknown"};

    //! Default build route string (all knobs at their default values).
    std::string defaultBuildRoute;

    //! \brief Ask the strategy for its total route count without running or advancing the search.
    //! \return A known count, including zero for an empty search, or nullopt when a mixed strategy
    //! has not yet loaded its knob selection. Cache records alone do not make the count known;
    //! getBuildRouteAtIndex() loads the selection when combinations are first requested.
    std::optional<BigInt> count() const;

    //! \brief Compute one route without advancing globalIndex; return nullopt at the end.
    //! \param index Zero-based global route index; callers may request it directly without visiting
    //! earlier indices. For example, full search of -A=[on|off] -B=[on|off] returns
    //! "-A=off -B=on" at index 2 and nullopt at index 4.
    //! \return The requested concrete route, or nullopt when index is at or beyond the total count.
    //! \throws std::runtime_error if measurements required by a mixed search cannot be read.
    std::optional<std::string> getBuildRouteAtIndex(BigInt const& index);
};

//! \brief Enumeration interface for custom searches, owned by TuningContext.
//! Implement getBuildRouteAtIndex() and count(), then register the named strategy in
//! createSearchStrategy() and --tuningSearch option parsing. Execution, accuracy validation,
//! and cache writes remain in the existing tuning loop and child process.
class ISearchStrategy
{
public:
    //! \brief Allow TuningContext to destroy derived strategies through this interface.
    virtual ~ISearchStrategy() = default;

    //! \brief Compute one route by index without enumerating or storing all earlier routes.
    //! \param ctx Parsed expressions and defaults, unchanged for the lifetime of this search.
    //! \param index Zero-based global index, including both individual and combined trials.
    //! \return A concrete route, or nullopt when index is at or beyond the end of the search.
    //! A strategy may read cached measurements and retain a bounded selection on its first
    //! combination lookup. It must not advance ctx.globalIndex, launch children, or append records.
    //! Report unreadable or incomplete required measurements as exceptions, not as end-of-search.
    virtual std::optional<std::string> getBuildRouteAtIndex(TuningContext const& ctx, BigInt const& index) = 0;

    //! \brief Return the total count without loading measurements or changing strategy state.
    //! \return The full count when known, or nullopt while it depends on measurements not yet
    //! loaded by getBuildRouteAtIndex(). Zero is a known empty search, not an unknown count.
    virtual std::optional<BigInt> count(TuningContext const& ctx) const = 0;
};

//! \brief Exhaustive enumeration using mixed-radix arithmetic without expanding the search space.
//! Preserve the expression/value-list order, with the rightmost knob changing fastest.
//! For -A=[on|off] -B=[on|off], indices 0..3 select on/on, on/off, off/on, off/off.
class FullSearch : public ISearchStrategy
{
public:
    //! \brief Decode index directly into one value per expression; return nullopt past the end.
    //! Fixed expressions contribute their single explicit value. No cache is required.
    std::optional<std::string> getBuildRouteAtIndex(TuningContext const& ctx, BigInt const& index) override;
    //! \brief Return the product of the value-list sizes, or zero when there are no expressions.
    //! The count is always known and uses BigInt rather than a native integer product.
    std::optional<BigInt> count(TuningContext const& ctx) const override;
};

//! \brief Enumerate the baseline, then individual non-default values from right to left.
//! Index 0 uses defaults for variable knobs and explicit values for fixed knobs. Subsequent
//! indices vary one knob at a time, preserving each knob's value-list order and skipping defaults.
//! Defaults remain part of the baseline even when the expression's candidate list omits them.
class FastSearch : public ISearchStrategy
{
public:
    //! \brief Return the baseline or one individual variation; return nullopt past the end.
    //! For -A=[on|off] -B=[on|off] with defaults on/on, indices 0..2 produce
    //! on/on, on/off, off/on. No cache is required and fixed arguments never vary.
    std::optional<std::string> getBuildRouteAtIndex(TuningContext const& ctx, BigInt const& index) override;
    //! \brief Return one baseline plus the number of non-default values across variable knobs.
    //! Fixed knobs add no trials. With no expressions the count is zero; it is always known.
    std::optional<BigInt> count(TuningContext const& ctx) const override;
};

//! \brief One successful individual variation that improved on the measured baseline.
//! Ranking keeps the fastest value per distinct knob for the subsequent combination search.
struct MixedSearchKnobResult
{
    int32_t knobIndex; //!< Index into TuningContext::parsedExprs.
    std::string value; //!< The non-default value that was tested.
    double gpuTimeMs;  //!< Mean GPU time; lower is better.
};

//! \brief Follow the fast scan with exhaustive enumeration of the selected knobs' original value lists.
//! Select the fastest distinct improving knobs, but retain all original candidate values for
//! each selected knob. Unselected knobs use their baseline values; explicit fixed values remain.
//! With fewer than two selected knobs, there are no combination trials. Otherwise the count
//! adds the product of the selected domain sizes, including repeated baseline/individual routes
//! when their values appear in those domains.
class MixedSearch : public ISearchStrategy
{
public:
    //! Maximum distinct knobs whose complete value lists participate in the exhaustive search.
    static constexpr uint32_t kMAX_POSITIVE_KNOBS = 10;

    //! \brief Store the cache path without reading it or starting a search.
    //! \param cacheFile JSON-lines measurements produced by this run or restored by --continue.
    explicit MixedSearch(std::string cacheFile);
    //! \brief Return an individual route or lazily calculate an exhaustive combination.
    //! The first combination lookup reads the complete fast-scan prefix and retains the selected
    //! domains. A resumed run can request any later index directly. Missing, malformed, or
    //! unreadable required measurements throw std::runtime_error; the search end returns nullopt.
    std::optional<std::string> getBuildRouteAtIndex(TuningContext const& ctx, BigInt const& index) override;
    //! \brief Return nullopt until selection is loaded, then the fast count plus combination count.
    std::optional<BigInt> count(TuningContext const& ctx) const override;

private:
    std::string mCacheFile; //!< Source of the initial individual measurements.
    FastSearch mFastSearch; //!< Shared initial enumeration and count.
    //! Selected domains, not expanded routes; nullopt means not loaded, empty means no combinations.
    std::optional<std::vector<BuildRouteParsedExpr>> mSelectedExprs;
};

//! \brief Follow the fast scan with descending subsets of the best individual knob values.
//! Keep one best value per selected knob. Enumerate subsets containing at least two knobs,
//! placing the strongest knobs in the highest mask bits. If C is faster than B, and B faster
//! than A, the combination order is ABC, BC, AC, AB. Omitted knobs keep baseline values.
//! For K selected knobs there are 2^K - 1 - K additional routes, computed individually on demand.
class MixedFastSearch : public ISearchStrategy
{
public:
    //! Maximum distinct knobs to combine; seven limits the additional search to 120 trials.
    static constexpr uint32_t kMAX_POSITIVE_KNOBS = 7;

    //! \brief Store the cache path without reading it or starting a search.
    //! \param cacheFile JSON-lines measurements produced by this run or restored by --continue.
    explicit MixedFastSearch(std::string cacheFile);
    //! \brief Return an individual route or calculate one descending subset of the selected values.
    //! The first combination lookup loads only the bounded selection from the fast-scan prefix,
    //! even when continuation starts midway through combinations. Later lookups reuse it.
    //! Missing, malformed, or unreadable required measurements throw std::runtime_error;
    //! the search end returns nullopt. Baseline and single-knob subsets are not repeated.
    std::optional<std::string> getBuildRouteAtIndex(TuningContext const& ctx, BigInt const& index) override;
    //! \brief Return nullopt until selection is loaded, then the fast count plus 2^K - 1 - K.
    std::optional<BigInt> count(TuningContext const& ctx) const override;

private:
    std::string mCacheFile; //!< Source of the initial individual measurements.
    FastSearch mFastSearch; //!< Shared initial enumeration and count.
    //! Bounded best-first selection; nullopt means not loaded, empty means no improving knobs.
    std::optional<std::vector<MixedSearchKnobResult>> mPositiveKnobs;
};

//! \brief Construct the search selected by --tuningSearch.
//! \param algorithm Parsed algorithm selector; extend this factory when adding a named strategy.
//! \param cacheFile Measurement cache used by mixed strategies; ignored by full and fast searches.
//! \return An owned strategy ready for route lookup; construction does not read the cache.
//! \throws std::invalid_argument if algorithm is unsupported.
std::unique_ptr<ISearchStrategy> createSearchStrategy(TuningSearchAlgorithm algorithm, std::string const& cacheFile);

//! \brief Stream the fast-scan prefix, keeping at most maxKnobs distinct improving knobs.
//!
//! The first record supplies the baseline. Other records are eligible only if they succeeded
//! with finite positive GPU time strictly below that valid baseline. Rank by GPU time, expression
//! index, then value text, retaining the fastest value per distinct knob before applying the limit.
//! Map records to individual variations by their position in the prefix; do not regenerate or
//! compare saved routes/iteration numbers. Ignore all later combination records.
//!
//! Caches and initial scans can be very large. Read one record at a time and trim the selection
//! after each record, so memory depends on the largest record and maxKnobs, not cache length.
//! \param ctx Expressions and defaults defining the initial fast scan.
//! \param cacheFile JSON-lines cache with a header followed by the complete initial scan.
//! \param maxKnobs Maximum number of distinct knobs returned, supplied by the strategy's limit.
//! \return Best-first selection, possibly empty when no valid improvement was measured.
//! \throws std::runtime_error if the required prefix is missing, incomplete, malformed, or unreadable.
[[nodiscard]] std::vector<MixedSearchKnobResult> readPositiveKnobsFromCache(
    TuningContext const& ctx, std::string const& cacheFile, uint32_t maxKnobs);

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
//! it to the positiveKnobs vector. Fresh and resumed searches use cached individual measurements.
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
