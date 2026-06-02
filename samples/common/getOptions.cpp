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

#include "getOptions.h"
#include "logger.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstring>
#include <set>

namespace nvinfer1::utility
{

namespace
{

using namespace std::string_view_literals;
using sample::gLogWarning;

//! Matching for TRTOptions is defined as follows:
//!
//! If A and B both have longName set, A matches B if and only if A.longName ==
//! B.longName and (A.shortName == B.shortName if both have short name set).
//!
//! If A only has shortName set and B only has longName set, then A does not
//! match B. It is assumed that when 2 TRTOptions are compared, one of them is
//! the definition of a TRTOption in the input to getOptions. As such, if the
//! definition only has shortName set, it will never be equal to a TRTOption
//! that does not have shortName set (and same for longName).
//!
//! If A and B both have shortName set but B does not have longName set, A
//! matches B if and only if A.shortName == B.shortName.
//!
//! If A has neither long or short name set, A matches B if and only if B has
//! neither long or short name set.
[[nodiscard]] bool matches(TRTOption const& a, TRTOption const& b)
{
    if (!a.longName.empty() && !b.longName.empty())
    {
        if (a.shortName != '\0' && b.shortName != '\0')
        {
            return (a.longName == b.longName) && (a.shortName == b.shortName);
        }
        return a.longName == b.longName;
    }

    // If only one of them is not set, this will return false anyway.
    return a.shortName == b.shortName;
}

//! getTRTOptionIndex returns the index of a TRTOption in a vector of
//! TRTOptions, -1 if not found.
[[nodiscard]] int32_t getTRTOptionIndex(std::vector<TRTOption> const& options, TRTOption const& opt)
{
    auto it = std::find_if(
        options.begin(), options.end(), [&opt](TRTOption const& option) { return matches(opt, option); });
    return it != options.end() ? static_cast<int32_t>(std::distance(options.begin(), it)) : -1;
}

//! validateTRTOption will return a string containing an error message if options
//! contain non-numeric characters, or if there are duplicate option names found.
//! Otherwise, returns the empty string.
[[nodiscard]] std::string validateTRTOption(
    std::set<char> const& seenShortNames, std::set<std::string> const& seenLongNames, TRTOption const& opt)
{
    if (opt.shortName != '\0')
    {
        if (!std::isalnum(opt.shortName))
        {
            return "Short name '" + std::to_string(opt.shortName) + "' is non-alphanumeric";
        }

        if (seenShortNames.count(opt.shortName) != 0)
        {
            return "Short name '" + std::to_string(opt.shortName) + "' is a duplicate";
        }
    }

    if (!opt.longName.empty())
    {
        for (char const& c : opt.longName)
        {
            if (!std::isalnum(c) && c != '-' && c != '_')
            {
                return "Long name '" + opt.longName + "' contains characters that are not '-', '_', or alphanumeric";
            }
        }

        if (seenLongNames.count(opt.longName) != 0)
        {
            return "Long name '" + opt.longName + "' is a duplicate";
        }
    }
    return "";
}

//! validateTRTOptions will return a string containing an error message if any
//! options contain non-numeric characters, or if there are duplicate option
//! names found. Otherwise, returns the empty string.
[[nodiscard]] std::string validateTRTOptions(std::vector<TRTOption> const& options)
{
    std::set<char> seenShortNames;
    std::set<std::string> seenLongNames;
    for (size_t i = 0; i < options.size(); ++i)
    {
        std::string const errMsg = validateTRTOption(seenShortNames, seenLongNames, options[i]);
        if (!errMsg.empty())
        {
            return "Error '" + errMsg + "' at TRTOption " + std::to_string(i);
        }

        seenShortNames.insert(options[i].shortName);
        seenLongNames.insert(options[i].longName);
    }
    return "";
}

//! Structure to hold a parsed option and its inline value (if any)
struct ParsedOption
{
    TRTOption opt;
    std::string inlineValue;
};

//! Parse an option string (starting with '-' or '--') into a TRTOption and optional inline value.
//! \param[in] argStr The option string to parse.
//! \param[out] result The parsed option and inline value.
//! \return error message if parsing fails, empty string otherwise.
[[nodiscard]] std::string parseOptionString(std::string_view const argStr, ParsedOption& result)
{
    // C++23: Return a `std::expected<ParsedOption, std::string>` instead.
    if (argStr.size() < 2)
    {
        return "Option string is too short";
    }
    if (argStr[1] != '-')
    {
        // Short option: must only have 1 char after the hyphen
        if (argStr.size() > 2)
        {
            return "Short arg contains more than 1 character";
        }
        result = ParsedOption{TRTOption{argStr[1]}};
        return {};
    }
    else
    {
        // Long option: extract name and check for --foo=bar syntax
        auto longName = argStr.substr(2);
        size_t const eqIndex = longName.find('=');

        auto inlineValue = eqIndex != std::string_view::npos ? longName.substr(eqIndex + 1) : ""sv;

        // Note: If `eqIndex == std::string_view::npos`, then `longName.substr(0, eqIndex)` is the entire string_view.
        result = ParsedOption{TRTOption{{}, std::string{longName.substr(0, eqIndex)}}, std::string{inlineValue}};
        return {};
    }
}

//! Handle an option that requires a value. Returns error message if value cannot be obtained.
//! Updates currentArgIdx if a value is consumed from the next argument.
[[nodiscard]] std::string handleRequiredValue(TRTParsedArgs& parsedArgs, int32_t idx, std::string inlineValue,
    std::string_view const argStr, int32_t& currentArgIdx, int32_t argc, char const* const* argv)
{
    // If we have an inline value (from --foo=bar), use it
    if (!inlineValue.empty())
    {
        parsedArgs.values[idx].addOccurrence(std::move(inlineValue));
        return {};
    }

    // Otherwise, consume the next argument as the value
    if (currentArgIdx + 1 >= argc)
    {
        return "Last argument requires value, but none given";
    }

    std::string_view const nextArg(argv[currentArgIdx + 1]);
    if (!nextArg.empty() && nextArg[0] == '-')
    {
        gLogWarning << "Warning: Using '" << nextArg << "' as a value for '" << argStr
                    << "', Should this be its own flag?" << std::endl;
    }

    parsedArgs.values[idx].addOccurrence(std::string{nextArg});
    ++currentArgIdx; // Next argument consumed
    return {};
}

//! parseArgs parses an argument list and returns a TRTParsedArgs with the
//! fields set accordingly. Assumes that options is validated.
//! ErrMsg will be set if:
//!     - an argument is null
//!     - an argument is empty
//!     - an argument does not have option (i.e. "-" and "--")
//!     - a short argument has more than 1 character
//!     - the last argument in the list requires a value
[[nodiscard]] TRTParsedArgs parseArgs(
    int32_t const argc, char const* const* const argv, std::vector<TRTOption> const& options)
{
    TRTParsedArgs parsedArgs;
    parsedArgs.values.resize(options.size());

    for (int32_t i = 1; i < argc; ++i) // index of current command-line argument
    {
        if (argv[i] == nullptr)
        {
            return TRTParsedArgs{"Null argument at index " + std::to_string(i)};
        }

        std::string_view const argStr(argv[i]);
        if (argStr.empty())
        {
            return TRTParsedArgs{"Empty argument at index " + std::to_string(i)};
        }

        // No starting hyphen means it is a positional argument
        if (argStr[0] != '-')
        {
            parsedArgs.positionalArgs.push_back(std::string{argStr});
            continue;
        }
        if (argStr == "-"sv || argStr == "--"sv)
        {
            return TRTParsedArgs{"Argument does not specify an option at index " + std::to_string(i)};
        }

        // Parse the option string
        ParsedOption parsed;
        if (std::string const parseErr = parseOptionString(argStr, parsed); !parseErr.empty())
        {
            return TRTParsedArgs{parseErr + " at index " + std::to_string(i)};
        }

        // Find the option in the registered options list
        int32_t const idx = getTRTOptionIndex(options, parsed.opt);
        if (idx < 0)
        {
            continue;
        }

        // Handle value-required options vs. flag options
        if (options[idx].valueRequired)
        {
            if (std::string valueErr = handleRequiredValue(parsedArgs, idx, parsed.inlineValue, argStr, i, argc, argv);
                !valueErr.empty())
            {
                return TRTParsedArgs{std::move(valueErr)};
            }
        }
        else
        {
            parsedArgs.values[idx].addOccurrence();
        }
    }
    return parsedArgs;
}

} // namespace

TRTParsedArgs getOptions(int32_t argc, char const* const* argv, std::vector<TRTOption> const& options)
{
    if (std::string errMsg = validateTRTOptions(options); !errMsg.empty())
    {
        return TRTParsedArgs{std::move(errMsg)};
    }

    return parseArgs(argc, argv, options);
}
} // namespace nvinfer1::utility
