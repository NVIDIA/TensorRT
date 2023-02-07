/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace nvinfer1
{
namespace utility
{

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
bool matches(const TRTOption& a, const TRTOption& b)
{
    if (!a.longName.empty() && !b.longName.empty())
    {
        if (a.shortName && b.shortName)
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
int getTRTOptionIndex(const std::vector<TRTOption>& options, const TRTOption& opt)
{
    for (size_t i = 0; i < options.size(); ++i)
    {
        if (matches(opt, options[i]))
        {
            return i;
        }
    }
    return -1;
}

//! validateTRTOption will return a string containing an error message if options
//! contain non-numeric characters, or if there are duplicate option names found.
//! Otherwise, returns the empty string.
std::string validateTRTOption(
    const std::set<char>& seenShortNames, const std::set<std::string>& seenLongNames, const TRTOption& opt)
{
    if (opt.shortName != 0)
    {
        if (!std::isalnum(opt.shortName))
        {
            return "Short name '" + std::to_string(opt.shortName) + "' is non-alphanumeric";
        }

        if (seenShortNames.find(opt.shortName) != seenShortNames.end())
        {
            return "Short name '" + std::to_string(opt.shortName) + "' is a duplicate";
        }
    }

    if (!opt.longName.empty())
    {
        for (const char& c : opt.longName)
        {
            if (!std::isalnum(c) && c != '-' && c != '_')
            {
                return "Long name '" + opt.longName + "' contains characters that are not '-', '_', or alphanumeric";
            }
        }

        if (seenLongNames.find(opt.longName) != seenLongNames.end())
        {
            return "Long name '" + opt.longName + "' is a duplicate";
        }
    }
    return "";
}

//! validateTRTOptions will return a string containing an error message if any
//! options contain non-numeric characters, or if there are duplicate option
//! names found. Otherwise, returns the empty string.
std::string validateTRTOptions(const std::vector<TRTOption>& options)
{
    std::set<char> seenShortNames;
    std::set<std::string> seenLongNames;
    for (size_t i = 0; i < options.size(); ++i)
    {
        const std::string errMsg = validateTRTOption(seenShortNames, seenLongNames, options[i]);
        if (!errMsg.empty())
        {
            return "Error '" + errMsg + "' at TRTOption " + std::to_string(i);
        }

        seenShortNames.insert(options[i].shortName);
        seenLongNames.insert(options[i].longName);
    }
    return "";
}

//! parseArgs parses an argument list and returns a TRTParsedArgs with the
//! fields set accordingly. Assumes that options is validated.
//! ErrMsg will be set if:
//!     - an argument is null
//!     - an argument is empty
//!     - an argument does not have option (i.e. "-" and "--")
//!     - a short argument has more than 1 character
//!     - the last argument in the list requires a value
TRTParsedArgs parseArgs(int argc, const char* const* argv, const std::vector<TRTOption>& options)
{
    TRTParsedArgs parsedArgs;
    parsedArgs.values.resize(options.size());

    for (int i = 1; i < argc; ++i) // index of current command-line argument
    {
        if (argv[i] == nullptr)
        {
            return TRTParsedArgs{"Null argument at index " + std::to_string(i)};
        }

        const std::string argStr(argv[i]);
        if (argStr.empty())
        {
            return TRTParsedArgs{"Empty argument at index " + std::to_string(i)};
        }

        // No starting hyphen means it is a positional argument
        if (argStr[0] != '-')
        {
            parsedArgs.positionalArgs.push_back(argStr);
            continue;
        }

        if (argStr == "-" || argStr == "--")
        {
            return TRTParsedArgs{"Argument does not specify an option at index " + std::to_string(i)};
        }

        // If only 1 hyphen, char after is the flag.
        TRTOption opt{' ', "", false, ""};
        std::string value;
        if (argStr[1] != '-')
        {
            // Must only have 1 char after the hyphen
            if (argStr.size() > 2)
            {
                return TRTParsedArgs{"Short arg contains more than 1 character at index " + std::to_string(i)};
            }
            opt.shortName = argStr[1];
        }
        else
        {
            opt.longName = argStr.substr(2);

            // We need to support --foo=bar syntax, so look for '='
            const size_t eqIndex = opt.longName.find('=');
            if (eqIndex < opt.longName.size())
            {
                value = opt.longName.substr(eqIndex + 1);
                opt.longName = opt.longName.substr(0, eqIndex);
            }
        }

        const int idx = getTRTOptionIndex(options, opt);
        if (idx < 0)
        {
            continue;
        }

        if (options[idx].valueRequired)
        {
            if (!value.empty())
            {
                parsedArgs.values[idx].second.push_back(value);
                parsedArgs.values[idx].first = parsedArgs.values[idx].second.size();
                continue;
            }

            if (i + 1 >= argc)
            {
                return TRTParsedArgs{"Last argument requires value, but none given"};
            }

            const std::string nextArg(argv[i + 1]);
            if (nextArg.size() >= 1 && nextArg[0] == '-')
            {
                sample::gLogWarning << "Warning: Using '" << nextArg << "' as a value for '" << argStr
                                    << "', Should this be its own flag?" << std::endl;
            }

            parsedArgs.values[idx].second.push_back(nextArg);
            i += 1; // Next argument already consumed

            parsedArgs.values[idx].first = parsedArgs.values[idx].second.size();
        }
        else
        {
            parsedArgs.values[idx].first += 1;
        }
    }
    return parsedArgs;
}

TRTParsedArgs getOptions(int argc, const char* const* argv, const std::vector<TRTOption>& options)
{
    const std::string errMsg = validateTRTOptions(options);
    if (!errMsg.empty())
    {
        return TRTParsedArgs{errMsg};
    }
    return parseArgs(argc, argv, options);
}
} // namespace utility
} // namespace nvinfer1
