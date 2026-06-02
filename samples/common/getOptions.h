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

#ifndef TRT_GET_OPTIONS_H
#define TRT_GET_OPTIONS_H

#include <string>
#include <utility>
#include <vector>

namespace nvinfer1::utility
{

//! TRTOption defines a command line option. At least 1 of shortName and longName
//! must be defined.
//! If bool initialization is undefined behavior on your system, valueRequired
//! must also be explicitly defined.
//! helpText is optional.
struct TRTOption
{
    char shortName{};     //!< Option name in short (single hyphen) form (e.g., -a, -b); '\0' for no short name.
    std::string longName; //!< Option name in long (double hyphen) form (e.g., --foo, --bar); empty for no long name.
    bool valueRequired{}; //!< True if a value is needed for an option (e.g., -N 4, --foo bar); false for not required.
    std::string helpText; //!< Text to show when printing out the command usage
};

//! TRTParsedArgs is returned by getOptions after it has parsed a command line
//! argument list (argv).
struct TRTParsedArgs
{
    //! An error message if any errors occurred. Empty if no errors occurred.
    std::string errMsg;

    //! A value for an option.
    struct Value
    {
        //! The number of occurrences of the option (for value-required options, this equals `values.size()`).
        int32_t occurrences{};
        //! The values for the option. For non-value args, will be empty.
        std::vector<std::string> values;
        //! Increment the number of occurrences (for a non-value arg).
        void addOccurrence()
        {
            ++occurrences;
        }
        //! Append \p value and set \p occurrences to the number of values.
        void addOccurrence(std::string value)
        {
            values.push_back(std::move(value));
            occurrences = values.size();
        }
    };
    //! A list of values for each option.
    std::vector<Value> values;
    //! Positional arguments that are passed in without an option (these must not start with a hyphen).
    std::vector<std::string> positionalArgs;
};

//! Parse the input arguments passed to main() and extract options as well as
//! positional arguments.
//!
//! Options are supposed to be passed to main() with a preceding hyphen '-'.
//!
//! If there is a single preceding hyphen, there should be exactly 1 character
//! after the hyphen, which is interpreted as the option.
//!
//! If there are 2 preceding hyphens, the entire argument (without the hyphens)
//! is interpreted as the option.
//!
//! If the option requires a value, the next argument is used as the value.
//!
//! Positional arguments must not start with a hyphen.
//!
//! If an argument requires a value, the next argument is interpreted as the
//! value, even if it is the form of a valid option (i.e. --foo --bar will store
//! "--bar" as a value for option "foo" if "foo" requires a value).
//! We also support --name=value syntax. In this case, 'value' would be used as
//! the value, NOT the next argument.
//!
//! For options:
//!   { { 'a', "", false },
//!     { 'b', "", false },
//!     { 0, "cee", false },
//!     { 'd', "", true },
//!     { 'e', "", true },
//!     { 'f', "foo", true } }
//!
//! ./main hello world -a -a --cee -d 12 -f 34
//! and
//! ./main hello world -a -a --cee -d 12 --foo 34
//!
//! will result in:
//!
//! TRTParsedArgs {
//!      errMsg: "",
//!      values: { { 2, {} },
//!                { 0, {} },
//!                { 1, {} },
//!                { 1, {"12"} },
//!                { 0, {} },
//!                { 1, {"34"} } }
//!      positionalArgs: {"hello", "world"},
//! }
//!
//! Non-POSIX behavior:
//!      - Does not support "-abcde" as a shorthand for "-a -b -c -d -e". Each
//!        option must have its own hyphen prefix.
//!      - Does not support -e12 as a shorthand for "-e 12". Values MUST be
//!        whitespace-separated from the option it is for.
//!
//! @param[in] argc The number of arguments passed to main (including the
//!            file name, which is disregarded)
//! @param[in] argv The arguments passed to main (including the file name,
//!            which is disregarded)
//! @param[in] options List of TRTOptions to parse
//! @return TRTParsedArgs. See TRTParsedArgs documentation for descriptions of
//!         the fields.
[[nodiscard]] TRTParsedArgs getOptions(int argc, char const* const* argv, std::vector<TRTOption> const& options);
} // namespace nvinfer1::utility

#endif // TRT_GET_OPTIONS_H
