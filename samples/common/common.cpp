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

#include "common.h"

namespace samplesCommon
{

using namespace std::string_view_literals;

std::optional<std::string_view> matchFlag(std::string_view arg, std::string_view flag)
{
    auto const start = arg.find_first_not_of(' ');
    if (start == std::string_view::npos)
    {
        return std::nullopt;
    }
    arg.remove_prefix(start);
    if (startsWith(arg, flag))
    {
        return arg.substr(flag.size());
    }
    return std::nullopt;
}

int32_t parseDLA(int32_t argc, char** argv)
{
    for (int32_t i = 1; i < argc; i++)
    {
        if (auto v = matchFlag(argv[i], "--useDLACore="sv))
        {
            return std::stoi(std::string{v.value()});
        }
    }
    return -1;
}

} // namespace samplesCommon
