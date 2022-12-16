/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "common/plugin.h"

namespace nvinfer1
{
namespace plugin
{

void validateRequiredAttributesExist(std::set<std::string> requiredFieldNames, PluginFieldCollection const* fc)
{
    for (int32_t i = 0; i < fc->nbFields; i++)
    {
        requiredFieldNames.erase(fc->fields[i].name);
    }
    if (!requiredFieldNames.empty())
    {
        std::stringstream msg{"PluginFieldCollection missing required fields: {"};
        char const* seperator = "";
        for (auto const& field : requiredFieldNames)
        {
            msg << seperator << field;
            seperator = ", ";
        }
        msg << "}";
        std::string msg_str = msg.str();
        PLUGIN_ERROR(msg_str.c_str());
    }
}

} // namespace plugin
} // namespace nvinfer1
