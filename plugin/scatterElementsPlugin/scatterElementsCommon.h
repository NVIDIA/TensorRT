/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SCATTER_ELEMENTS_COMMON_H
#define SCATTER_ELEMENTS_COMMON_H

#include <map>
#include <memory>
#include <stdint.h>
#include <string>
#include <vector>

#include "common/plugin.h"

enum class ReductionType : int32_t
{
    kSUM,
    kMUL,
    kMEAN,
    kMIN,
    kMAX
};

extern std::unordered_map<std::string, ReductionType> const kREDUCE_STR_TO_ENUM;
extern std::unordered_map<ReductionType, std::string> const kREDUCE_ENUM_TO_STR;

#endif // SCATTER_ELEMENTS_COMMON_H
