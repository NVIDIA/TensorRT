# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# \brief Converts a truthy flag (usually a boolean) to an integer (0 or 1).
# \param flagName The name of the flag to convert.
# \return A CMake variable with the same name as the flag, but suffixed with "_INT" containing 1 if the flag was true and 0 otherwise.
function(flagToInt flagName)
    if(${${flagName}})
        set(${flagName}_INT 1 PARENT_SCOPE)
    else()
        set(${flagName}_INT 0 PARENT_SCOPE)
    endif()
endfunction()
