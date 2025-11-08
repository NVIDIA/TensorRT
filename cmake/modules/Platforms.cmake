# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Contains constants for the various platform names TRT supports.

set(TRT_PLATFORM_X86
    "x86_64"
    CACHE INTERNAL "Linux")
set(TRT_PLATFORM_AARCH64
    "aarch64"
    CACHE INTERNAL "ARM Linux")
set(TRT_PLATFORM_QNX
    "qnx"
    CACHE INTERNAL "QNX")
set(TRT_PLATFORM_QNX_SAFE
    "qnx-safe"
    CACHE INTERNAL "QNX Safe")
set(TRT_PLATFORM_WIN10
    "win10"
    CACHE INTERNAL "Windows 10")


# Checks if the current build platform matches any of the passed (ARGN) platforms.
#
# \param outVar The output variable name.
# \param argn   The list of platforms to check against.
# \returns      TRUE if TRT_BUILD_PLATFORM matches any of the platforms, FALSE otherwise.
function(checkPlatform outVar)
    if(NOT DEFINED TRT_BUILD_PLATFORM)
        message(FATAL_ERROR "checkPlatform was called before TRT_BUILD_PLATFORM was defined!")
    endif()

    set(isPlatform FALSE)
    foreach(platform IN LISTS ARGN)
        if(${platform} STREQUAL ${TRT_BUILD_PLATFORM})
            set(isPlatform TRUE)
        endif()
    endforeach()

    set(${outVar} ${isPlatform} PARENT_SCOPE)
endfunction()
