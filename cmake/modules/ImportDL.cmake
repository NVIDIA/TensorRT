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

include_guard()

if(NOT TARGET dl)
    # libdl is included in the system library on Windows and QNX.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        find_library(DL_LIB_PATH
            NAMES ${CMAKE_DL_LIBS}
            REQUIRED
        )

        message(STATUS "Creating imported target 'dl' for ${DL_LIB_PATH}")
        add_library(dl SHARED IMPORTED)
        set_target_properties(dl PROPERTIES IMPORTED_LOCATION "${DL_LIB_PATH}")
    else()
        message(STATUS "Creating no-op target 'dl' since libdl is not available on this platform.")
        add_library(dl INTERFACE) # Add a fake dl target so we can still call target_link_libraries without error, even though it's a no-op.
    endif()
endif()
