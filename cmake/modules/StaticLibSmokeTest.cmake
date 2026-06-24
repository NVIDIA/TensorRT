# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Utility module for smoke testing a static library.
# This allows us to verify that the static library can be linked without any unexpected dependencies.

include_guard()

define_property(TARGET
    PROPERTY STATIC_LIBRARY_SMOKE_TEST_SOURCE_CODE
    BRIEF_DOCS "Source code to use when executing the static library smoke test. Defaults to 'int main() { return 0; }'"
)

# Creates a new target that links the given static library and builds a simple executable.
# The executable will be added to the ALL target if the static library target is also in the ALL target.
function(smoke_test_static_lib static_lib_target)
    get_target_property(static_lib_type ${static_lib_target} TYPE)
    if(NOT static_lib_type STREQUAL "STATIC_LIBRARY")
        message(FATAL_ERROR "Target ${static_lib_target} is not a static library.")
    endif()

    set(smoke_test_target "${static_lib_target}_smoke_test")
    get_target_property(smoke_test_src ${static_lib_target} STATIC_LIBRARY_SMOKE_TEST_SOURCE_CODE)
    # Handle NOTFOUND sentinel and empty string; otherwise treat property as source content.
    if(NOT smoke_test_src)
        set(smoke_test_src "int main() { return 0; }\n")
    endif()

    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${smoke_test_target}.cpp" "${smoke_test_src}")

    add_executable(${smoke_test_target} "${CMAKE_CURRENT_BINARY_DIR}/${smoke_test_target}.cpp")

    target_link_libraries(${smoke_test_target} PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,${static_lib_target}>)
    # cuDLA is not available as a static library, so we link it dynamically
    if(${TRT_BUILD_ENABLE_DLA} AND TARGET CUDA::cudla)
        target_link_libraries(${smoke_test_target} PRIVATE CUDA::cudla)
    endif()

    # If the static library target is in the ALL target, add the smoke test target to the ALL target as well.
    get_target_property(static_lib_excluded_from_all ${static_lib_target} EXCLUDE_FROM_ALL)
    if(static_lib_excluded_from_all)
        set_target_properties(${smoke_test_target} PROPERTIES
            EXCLUDE_FROM_ALL ON
        )
    endif()
endfunction()
