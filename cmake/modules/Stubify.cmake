# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This file provides functionality to create stub libraries from existing CMake targets
# by calling the existing stubify.sh script.
#
# Stub libraries contain the same exported symbols as the original shared library but with empty function bodies,
# removing all dependencies while maintaining the same API surface for linking purposes.

# Creates a stub library from an existing CMake target by calling stubify.sh.
#
# \param target_name Name of an existing CMake shared library target
#
# Creates a new target called ${target_name}_stub that contains empty implementations
# of all exported symbols from the original target. The stub library is also installed
# to lib/stubs during the install phase.
function(create_stub_lib target_name)
    if(MSVC)
        message(FATAL_ERROR "Creating stub libs is not supported on Windows.")
    endif()

    if(NOT TARGET ${target_name})
        message(FATAL_ERROR "Target ${target_name} does not exist")
    endif()

    get_target_property(target_type ${target_name} TYPE)
    if(NOT target_type STREQUAL "SHARED_LIBRARY")
        message(FATAL_ERROR "Target ${target_name} is not a shared library")
    endif()

    set(stub_target "${target_name}_stub")

    # The stub output should be the same base name of the library with a _stub suffix.
    # We need to read the OUTPUT_NAME of the target, and fallback to the target name otherwise.
    get_target_property(output_name ${target_name} OUTPUT_NAME)
    if(NOT output_name)
        set(output_name ${target_name})
    endif()
    set(stub_output "${CMAKE_CURRENT_BINARY_DIR}/stubs/lib${output_name}.so")

    # Find the stubify.sh script
    find_file(STUBIFY_SCRIPT stubify.sh 
        PATHS ${CMAKE_SOURCE_DIR}/scripts
        NO_DEFAULT_PATH
        REQUIRED
    )

    # Create a custom target that calls stubify.sh
    add_custom_command(
        OUTPUT ${stub_output}
        COMMAND ${CMAKE_COMMAND} -E env
                CC=${CMAKE_C_COMPILER}
                CC_ARGS=${STUBIFY_CC_ARGS}
                --
                ${STUBIFY_SCRIPT} $<TARGET_FILE:${target_name}> ${stub_output}
        DEPENDS $<TARGET_FILE:${target_name}> ${STUBIFY_SCRIPT}
        COMMENT "Creating stub library ${stub_output} for ${target_name}"
        VERBATIM
    )

    # Create an imported library target for the stub
    add_library(${stub_target} SHARED IMPORTED)
    set_target_properties(${stub_target} PROPERTIES
        IMPORTED_LOCATION ${stub_output}
    )

    # Create a custom target to ensure the stub gets built
    add_custom_target(${stub_target}_build
        ALL
        DEPENDS ${stub_output}
    )

    # Make sure the stub builds after the original
    add_dependencies(${stub_target}_build ${target_name})

    # Make the imported target depend on the build target
    add_dependencies(${stub_target} ${stub_target}_build)

    # Install the stub library to lib/stubs
    install(FILES ${stub_output}
        DESTINATION lib/stubs
        COMPONENT external
        OPTIONAL
    )
endfunction()
