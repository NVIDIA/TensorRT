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
include(GNUInstallDirs)

# Install one or more targets including PDB files on Windows/MSVC
# Usage:
#   installLibraries(
#     TARGETS target1 [target2 ...]
#     [COMPONENT component]       # Optional component name for packaging
#     [CONFIGURATIONS config1 [config2 ...]]  # Optional configurations to install
#   )
function(installLibraries)
    cmake_parse_arguments(
        ARG                       # Prefix for parsed args
        "OPTIONAL;RUNTIME_ONLY"   # Options (flags)
        "COMPONENT"               # Single value args
        "TARGETS;CONFIGURATIONS"  # Multi-value args
        ${ARGN}
    )

    # Validate required arguments
    if(NOT ARG_TARGETS)
        message(FATAL_ERROR "installLibrary() requires TARGETS argument")
    endif()

    # Prepare optional arguments for regular install command
    if(ARG_COMPONENT)
        set(component_arg COMPONENT ${ARG_COMPONENT})
    endif()
    
    if(ARG_CONFIGURATIONS)
        set(config_arg CONFIGURATIONS ${ARG_CONFIGURATIONS})
    endif()

    if(ARG_OPTIONAL)
        set(optional_arg OPTIONAL)
    endif()

    # When RUNTIME_ONLY is passed, we only want to install .dll files.
    # Instead of also installing the import library (.lib) files.
    # This is only relevant on Windows since Linux doesn't have this distinction.
    if(ARG_RUNTIME_ONLY AND WIN32)
        set(runtime_only_arg
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()

    # Install the libraries
    install(
        TARGETS ${ARG_TARGETS}
        ${optional_arg}
        ${component_arg}
        ${config_arg}
        ${runtime_only_arg}
    )

    # Install PDB files for MSVC builds
    if(MSVC)
        foreach(target ${ARG_TARGETS})
            # Get target type (SHARED_LIBRARY, STATIC_LIBRARY, EXECUTABLE)
            get_target_property(target_type ${target} TYPE)

            # For shared libraries and executables, PDBs are placed alongside the binaries
            if(target_type STREQUAL "SHARED_LIBRARY" OR target_type STREQUAL "EXECUTABLE")
                # Use generator expression to get the PDB file path
                install(
                    FILES "$<TARGET_PDB_FILE:${target}>"
                    DESTINATION ${CMAKE_INSTALL_BINDIR}
                    ${component_arg}
                    CONFIGURATIONS Debug RelWithDebInfo
                    OPTIONAL
                )
            endif()
        endforeach()
    endif()
endfunction()
