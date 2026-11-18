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

## Helper file for management of symbol exports in shared libraries via linker scripts.
## This allows controlling symbol visibility across different platforms without needing platform-specific code.

# Applies an export map to one or more targets (specified by TARGETS).
# The export map file should be provided without extension; the appropriate extension will be added based on the platform (Windows: .def, Unix: .map).
# If CONFIGURE_FIRST is set, the export map will be configured via CMake configure_file (@ONLY). The source file must have the additional extension .in.
#
# \param CONFIGURE_FIRST Optional flag to indicate that the export map file should be configured first.
# \param EXPORT_MAP_FILE The base name of the export map file (without extension). If not provided, the function will look for a file matching the target's output name in the 'exports' directory.
# \param BINARY_DIR Optional path to the binary directory where the configured file will be placed. If not set, the current binary directory is used.
# \param TARGETS The targets to which the export map should be applied. Must not be empty.
function(apply_export_map)
    cmake_parse_arguments(ARG "CONFIGURE_FIRST" "EXPORT_MAP_FILE;BINARY_DIR" "TARGETS" ${ARGN})

    if(NOT ARG_TARGETS)
        message(FATAL_ERROR "apply_export_map: TARGETS argument must not be empty.")
    endif()

    if(NOT ARG_BINARY_DIR)
        set(ARG_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    endif()

    if(MSVC)
        set(export_extension ".def")
    else()
        set(export_extension ".map")
    endif()

   foreach(target IN LISTS ARG_TARGETS)
        if(ARG_EXPORT_MAP_FILE)
            cmake_path(IS_RELATIVE ARG_EXPORT_MAP_FILE is_relative)
            if(is_relative)
                set(export_map_file "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_EXPORT_MAP_FILE}")
            else()
                set(export_map_file "${ARG_EXPORT_MAP_FILE}")
            endif()
        else()
            # Otherwise, default to searching for a file that matches the target's output name.
            get_target_property(output_name ${target} OUTPUT_NAME)
            if(NOT output_name)
                set(output_name ${target})
            endif()
            set(export_map_file "${CMAKE_CURRENT_SOURCE_DIR}/exports/${output_name}")
        endif()

        if(ARG_CONFIGURE_FIRST)
            if(NOT EXISTS "${export_map_file}${export_extension}.in")
                message(FATAL_ERROR "apply_export_map: Expected to find export map template file '${export_map_file}${export_extension}.in' for configuration.")
            endif()
            get_filename_component(export_file_name ${export_map_file} NAME_WE)
            configure_file(
                "${export_map_file}${export_extension}.in"
                "${ARG_BINARY_DIR}/${export_file_name}${export_extension}"
                @ONLY
            )
            set(export_map_file "${ARG_BINARY_DIR}/${export_file_name}")
        endif()

        if(NOT EXISTS "${export_map_file}${export_extension}")
            message(FATAL_ERROR "apply_export_map: Expected to find export map file '${export_map_file}${export_extension}'.")
        endif()

        if(MSVC)
            # On Windows, use a .def file to control exported symbols
            target_sources(${target} PRIVATE "${export_map_file}${export_extension}")
        else()
            # On Unix-like systems, use a version script for symbol visibility
            target_link_options(${target} PRIVATE "LINKER:--version-script=${export_map_file}${export_extension}")
            # We also need to update the link dependencies. We don't need to do this on Windows since the .def file is a source file.
            get_target_property(link_deps ${target} LINK_DEPENDS)
            if(NOT link_deps)
                set(link_deps "")
            endif()
            list(APPEND link_deps "${export_map_file}${export_extension}")
            set_target_properties(${target} PROPERTIES LINK_DEPENDS "${link_deps}")
        endif()
    endforeach()
endfunction()

# Reports the version of lld this build links with, or an empty string when it does not use lld.
#
# \param outVar Name of the variable to receive the version.
function(get_lld_version outVar)
    set(version "")

    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_USING_LINKER_${CMAKE_LINKER_TYPE}} -Wl,--version
        OUTPUT_VARIABLE versionOutput
        ERROR_QUIET
    )
    if(versionOutput MATCHES "LLD ([0-9]+(\\.[0-9]+)*)")
        set(version "${CMAKE_MATCH_1}")
    endif()

    set(${outVar} "${version}" PARENT_SCOPE)
endfunction()

# Links one or more targets (specified by TARGETS) with BFD, for targets that rely on --retain-symbols-file.
#
# lld accepts --retain-symbols-file, but doesn't filter .symtab (https://github.com/llvm/llvm-project/pull/209063).
# So using this flag to shrink the symbol table must link with BFD unless lld is new enough. Remove this once every
# container we use for building ships with lld 23 or newer.
#
# \param TARGETS The targets to link with BFD. Must not be empty.
function(force_bfd_for_retained_symbols)
    cmake_parse_arguments(ARG "" "" "TARGETS" ${ARGN})

    if(NOT ARG_TARGETS)
        message(FATAL_ERROR "force_bfd_for_retained_symbols: TARGETS argument must not be empty.")
    endif()

    # --retain-symbols-file is not supported by MSVC
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        return()
    endif()

    # Only force BFD when the build is actually using lld. An empty version means the linker is
    # already BFD.
    get_lld_version(lldVersion)
    if(NOT lldVersion)
        return()
    endif()
    if(lldVersion VERSION_GREATER_EQUAL 23)
        return()
    endif()

    set_target_properties(${ARG_TARGETS} PROPERTIES LINKER_TYPE BFD)
endfunction()
