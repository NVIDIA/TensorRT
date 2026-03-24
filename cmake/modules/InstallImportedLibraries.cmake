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

# This module provides functionality to install imported shared libraries
# while properly resolving and installing complete symlink chains.
#
# This is particularly useful for system libraries that use versioned symlinks
# (e.g., libfoo.so -> libfoo.so.1 -> libfoo.so.1.2.3).

# Installs an imported library target with all its symlinks.
#
# \param targets     One or more CMake targets to install. Targets must be a shared library (or an unknown library pointing to a shared library).
# \param destination Destination directory relative to CMAKE_INSTALL_PREFIX (default: lib)
# \param component   Installation component name (optional)
#
# This function attempts to resolve symlinks by globbing for libname.so* in the directory where the target is located.
function(installImportedLibraries)
    set(options)
    set(oneValueArgs DESTINATION COMPONENT)
    set(multiValueArgs TARGETS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Set defaults
    if(NOT ARG_DESTINATION)
        # On Windows, shared libraries (dlls) go to the bin dir.
        # Since this function only handles shared libraries, we can simply default to bin/ here.
        if(WIN32)
            set(ARG_DESTINATION ${CMAKE_INSTALL_BINDIR})
        else()
            set(ARG_DESTINATION ${CMAKE_INSTALL_LIBDIR})
        endif()
    endif()

    if(ARG_COMPONENT)
        set(INSTALL_COMPONENT_INPUT COMPONENT ${ARG_COMPONENT}) # Don't quote this, or the component will fail to eval in the install() command.
    else()
        set(INSTALL_COMPONENT_INPUT "")
    endif()

    if(NOT ARG_TARGETS)
        message(FATAL_ERROR "installImportedLibraries requires at least one target to install.")
    endif()

    foreach(target_name IN LISTS ARG_TARGETS)
        if(NOT TARGET ${target_name})
            message(FATAL_ERROR "Target ${target_name} does not exist")
        endif()

        get_target_property(target_type ${target_name} TYPE)
        if(NOT target_type MATCHES "SHARED_LIBRARY|UNKNOWN_LIBRARY")
            message(FATAL_ERROR "Target ${target_name} is not a shared library (type: ${target_type})")
        endif()

        # Install all library files and symlinks directly
        install(CODE "
            # Resolve where the target is located at install time so we can grab any adjacent symlinks.
            # Trying to do so earlier will require knowing the build mode at configure time (which we don't).
            get_filename_component(TARGET_DIR \$<TARGET_FILE:${target_name}> DIRECTORY)
            get_filename_component(TARGET_BASE_NAME \$<TARGET_FILE:${target_name}> NAME_WE)

            set(TARGET_LIBS_EXPR \"\${TARGET_DIR}/\${TARGET_BASE_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}*\")

            file(GLOB TARGET_LIBS \${TARGET_LIBS_EXPR})
            file(INSTALL \${TARGET_LIBS}
                DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${ARG_DESTINATION}\"
                FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
            )
            "
            ${INSTALL_COMPONENT_INPUT}
        )
    endforeach()
endfunction()
