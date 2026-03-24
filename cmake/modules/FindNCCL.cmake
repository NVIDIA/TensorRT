#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#

# FindNCCL.cmake
#
# This module finds and imports the NCCL (NVIDIA Collective Communication Library) from system paths.
#
# Installation:
# -------------
# On Ubuntu/Debian:
#   apt-get install libnccl2 libnccl-dev
# On RHEL/CentOS:
#   yum install libnccl libnccl-devel
# From source:
#   See https://github.com/NVIDIA/nccl
#
# Input Variables (optional):
# ---------------------------
#   NCCL_ROOT         - Root directory where NCCL is installed (e.g., /usr, /usr/local)
#   NCCL_INCLUDE_DIR  - Directory containing nccl.h
#   NCCL_LIBRARY      - Full path to libnccl.so
#
# Provided Targets:
# -----------------
#   NCCL::nccl        - Imported NCCL library target
#
# Provided Variables:
# -------------------
#   NCCL_FOUND        - True if NCCL was found
#   NCCL_INCLUDE_DIRS - Include directories for NCCL
#   NCCL_LIBRARIES    - NCCL libraries to link against
#   NCCL_VERSION      - Version of NCCL found
#
# Usage:
# ------
#   find_package(NCCL REQUIRED)
#   target_link_libraries(your_target PRIVATE NCCL::nccl)
#
# For custom installations:
#   cmake -DNCCL_ROOT=/path/to/nccl ..
# or
#   set(NCCL_ROOT "/path/to/nccl")
#   find_package(NCCL REQUIRED)

# Search for NCCL header
find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS
        ${NCCL_ROOT}
        ${NCCL_ROOT}/include
        $ENV{NCCL_ROOT}
        $ENV{NCCL_ROOT}/include
    PATHS
        /usr/include
        /usr/local/include
        /usr/include/x86_64-linux-gnu
        /usr/local/cuda/include
    DOC "Path to NCCL include directory"
)

# Search for NCCL library
find_library(NCCL_LIBRARY
    NAMES nccl libnccl
    HINTS
        ${NCCL_ROOT}
        ${NCCL_ROOT}/lib
        ${NCCL_ROOT}/lib64
        $ENV{NCCL_ROOT}
        $ENV{NCCL_ROOT}/lib
        $ENV{NCCL_ROOT}/lib64
    PATHS
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
    DOC "Path to NCCL library"
)

# Extract version from header if found
if(NCCL_INCLUDE_DIR AND EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
    file(READ "${NCCL_INCLUDE_DIR}/nccl.h" NCCL_HEADER_CONTENTS)
    
    string(REGEX MATCH "#define NCCL_MAJOR[ \t]+([0-9]+)" _ "${NCCL_HEADER_CONTENTS}")
    set(NCCL_VERSION_MAJOR "${CMAKE_MATCH_1}")
    
    string(REGEX MATCH "#define NCCL_MINOR[ \t]+([0-9]+)" _ "${NCCL_HEADER_CONTENTS}")
    set(NCCL_VERSION_MINOR "${CMAKE_MATCH_1}")
    
    string(REGEX MATCH "#define NCCL_PATCH[ \t]+([0-9]+)" _ "${NCCL_HEADER_CONTENTS}")
    set(NCCL_VERSION_PATCH "${CMAKE_MATCH_1}")
    
    if(NCCL_VERSION_MAJOR AND NCCL_VERSION_MINOR AND NCCL_VERSION_PATCH)
        set(NCCL_VERSION "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}.${NCCL_VERSION_PATCH}")
    endif()
endif()

# Standard FindPackage handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
    REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR
    VERSION_VAR NCCL_VERSION
)

if(NCCL_FOUND)
    set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
    set(NCCL_LIBRARIES ${NCCL_LIBRARY})
    
    # Create imported target if it doesn't exist
    if(NOT TARGET NCCL::nccl)
        add_library(NCCL::nccl SHARED IMPORTED)
        set_target_properties(NCCL::nccl PROPERTIES
            IMPORTED_LOCATION "${NCCL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
        )
        
        # Get library directory for installation purposes
        get_filename_component(NCCL_LIB_DIR "${NCCL_LIBRARY}" DIRECTORY)
        message(STATUS "Found NCCL ${NCCL_VERSION} at ${NCCL_LIB_DIR}")
    endif()
endif()

mark_as_advanced(
    NCCL_INCLUDE_DIR
    NCCL_LIBRARY
)
