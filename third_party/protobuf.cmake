#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#

# This module handles the acquisition and configuration of all dependencies
# for the ONNX parser. It manages:
#  - Fetching and configuring Protobuf and its Abseil dependency
#  - Setting up cross-compilation support when needed
#  - Configuring build options for both libraries
#  - Setting up namespace control to avoid conflicts
#  - Setting up the onnx_proto dependency with the newly-added protobuf target(s).

include_guard(GLOBAL)

include(FetchContent)
include(ExternalProject)

# We redefine this function to do nothing because it fails to execute on QNX.
# The flags that would be invoked by it through Protobuf and Abseil are setup manually.
function(target_compile_features)
endfunction()

# Dependencies of this project (absl / onnx) do not follow this rule.
if(${CMAKE_LINK_LIBRARIES_ONLY_TARGETS})
    set(CMAKE_LINK_LIBRARIES_ONLY_TARGETS OFF)
    set(RESTORE_CMAKE_LINK_LIBRARIES_ONLY_TARGETS ON)
endif()

# Manually include Abseil, a dependency of Protobuf, instead of relying on submodule resolution.
set(ABSEIL_ID "absl")
set(ABSEIL_REPO
    "https://github.com/abseil/abseil-cpp.git"
    CACHE STRING "The https git repository that ONNX2TRT will pull Abseil C++ from")
set(ABSEIL_TAG "20250512.1")
set(ABSEIL_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party/${ABSEIL_ID}")

# Include Protobuf for usage by both ONNX and the ONNX Parser
set(PROTOBUF_VERSION
    "33.4"
    CACHE STRING "The version of Protobuf used when compiling ONNX2TRT. Must be a valid Protobuf git tag version.")
message(STATUS "Protobuf version set to ${PROTOBUF_VERSION}")

# Protobuf location data
set(PROTOBUF_ID "protobuf")
set(PROTOBUF_REPO
    "https://github.com/protocolbuffers/protobuf.git"
    CACHE STRING "The https git repository that ONNX2TRT will pull Protobuf from")
set(PROTOBUF_TAG "v${PROTOBUF_VERSION}")
set(PROTOBUF_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party/${PROTOBUF_ID}")

# Protobuf internal build options
set(protobuf_BUILD_TESTS OFF CACHE INTERNAL "")
set(protobuf_BUILD_LIBPROTOC ON CACHE INTERNAL "")
set(protobuf_BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
# LibUPB is mandatory in newer Protobuf versions but was previously optional.
# https://github.com/protocolbuffers/protobuf/issues/13739
set(protobuf_BUILD_LIBUPB ON CACHE INTERNAL "")
set(protobuf_ALLOW_CCACHE ON CACHE INTERNAL "")
set(protobuf_MSVC_STATIC_RUNTIME ON CACHE INTERNAL "")
set(protobuf_WITH_ZLIB OFF CACHE INTERNAL "")
set(protobuf_INSTALL OFF CACHE INTERNAL "")

# Abseil internal build options
set(ABSL_USE_SYSTEM_INCLUDES ON CACHE INTERNAL "")
set(ABSL_MSVC_STATIC_RUNTIME ON CACHE INTERNAL "")
set(ABSL_PROPAGATE_CXX_STD ON CACHE INTERNAL "")
set(ABSL_ENABLE_INSTALL OFF CACHE INTERNAL "") # Installing Protobuf requires telling Abseil to install itself.

# UTF8 Range internal build options
set(utf8_range_ENABLE_INSTALL OFF CACHE INTERNAL "")

# When doing a cross build, we need an extra host (x86) copy of protoc to run over the .proto files with.
if(("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Linux"))
    # Artifact URL for the protoc compiler used by cross builds (x86 Linux -> Target).
    set(PROTOBUF_HOST_ARTIFACT_URL
        "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protoc-${PROTOBUF_VERSION}-linux-x86_64.zip"
    )
    set(PROTOBUF_ARTIFACT_MD5 "7ebf585e6eaa6578851ebd049fb9e15e")
endif()

if(CMAKE_CROSSCOMPILING AND NOT TARGET protobuf::protoc)
    message(STATUS "Setting up an x86_64 host copy of Protobuf to run protoc during cross compilation.")

    set(PROTOBUF_HOST_ID "${PROTOBUF_ID}_x86_64")
    set(PROTOBUF_HOST_DL_DIR "${TensorRT_BINARY_DIR}/${PROTOBUF_HOST_ID}")

    ExternalProject_Add(
        ${PROTOBUF_HOST_ID}
        URL ${PROTOBUF_HOST_ARTIFACT_URL}
        URL_MD5 ${PROTOBUF_ARTIFACT_MD5}
        DOWNLOAD_DIR ${PROTOBUF_HOST_DL_DIR}
        SOURCE_DIR ${PROTOBUF_HOST_DL_DIR} # The zip gets extracted into the source dir
        CONFIGURE_COMMAND "" # We do not need to do anything since we download the pre-built binary.
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        TEST_COMMAND ""
        BYPRODUCTS
            "${PROTOBUF_HOST_DL_DIR}/bin/protoc${CMAKE_EXECUTABLE_SUFFIX}"
    )

    set(ONNX_CUSTOM_PROTOC_EXECUTABLE protobuf::protoc)
    add_executable(protobuf::protoc IMPORTED GLOBAL)
    add_dependencies(protobuf::protoc ${PROTOBUF_HOST_ID})
    set_target_properties(protobuf::protoc PROPERTIES IMPORTED_LOCATION "${PROTOBUF_HOST_DL_DIR}/bin/protoc${CMAKE_EXECUTABLE_SUFFIX}")

    # Disable building protoc for the target platform (this only applies to the FetchContent_MakeAvailable below)
    set(protobuf_BUILD_PROTOC_BINARIES
        OFF
        CACHE INTERNAL "")
endif()

# Hide all warnings for third-party code.
if (MSVC)
else()
    add_compile_options(-w)
endif()

if(NOT TARGET libprotobuf)
    # Fetch and include Abseil
    FetchContent_Declare(
        ${ABSEIL_ID}
        PREFIX ${ABSEIL_DIR}
        GIT_REPOSITORY ${ABSEIL_REPO}
        GIT_TAG ${ABSEIL_TAG}
        GIT_PROGRESS true
        GIT_SHALLOW true
        SOURCE_DIR          ${TensorRT_BINARY_DIR}/${ABSEIL_ID}/${ABSEIL_TAG}
        EXCLUDE_FROM_ALL
        UPDATE_DISCONNECTED ${TRT_FETCH_CONTENT_UPDATES_DISCONNECTED}
    )

    FetchContent_MakeAvailable(${ABSEIL_ID})

    # Fetch and include protobuf for the target arch.
    FetchContent_Declare(
        ${PROTOBUF_ID}
        PREFIX ${PROTOBUF_DIR}
        GIT_REPOSITORY ${PROTOBUF_REPO}
        GIT_TAG ${PROTOBUF_TAG}
        GIT_PROGRESS true
        GIT_SHALLOW true
        GIT_SUBMODULES "" # Skip submodule resolution since we directly include Abseil.
        SOURCE_DIR          ${TensorRT_BINARY_DIR}/${PROTOBUF_ID}/${PROTOBUF_TAG}
        EXCLUDE_FROM_ALL
        UPDATE_DISCONNECTED ${TRT_FETCH_CONTENT_UPDATES_DISCONNECTED}
    )

    FetchContent_MakeAvailable(${PROTOBUF_ID})
endif()

# Mark protobuf targets as SYSTEM so their headers don't trigger compiler warnings.
foreach(_protobuf_tgt IN ITEMS libprotobuf-lite libprotobuf libprotoc libupb)
    if(TARGET ${_protobuf_tgt})
        set_target_properties(${_protobuf_tgt} PROPERTIES SYSTEM TRUE)
    endif()
endforeach()

if(${ONNX2TRT_USE_PROTOBUF_LITE})
    # Protobuf-lite needs libutf8_validity, including at compile time, but currently exposes a PRIVATE link against it.
    # It should no longer be needed at compile time after https://github.com/protocolbuffers/protobuf/issues/24765 is resolved (v34.0+).
    target_link_libraries(libprotobuf-lite PUBLIC utf8_validity)
endif()

# Abseil ignores any previously set MSVC Runtime lib flags, so we need to override them.
if (MSVC)
    # CMake provides no way to actually list all the known targets, so we have to recursively traverse the subdirectories to find the abseil targets.
    function(get_all_targets var)
        set(targets)
        get_all_targets_recursive(targets ${CMAKE_CURRENT_SOURCE_DIR})
        set(${var} ${targets} PARENT_SCOPE)
    endfunction()

    macro(get_all_targets_recursive targets dir)
        get_property(subdirectories DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
        foreach(subdir ${subdirectories})
            get_all_targets_recursive(${targets} ${subdir})
        endforeach()

        get_property(current_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
        list(APPEND ${targets} ${current_targets})
    endmacro()

    get_all_targets(all_targets)

    # Go over every target we just added and set the MSVC_RUNTIME_LIBRARY to the toolchain-defined one.
    foreach(target IN LISTS all_targets)
        set_target_properties(${target} PROPERTIES
            MSVC_RUNTIME_LIBRARY ${CMAKE_MSVC_RUNTIME_LIBRARY}
        )
    endforeach()
endif()

set(ONNX_NAMESPACE "onnx_trtrepack" CACHE STRING "C++ Namespace to used for ONNX symbols to avoid conflicts.")
set(ONNX_USE_LITE_PROTO ${ONNX2TRT_USE_PROTOBUF_LITE} CACHE BOOL "Use the lite version of protobuf when building ONNX.")
set(ONNX_INSTALL OFF CACHE BOOL "Disable ONNX installtions.")
if(WIN32)
    set(ONNX_USE_MSVC_STATIC_RUNTIME ON CACHE BOOL "Use the static MSVC runtime when building ONNX on Windows with MSVC.")
endif()

# ONNX is expecting python to be supplied as PYTHON_EXECUTABLE, so if we don't already supply that, find it.
if(NOT DEFINED PYTHON_EXECUTABLE)
    find_package(
        Python3
        COMPONENTS Interpreter
        REQUIRED)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
endif()

if(NOT TARGET onnx_proto)
    include(${TensorRT_SOURCE_DIR}/third_party/onnx.cmake)
endif()

target_compile_definitions(onnx_proto PUBLIC
    ONNX_NAMESPACE=${ONNX_NAMESPACE}
)
add_dependencies(onnx_proto protobuf::protoc) # ONNX depends on protoc being available to compile .proto files
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
    target_link_libraries(onnx_proto PUBLIC cppfilesystem)
endif()

if(${RESTORE_CMAKE_LINK_LIBRARIES_ONLY_TARGETS})
    set(CMAKE_LINK_LIBRARIES_ONLY_TARGETS ON)
    unset(RESTORE_CMAKE_LINK_LIBRARIES_ONLY_TARGETS)
endif()
