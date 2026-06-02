#
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
#

cmake_minimum_required(VERSION 3.31 FATAL_ERROR)
include(cmake/modules/set_ifndef.cmake)
include(cmake/modules/find_library_create_target.cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

set_ifndef(TRT_LIB_DIR ${CMAKE_BINARY_DIR})
set_ifndef(TRT_OUT_DIR ${CMAKE_BINARY_DIR})

# Converts Windows paths
cmake_path(SET TRT_LIB_DIR ${TRT_LIB_DIR})
cmake_path(SET TRT_OUT_DIR ${TRT_OUT_DIR})

cmake_path(SET TRT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Required to export symbols to build *.libs
if(WIN32)
    add_compile_definitions(TENSORRT_BUILD_LIB 1)
endif()

# Set output paths
set(RUNTIME_OUTPUT_DIRECTORY ${TRT_OUT_DIR} CACHE PATH "Output directory for runtime target files")
set(LIBRARY_OUTPUT_DIRECTORY ${TRT_OUT_DIR} CACHE PATH "Output directory for library target files")
set(ARCHIVE_OUTPUT_DIRECTORY ${TRT_OUT_DIR} CACHE PATH "Output directory for archive target files")

if(WIN32)
    set(STATIC_LIB_EXT "lib")
else()
    set(STATIC_LIB_EXT "a")
endif()

file(STRINGS "${TRT_INCLUDE_DIR}/NvInferVersion.h" VERSION_STRINGS REGEX "#define TRT_.*_ENTERPRISE")

foreach(TYPE MAJOR MINOR PATCH BUILD)
    string(REGEX MATCH "TRT_${TYPE}_ENTERPRISE [0-9]+" TRT_TYPE_STRING ${VERSION_STRINGS})
    string(REGEX MATCH "[0-9]+" TRT_${TYPE} ${TRT_TYPE_STRING})
endforeach(TYPE)

set(TRT_VERSION "${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}" CACHE STRING "TensorRT project version")
set(ONNX2TRT_VERSION "${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}" CACHE STRING "ONNX2TRT project version")
set(TRT_SOVERSION "${TRT_MAJOR}" CACHE STRING "TensorRT library so version")
message("Building for TensorRT version: ${TRT_VERSION}, library version: ${TRT_SOVERSION}")

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++)
endif()

set(CMAKE_SKIP_BUILD_RPATH True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90 100 120)
  # SM 110 (Thor) requires CUDA 13.0+.
  if(DEFINED CUDA_VERSION AND CUDA_VERSION VERSION_GREATER_EQUAL "13.0")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 110)
  endif()
endif()
message(STATUS "Generating CUDA code for SMs: ${CMAKE_CUDA_ARCHITECTURES}")

project(TensorRT
        LANGUAGES CXX CUDA
        VERSION ${TRT_VERSION}
        DESCRIPTION "TensorRT is a C++ library that facilitates high-performance inference on NVIDIA GPUs and deep learning accelerators."
        HOMEPAGE_URL "https://github.com/NVIDIA/TensorRT")

if (WIN32)
    enable_language(C)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${TRT_LIB_DIR}/../ CACHE PATH "TensorRT installation" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

option(BUILD_PLUGINS "Build TensorRT plugin" ON)
option(BUILD_PARSERS "Build TensorRT parsers" ON)
option(BUILD_SAMPLES "Build TensorRT samples" ON)
option(BUILD_SAFE_SAMPLES "Build TensorRT safety samples" OFF)
option(TRT_SAFETY_INFERENCE_ONLY "Build only the safety inference components (no safety builders)" OFF)
option(TRT_BUILD_TESTING "Build gtests for TensorRT components" OFF)

############################################################################################
# Early dependency discovery
# These must be found before they are used in target definitions

set(THREADS_PREFER_PTHREAD_FLAG ON)
# QNX has built-in threading support and doesn't need FindThreads
if(NOT CMAKE_SYSTEM_NAME STREQUAL "QNX")
    find_package(Threads REQUIRED)
else()
    # For QNX, create a dummy Threads::Threads target if it doesn't exist
    if(NOT TARGET Threads::Threads)
        add_library(Threads::Threads INTERFACE IMPORTED GLOBAL)
        # QNX threading is built into libc, no explicit linking needed
    endif()
endif()

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")

############################################################################################
# Safety runtime libraries (libnvinfer_safe) used by safety samples and
# inference-only builds.
if(BUILD_SAFE_SAMPLES OR TRT_SAFETY_INFERENCE_ONLY)
    set(TRT_NVINFER_SAFE_NAME "nvinfer_safe")

    # Shared safety runtime.
    find_library(nvinfer_safe_path
        ${TRT_NVINFER_SAFE_NAME}
        PATHS ${TRT_LIB_DIR}
        NO_CMAKE_FIND_ROOT_PATH
    )
    if(NOT nvinfer_safe_path)
        message(FATAL_ERROR "nvinfer_safe library not found. Please ensure safety runtime libraries are available in TRT_LIB_DIR ('${TRT_LIB_DIR}').")
    endif()
    add_library(TRTSAFE::nvinfer_safe_shared SHARED IMPORTED)
    set_target_properties(TRTSAFE::nvinfer_safe_shared PROPERTIES IMPORTED_LOCATION ${nvinfer_safe_path})

    target_link_libraries(TRTSAFE::nvinfer_safe_shared INTERFACE cuda) # nvinfer_safe needs the cuda driver.

    # Debug runtime library (provides debugging features like tensor dumping).
    set(nvinfer_safe_debug_lib_name "${TRT_NVINFER_SAFE_NAME}_debug")
    find_library(nvinfer_safe_debug_path
        ${nvinfer_safe_debug_lib_name}
        PATHS ${TRT_LIB_DIR}
        NO_CMAKE_FIND_ROOT_PATH
    )
    if(NOT nvinfer_safe_debug_path)
        message(FATAL_ERROR "nvinfer_safe_debug library not found. Please ensure debug runtime library is available in TRT_LIB_DIR.")
    endif()
    add_library(TRTSAFE::nvinfer_safe_debug SHARED IMPORTED)
    set_target_properties(TRTSAFE::nvinfer_safe_debug PROPERTIES IMPORTED_LOCATION ${nvinfer_safe_debug_path})

    # Headers for the safety runtime.
    # Try to find include directory relative to lib dir first, then fall back to standard locations
    if(EXISTS "${TRT_LIB_DIR}/../include/NvInfer.h")
        target_include_directories(TRTSAFE::nvinfer_safe_shared INTERFACE ${TRT_LIB_DIR}/../include)
        target_include_directories(TRTSAFE::nvinfer_safe_debug INTERFACE ${TRT_LIB_DIR}/../include)
    elseif(EXISTS "/usr/include/NvInfer.h")
        target_include_directories(TRTSAFE::nvinfer_safe_shared INTERFACE /usr/include)
        target_include_directories(TRTSAFE::nvinfer_safe_debug INTERFACE /usr/include)
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/NvInfer.h")
        target_include_directories(TRTSAFE::nvinfer_safe_shared INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
        target_include_directories(TRTSAFE::nvinfer_safe_debug INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    else()
        message(WARNING "Could not find TensorRT headers. Please ensure they are installed.")
    endif()

    # On QNX, TRT depends on DLA symbols stored in the DriveOS PDK.
    # Since trying to find these shared libs at link time will be difficult, we ignore unresolved symbols in shared libs.
    if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
        target_link_options(TRTSAFE::nvinfer_safe_shared INTERFACE LINKER:--unresolved-symbols=ignore-in-shared-libs)
        target_link_options(TRTSAFE::nvinfer_safe_debug INTERFACE LINKER:--unresolved-symbols=ignore-in-shared-libs)
    endif()
endif()

# OSS safety inference-only mode: require safety samples and disable enterprise
# components.
if(TRT_SAFETY_INFERENCE_ONLY)
    if(NOT BUILD_SAFE_SAMPLES)
        set(BUILD_SAFE_SAMPLES ON CACHE BOOL "Build TensorRT safety samples" FORCE)
    endif()

    set(TRT_SAFETY_INFERENCE_ONLY ON CACHE BOOL "" FORCE)

    # Disable enterprise OSS components for this configuration.
    set(BUILD_PLUGINS OFF CACHE BOOL "" FORCE)
    set(BUILD_PARSERS OFF CACHE BOOL "" FORCE)
    set(BUILD_SAMPLES OFF CACHE BOOL "" FORCE)

    # Add CUDA library directory early so all samples can find it
    if(CUDAToolkit_LIBRARY_DIR)
        link_directories(${CUDAToolkit_LIBRARY_DIR})
    endif()

    # Interface target for safety samples in inference-only mode.
    add_library(trt_global_definitions INTERFACE)

    target_link_libraries(trt_global_definitions INTERFACE
        TRTSAFE::nvinfer_safe_shared
        cudart
        Threads::Threads
    )

    if(NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL "QNX")
        target_link_libraries(trt_global_definitions INTERFACE dl rt)
    endif()
    target_include_directories(trt_global_definitions INTERFACE
        ${TRT_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/samples/common
        ${CMAKE_CURRENT_SOURCE_DIR}/shared
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_compile_options(trt_global_definitions INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

# C++17
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS} -DBUILD_SYSTEM=cmake_oss")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_SYSTEM=cmake_oss")
endif()

############################################################################################
# Cross-compilation settings

set_ifndef(TRT_PLATFORM_ID "x86_64")
message(STATUS "Targeting TRT Platform: ${TRT_PLATFORM_ID}")

############################################################################################
# Debug settings

set(TRT_DEBUG_POSTFIX _debug CACHE STRING "suffix for debug builds")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Building in debug mode ${DEBUG_POSTFIX}")
endif()

############################################################################################
# Dependencies
set(DEFAULT_CUDNN_VERSION 8.9)
set(DEFAULT_PROTOBUF_VERSION 3.20.3)

# Dependency Version Resolution
set_ifndef(CUDNN_VERSION ${DEFAULT_CUDNN_VERSION})
message(STATUS "cuDNN version set to ${CUDNN_VERSION}")
set_ifndef(PROTOBUF_VERSION ${DEFAULT_PROTOBUF_VERSION})
message(STATUS "Protobuf version set to ${PROTOBUF_VERSION}")

if (BUILD_PLUGINS OR BUILD_PARSERS)
    include(third_party/protobuf.cmake)
endif()
if(BUILD_PARSERS)
    configure_protobuf(${PROTOBUF_VERSION})
endif()

# Define library names
set(TRT_NVINFER_NAME "nvinfer")
set(TRT_ONNXPARSER_NAME "nvonnxparser")

# Windows library names have major version appended.
if (MSVC)
    set(nvinfer_lib_name "${TRT_NVINFER_NAME}_${TRT_SOVERSION}${TRT_LIB_SUFFIX}")
    set(nvinfer_plugin_lib_name "nvinfer_plugin_${TRT_SOVERSION}")
    set(nvinfer_vc_plugin_lib_name "nvinfer_vc_plugin_${TRT_SOVERSION}")
    set(nvonnxparser_lib_name "${TRT_ONNXPARSER_NAME}_${TRT_SOVERSION}${TRT_LIB_SUFFIX}")

else()
    set(nvinfer_lib_name ${TRT_NVINFER_NAME})
    set(nvinfer_plugin_lib_name "nvinfer_plugin")
    set(nvinfer_vc_plugin_lib_name "nvinfer_vc_plugin")
    set(nvonnxparser_lib_name ${TRT_ONNXPARSER_NAME})
endif()

find_library_create_target(nvinfer ${nvinfer_lib_name} SHARED "${TRT_LIB_DIR}")

set(CMAKE_CUDA_RUNTIME_LIBRARY "static" CACHE STRING "")

if (NOT MSVC)
    find_library(RT_LIB rt)
endif()

############################################################################################

if(NOT MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcompiler -Wno-deprecated-declarations")
else()
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcompiler")
endif()

############################################################################################
# TensorRT

set(HINT_PATHS "${TRT_OUT_DIR}" "${TRT_LIB_DIR}")

if(BUILD_PLUGINS)
    add_subdirectory(plugin)
else()
    find_library_create_target(${nvinfer_plugin_lib_name} ${nvinfer_plugin_lib_name} SHARED "${HINT_PATHS}")
endif()

if(BUILD_PARSERS)
    add_subdirectory(parsers)
else()
    find_library_create_target(${nvonnxparser_lib_name} ${nvonnxparser_lib_name} SHARED "${HINT_PATHS}")
endif()

if(NOT TARGET trt_global_definitions)
    add_library(trt_global_definitions INTERFACE)
    target_include_directories(trt_global_definitions INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
endif()

add_library(tensorrt_headers INTERFACE)
target_include_directories(tensorrt_headers INTERFACE ${TRT_INCLUDE_DIR})

# Samples
if(BUILD_SAMPLES OR BUILD_SAFE_SAMPLES)
    set(TRT_BUILD_ENABLE_NEW_SAMPLES_FLOW ON)

    # Map OSS option names to the internal names used by samples/CMakeLists.txt.
    set(TRT_BUILD_SAMPLES ${BUILD_SAMPLES})
    set(TRT_BUILD_TRTEXEC ${BUILD_SAMPLES})
    set(TRT_BUILD_ONNX_PARSER ${BUILD_PARSERS})
    set(TRT_BUILD_PLUGINS ${BUILD_PLUGINS})

    # Set defaults for specific feature enablement for samples.
    option(TRT_BUILD_ENABLE_DLA "Build TensorRT with DLA features enabled." OFF)
    option(TRT_BUILD_ENABLE_UNIFIED_BUILDER "Build TensorRT with unified builder (safety) features enabled." ${BUILD_SAFE_SAMPLES})
    option(TRT_BUILD_WINML "Build TensorRT with WinML support." OFF)
    option(TRT_BUILD_ENABLE_MULTIDEVICE "Build TensorRT with multi-device support." OFF)
    set(TRT_BUILD_SAMPLES_LINK_STATIC_TRT OFF CACHE INTERNAL "")

    find_library(NVINFER_LIB nvinfer PATHS ${TRT_LIB_DIR} REQUIRED)
    add_library(tensorrt INTERFACE IMPORTED)
    target_link_libraries(tensorrt INTERFACE ${NVINFER_LIB})
    target_include_directories(tensorrt INTERFACE ${TRT_INCLUDE_DIR})

    target_include_directories(${nvonnxparser_lib_name} INTERFACE ${TRT_INCLUDE_DIR})

    add_subdirectory(shared)

    include(InstallUtils)

    if (TRT_BUILD_TESTING)
        find_package(GTest QUIET)
        if (GTest_FOUND)
            if (NOT TARGET gtest_main)
                add_library(gtest_main ALIAS GTest::gtest_main)
            endif()
        else()
            include(FetchContent)
            FetchContent_Declare(
                googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG        v1.14.0
            )
            FetchContent_MakeAvailable(googletest)
        endif()
        set(TRT_GTEST_DISCOVERY_MODE PRE_TEST CACHE STRING "gtest discovery mode.")
    endif()

    # Route sample binaries to TRT_OUT_DIR.
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TRT_OUT_DIR})

    add_subdirectory(samples)
endif()
