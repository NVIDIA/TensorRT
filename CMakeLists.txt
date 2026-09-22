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

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_LINK_LIBRARIES_ONLY_TARGETS ON)

# Set CMAKE_CUDA_ARCHITECTURES before project() so CMake's CMP0104 policy does not
# auto-detect and overwrite it with the host GPU's SM. CUDAToolkit_VERSION is not
# available yet; SM 110 is conditionally appended after find_package(CUDAToolkit) below.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 100 120)
  set(_TRT_CUDA_ARCHS_DEFAULTED TRUE)
endif()

file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION TRT_VERSION LIMIT_COUNT 1)

project(TensorRT
    LANGUAGES CXX CUDA
    VERSION ${TRT_VERSION}
    DESCRIPTION "TensorRT is a C++ library that facilitates high-performance inference on NVIDIA GPUs and deep learning accelerators."
    HOMEPAGE_URL "https://github.com/NVIDIA/TensorRT"
)

if (WIN32)
    # Samples on Windows use getOpt.c
    enable_language(C)
    # Required to export symbols to build *.libs
    add_compile_definitions(TENSORRT_BUILD_LIB 1)
endif()

option(BUILD_PLUGINS "Build TensorRT plugin" ON)
option(BUILD_PARSERS "Build TensorRT parsers" ON)
option(BUILD_SAMPLES "Build TensorRT samples" ON)
option(BUILD_SAFE_SAMPLES "Build TensorRT safety samples" OFF)
option(BUILD_PYTHON "Build TensorRT python bindings" OFF)
option(TRT_SAFETY_INFERENCE_ONLY "Build only the safety inference components (no safety builders)" OFF)
option(TRT_BUILD_TESTING "Build gtests for TensorRT components" OFF)
option(TRT_BUILD_ALLOW_CCACHE "Allow ccache to be used for builds" ON)
set(TRT_BUILD_PRODUCT "" CACHE STRING
    "TensorRT product package to import: enterprise, automotive, or safe_inference. Empty preserves legacy selection.")
set_property(CACHE TRT_BUILD_PRODUCT PROPERTY STRINGS
    "" enterprise automotive safe_inference)

if(TRT_SAFETY_INFERENCE_ONLY AND NOT BUILD_SAFE_SAMPLES)
    message(STATUS "TRT_SAFETY_INFERENCE_ONLY is ON, enabling BUILD_SAFE_SAMPLES")
    set(BUILD_SAFE_SAMPLES ON CACHE BOOL "Build TensorRT safety samples" FORCE)
endif()

# Speed up rebuilds with ccache when available; respect a user-set launcher.
if(TRT_BUILD_ALLOW_CCACHE AND CMAKE_GENERATOR MATCHES "Makefiles|Ninja")
    find_program(CCACHE_PROGRAM ccache)
    mark_as_advanced(CCACHE_PROGRAM)
    if(CCACHE_PROGRAM)
        foreach(lang IN ITEMS C CXX CUDA)
            set(CMAKE_${lang}_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "Path to ccache")
        endforeach()
        message(STATUS "ccache enabled: ${CCACHE_PROGRAM}")
    endif()
endif()

############################################################################################
# Early dependency discovery
# These must be found before they are used in target definitions

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

# SM 110 (Thor) requires CUDA 13.0+.
# CUDAToolkit_VERSION is only available once the toolkit is found.
if (_TRT_CUDA_ARCHS_DEFAULTED AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
  list(APPEND CMAKE_CUDA_ARCHITECTURES 110)
endif()
message(STATUS "Generating CUDA code for SMs: ${CMAKE_CUDA_ARCHITECTURES}")

############################################################################################
# OSS bridging: top-level setup for plugin and samples code.

# OSS vendors third-party headers under third_party/.
set(TRT_EXTERNALS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

set(_TRT_OSS_SUPPORTED_PRODUCTS enterprise automotive safe_inference)
if(NOT TRT_BUILD_PRODUCT STREQUAL "")
    if(NOT "${TRT_BUILD_PRODUCT}" IN_LIST _TRT_OSS_SUPPORTED_PRODUCTS)
        message(FATAL_ERROR
            "Invalid TRT_BUILD_PRODUCT='${TRT_BUILD_PRODUCT}'. Expected one of: "
            "enterprise, automotive, safe_inference.")
    endif()
    set(_TRT_OSS_SELECTED_PRODUCT "${TRT_BUILD_PRODUCT}")
elseif(TRT_SAFETY_INFERENCE_ONLY)
    set(_TRT_OSS_SELECTED_PRODUCT safe_inference)
elseif(BUILD_SAFE_SAMPLES)
    set(_TRT_OSS_SELECTED_PRODUCT automotive)
else()
    set(_TRT_OSS_SELECTED_PRODUCT enterprise)
endif()

if(TRT_SAFETY_INFERENCE_ONLY
    AND NOT _TRT_OSS_SELECTED_PRODUCT STREQUAL "safe_inference")
    message(FATAL_ERROR
        "TRT_SAFETY_INFERENCE_ONLY=ON requires "
        "TRT_BUILD_PRODUCT=safe_inference.")
endif()
if(_TRT_OSS_SELECTED_PRODUCT STREQUAL "safe_inference"
    AND NOT TRT_SAFETY_INFERENCE_ONLY)
    message(FATAL_ERROR
        "TRT_BUILD_PRODUCT=safe_inference requires "
        "TRT_SAFETY_INFERENCE_ONLY=ON.")
endif()
if(BUILD_SAFE_SAMPLES AND _TRT_OSS_SELECTED_PRODUCT STREQUAL "enterprise")
    message(FATAL_ERROR
        "BUILD_SAFE_SAMPLES=ON is not supported with "
        "TRT_BUILD_PRODUCT=enterprise.")
endif()

message(STATUS "Importing TensorRT product: ${_TRT_OSS_SELECTED_PRODUCT}")
if(_TRT_OSS_SELECTED_PRODUCT STREQUAL "safe_inference")
    find_package(TensorRT-SafeInference CONFIG REQUIRED)
elseif(_TRT_OSS_SELECTED_PRODUCT STREQUAL "automotive")
    find_package(TensorRT-Automotive CONFIG REQUIRED)
else()
    find_package(TensorRT-Enterprise CONFIG REQUIRED)
endif()

# Translate back to internal names for the rest of the build.
set(TRT_PRODUCT_IS_ENTERPRISE     "${TensorRT_PRODUCT_IS_ENTERPRISE}")
set(TRT_PRODUCT_IS_RTX            "${TensorRT_PRODUCT_IS_RTX}")
set(TRT_PRODUCT_IS_AUTOMOTIVE     "${TensorRT_PRODUCT_IS_AUTOMOTIVE}")
set(TRT_PRODUCT_IS_SAFE_INFERENCE "${TensorRT_PRODUCT_IS_SAFE_INFERENCE}")
set(TRT_MAJOR "${TensorRT_VERSION_MAJOR}")
set(TRT_MINOR "${TensorRT_VERSION_MINOR}")
set(TRT_PATCH "${TensorRT_VERSION_PATCH}")

# Since the samples/ and plugins/ folders are mirrors of the TRT internal build, they use unnamespaced (i.e., non-imported) targets.
# So we have to rebuild that target index from the imported targets.

if(NOT TRT_SAFETY_INFERENCE_ONLY)
    add_library(tensorrt ALIAS ${TensorRT_CMAKE_NAMESPACE}::tensorrt)
    add_library(tensorrt_static INTERFACE) # Add a dummy TRT-static target, it's not supported anymore.

    if(NOT BUILD_PARSERS)
        add_library(nvonnxparser ALIAS ${TensorRT_CMAKE_NAMESPACE}::nvonnxparser)
    endif()

    if(NOT BUILD_PLUGINS)
        add_library(tensorrt_plugins ALIAS ${TensorRT_CMAKE_NAMESPACE}::tensorrt_plugins)
    endif()
endif()

if(BUILD_SAFE_SAMPLES)
    add_library(nvinfer_safe_shared ALIAS ${TensorRT_CMAKE_NAMESPACE}::nvinfer_safe_shared)
    add_library(nvinfer_safe_debug ALIAS ${TensorRT_CMAKE_NAMESPACE}::nvinfer_safe_debug)
endif()

add_library(tensorrt_headers ALIAS ${TensorRT_CMAKE_NAMESPACE}::tensorrt_headers)
add_library(TRT::cudart ALIAS ${TensorRT_CMAKE_NAMESPACE}::trt_cudart)

include(InstallUtils)
include(BundleLibraries)
include(SymbolExports)
include(Stubify)
include(WindowsLibSuffixes)
include(StaticLibSmokeTest)
include(ImportDL)

# OSS safety inference-only mode: require safety samples and disable enterprise
# components.
if(TRT_SAFETY_INFERENCE_ONLY)
    # Disable enterprise OSS components for this configuration.
    set(BUILD_PLUGINS OFF CACHE BOOL "" FORCE)
    set(BUILD_PARSERS OFF CACHE BOOL "" FORCE)
    set(BUILD_SAMPLES OFF CACHE BOOL "" FORCE)

    # Interface target for safety samples in inference-only mode.
    add_library(trt_global_definitions INTERFACE)

    target_link_libraries(trt_global_definitions INTERFACE
        TRTSAFE::nvinfer_safe_shared
        CUDA::cudart
        Threads::Threads
    )

    # CUDA compiler identification receives these PDK dependencies through the
    # QNX Safety toolchain. Add them here only for CXX-linked sample targets;
    # CUDA-linked targets already inherit the toolchain flags.
    if(CMAKE_SYSTEM_NAME STREQUAL "QNX" AND TRT_QNX_SAFE_PDK_LIBRARIES)
        foreach(TRT_QNX_SAFE_PDK_LIBRARY IN LISTS TRT_QNX_SAFE_PDK_LIBRARIES)
            target_link_libraries(trt_global_definitions INTERFACE
                "$<$<LINK_LANGUAGE:CXX>:${TRT_QNX_SAFE_PDK_LIBRARY}>")
        endforeach()
        target_link_options(trt_global_definitions INTERFACE
            "$<$<LINK_LANGUAGE:CXX>:-lslog2>")
    endif()

    if(NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL "QNX")
        target_link_libraries(trt_global_definitions INTERFACE dl rt)
    endif()
    target_include_directories(trt_global_definitions INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/samples/common
        ${CMAKE_CURRENT_SOURCE_DIR}/shared
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_compile_options(trt_global_definitions INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

# C++17
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define library names
set(TRT_ONNXPARSER_NAME "nvonnxparser")

set(CMAKE_CUDA_RUNTIME_LIBRARY "static" CACHE STRING "")

############################################################################################
# TensorRT

if(NOT TARGET trt_global_definitions)
    add_library(trt_global_definitions INTERFACE)
    target_include_directories(trt_global_definitions INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(trt_global_definitions INTERFACE
        TRT_WINML=$<BOOL:${TRT_PRODUCT_IS_RTX}>
    )
endif()

if(BUILD_PLUGINS)
    option(TRT_BUILD_ENABLE_DLA "Build TensorRT with DLA features enabled." ${TensorRT_ENABLE_DLA})
    set(TRT_BUILD_ENABLE_STATIC_LIBS OFF CACHE INTERNAL "Static libs are no longer supported")
    option(TRT_BUILD_STUB_LIBS "Build linker stub libraries." OFF)

    add_subdirectory(plugin)
endif()

if(BUILD_PARSERS)
    add_subdirectory(parsers)
endif()

# Python bindings
if(BUILD_PYTHON)
    include(FetchContent)
    include(Platforms)

    # nvonnxparser is always defined (built or imported), regardless of BUILD_PARSERS.
    set(TRT_BUILD_ONNX_PARSER ON)
    set(TRT_BUILD_PLUGINS ${BUILD_PLUGINS})

    if(NOT DEFINED TRT_BUILD_PLATFORM)
        set(TRT_BUILD_PLATFORM ${CMAKE_SYSTEM_PROCESSOR})
    endif()

    # Default to bindings for just the interpreter's Python version; callers
    # can still expand by passing -DTRT_BUILD_PYTHON_PY_VERSIONS=...
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(TRT_BUILD_PYTHON_PY_VERSIONS "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}"
        CACHE STRING "The list of python versions to build TensorRT bindings for.")

    add_library(tensorrt_lean_runtime ALIAS ${TensorRT_CMAKE_NAMESPACE}::tensorrt_lean_runtime)

    add_library(tensorrt_dispatch_runtime ALIAS ${TensorRT_CMAKE_NAMESPACE}::tensorrt_dispatch_runtime)

    # OSS uses the system Python headers (the Debian/Ubuntu python<ver>-dev layout).
    set(TRT_BUILD_PYTHON_EXTERNALS_PATH "/usr/include" CACHE PATH
        "Path to the parent folder of the versioned python headers/libs.")

    # Variables consumed by python/packaging when building the bindings wheel.
    # (TensorRT_VERSION / TensorRT_SOURCE_DIR are provided by project(TensorRT VERSION ...).)
    string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} TRT_LOWERCASE_CMAKE_SYSTEM_PROCESSOR)
    set(TENSORRT_BASE_NAME "nvinfer")
    set(TRT_CUDA_VERSION "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
    set(TensorRT_PACKAGE_VERSION "${TensorRT_VERSION}")
    set(TRT_BUILD_PYTHON_STANDALONE_WHEELS OFF)

    if(BUILD_PLUGINS)
        set(TRT_BUILD_PLUGINS ON)
    else()
        add_library(nvinfer_plugin ALIAS ${TensorRT_CMAKE_NAMESPACE}::tensorrt_plugins)
    endif()
    add_subdirectory(python)
endif()

# Samples
if(BUILD_SAMPLES OR BUILD_SAFE_SAMPLES)
    set(TRT_BUILD_ENABLE_NEW_SAMPLES_FLOW ON)

    # Map OSS option names to the internal names used by samples/CMakeLists.txt.
    set(TRT_BUILD_SAMPLES ${BUILD_SAMPLES})
    set(TRT_BUILD_TRTEXEC ${BUILD_SAMPLES})
    # nvonnxparser is always available, either built from source or imported
    # from the TensorRT package.
    set(TRT_BUILD_ONNX_PARSER ON)
    add_compile_definitions(TRT_BUILD_ONNX_PARSER=1)
    set(TRT_BUILD_PLUGINS ${BUILD_PLUGINS})

    # Set defaults for specific feature enablement for samples.
    option(TRT_BUILD_ENABLE_DLA "Build TensorRT with DLA features enabled." OFF)
    option(TRT_BUILD_ENABLE_UNIFIED_BUILDER "Build TensorRT with unified builder (safety) features enabled." ${BUILD_SAFE_SAMPLES})

    # The companion library a safety engine pairs with only exists in a TensorRT built with the unified
    # builder, so the sample code that saves and loads one compiles only there. TensorRT forwards the same
    # flag under the same name when it builds these samples itself.
    target_compile_definitions(trt_global_definitions INTERFACE
        ENABLE_UNIFIED_BUILDER=$<BOOL:${TRT_BUILD_ENABLE_UNIFIED_BUILDER}>
    )

    option(TRT_BUILD_ENABLE_MULTIDEVICE "Build TensorRT with multi-device support." OFF)
    set(TRT_BUILD_SAMPLES_LINK_STATIC_TRT OFF CACHE INTERNAL "")

    if(NOT (TRT_SAFETY_INFERENCE_ONLY AND CMAKE_SYSTEM_NAME STREQUAL "QNX"))
        add_subdirectory(shared)
    endif()

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

    add_subdirectory(samples)
endif()
