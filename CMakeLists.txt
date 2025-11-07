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

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
include(cmake/modules/set_ifndef.cmake)
include(cmake/modules/find_library_create_target.cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

set_ifndef(TRT_LIB_DIR ${CMAKE_BINARY_DIR})
set_ifndef(TRT_OUT_DIR ${CMAKE_BINARY_DIR})

# Converts Windows paths
if(CMAKE_VERSION VERSION_LESS 3.20)
    file(TO_CMAKE_PATH "${TRT_LIB_DIR}" TRT_LIB_DIR)
    file(TO_CMAKE_PATH "${TRT_OUT_DIR}" TRT_OUT_DIR)
else()
    cmake_path(SET TRT_LIB_DIR ${TRT_LIB_DIR})
    cmake_path(SET TRT_OUT_DIR ${TRT_OUT_DIR})
endif()

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

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/NvInferVersion.h" VERSION_STRINGS REGEX "#define TRT_.*_ENTERPRISE")

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

# CUDA targets
set(DEFAULT_CUDA_VERSION 13.0.0)
set_ifndef(CUDA_VERSION ${DEFAULT_CUDA_VERSION})
message(STATUS "CUDA version set to ${CUDA_VERSION}")

if (DEFINED GPU_ARCHS)
  message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
  separate_arguments(GPU_ARCHS)
  foreach(SM IN LISTS GPU_ARCHS)
    list(APPEND CMAKE_CUDA_ARCHITECTURES "${SM}")
  endforeach()
else()
  list(APPEND CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90)

  if(CUDA_VERSION VERSION_GREATER_EQUAL 12.8)
      list(APPEND CMAKE_CUDA_ARCHITECTURES 100 120)
  endif()

  if(CUDA_VERSION VERSION_GREATER_EQUAL 13.0)
        list(APPEND CMAKE_CUDA_ARCHITECTURES 110)
  endif()
  message(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
set(BERT_GENCODES)
# Generate SASS for each architecture
foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
    if (${arch} GREATER_EQUAL 75)
        set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
    endif()
    set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

# Generate PTX for the last architecture in the list.
list(GET CMAKE_CUDA_ARCHITECTURES -1 LATEST_SM)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
if (${LATEST_SM} GREATER_EQUAL 75)
    set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
endif()

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

# C++17
set(CMAKE_CXX_STANDARD 17)
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

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
if (BUILD_PLUGINS OR BUILD_PARSERS)
    include(third_party/protobuf.cmake)
endif()
if(NOT CUB_ROOT_DIR)
  if (CUDA_VERSION VERSION_LESS 11.0)
    set(CUB_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cub CACHE STRING "directory of CUB installation")
  endif()
endif()

## find_package(CUDA) is broken for cross-compilation. Enable CUDA language instead.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    find_package(CUDA ${CUDA_VERSION} REQUIRED)
endif()

include_directories(
    ${CUDA_INCLUDE_DIRS}
)
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

if (DEFINED USE_CUGFX)
    find_library(CUDART_LIB cugfx_dll HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib/x64 lib64)
else()
    find_library(CUDART_LIB cudart_static HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib/x64 lib64)
endif()

if (NOT MSVC)
    find_library(RT_LIB rt)
endif()

set(CUDA_LIBRARIES ${CUDART_LIB})

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

if(BUILD_SAMPLES)
    add_subdirectory(samples)
endif()
