#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set_ifndef(TRT_LIB_DIR ${CMAKE_BINARY_DIR})
set_ifndef(TRT_OUT_DIR ${CMAKE_BINARY_DIR})

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/NvInferVersion.h" VERSION_STRINGS REGEX "#define NV_TENSORRT_.*")

foreach(TYPE MAJOR MINOR PATCH BUILD)
    string(REGEX MATCH "NV_TENSORRT_${TYPE} [0-9]" TRT_TYPE_STRING ${VERSION_STRINGS})
    string(REGEX MATCH "[0-9]" TRT_${TYPE} ${TRT_TYPE_STRING})
endforeach(TYPE)

foreach(TYPE MAJOR MINOR PATCH)
    string(REGEX MATCH "NV_TENSORRT_SONAME_${TYPE} [0-9]" TRT_TYPE_STRING ${VERSION_STRINGS})
    string(REGEX MATCH "[0-9]" TRT_SO_${TYPE} ${TRT_TYPE_STRING})
endforeach(TYPE)

set(TRT_VERSION "${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}" CACHE STRING "TensorRT project version")
set(ONNX2TRT_VERSION "${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}" CACHE STRING "ONNX2TRT project version")
set(TRT_SOVERSION "${TRT_SO_MAJOR}" CACHE STRING "TensorRT library so version")
message("Building for TensorRT version: ${TRT_VERSION}, library version: ${TRT_SOVERSION}")

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++)
endif()

set(CMAKE_SKIP_BUILD_RPATH True)

project(TensorRT
        LANGUAGES CXX CUDA
        VERSION ${TRT_VERSION}
        DESCRIPTION "TensorRT is a C++ library that facilitates high performance inference on NVIDIA GPUs and deep learning accelerators."
        HOMEPAGE_URL "https://github.com/NVIDIA/TensorRT")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${TRT_LIB_DIR}/../ CACHE PATH "TensorRT installation" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

option(BUILD_PLUGINS "Build TensorRT plugin" ON)
option(BUILD_PARSERS "Build TensorRT parsers" ON)
option(BUILD_SAMPLES "Build TensorRT samples" ON)

# C++14
set(CMAKE_CXX_STANDARD 14)
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

set(DEFAULT_CUDA_VERSION 11.3.1)
set(DEFAULT_CUDNN_VERSION 8.2)
set(DEFAULT_PROTOBUF_VERSION 3.20.1)

# Dependency Version Resolution
set_ifndef(CUDA_VERSION ${DEFAULT_CUDA_VERSION})
message(STATUS "CUDA version set to ${CUDA_VERSION}")
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
    ${CUDNN_ROOT_DIR}/include
)
find_library(CUDNN_LIB cudnn HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} ${CUDNN_ROOT_DIR} PATH_SUFFIXES lib64 lib/x64 lib)
find_library(CUBLAS_LIB cublas HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/x64 lib/stubs)
find_library(CUBLASLT_LIB cublasLt HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/x64 lib/stubs)
if(BUILD_PARSERS)
    configure_protobuf(${PROTOBUF_VERSION})
endif()

find_library_create_target(nvinfer nvinfer SHARED ${TRT_LIB_DIR})
find_library_create_target(nvuffparser nvparsers SHARED ${TRT_LIB_DIR})

find_library(CUDART_LIB cudart_static HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib/x64 lib64)

if (NOT MSVC)
    find_library(RT_LIB rt)
endif()

set(CUDA_LIBRARIES ${CUDART_LIB})

############################################################################################
# CUDA targets

if (DEFINED GPU_ARCHS)
  message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
  separate_arguments(GPU_ARCHS)
else()
  list(APPEND GPU_ARCHS
      53
      60
      61
      70
      75
    )

  string(REGEX MATCH "aarch64" IS_ARM "${TRT_PLATFORM_ID}")
  if (IS_ARM)
    # Xavier (SM72) only supported for aarch64.
    list(APPEND GPU_ARCHS 72)
  endif()

  if (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
    # Ampere GPU (SM80) support is only available in CUDA versions > 11.0
    list(APPEND GPU_ARCHS 80)
  endif()
  if (CUDA_VERSION VERSION_GREATER_EQUAL 11.1)
    list(APPEND GPU_ARCHS 86)
  endif()

  message(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS}")
endif()
set(BERT_GENCODES)
set(STABLE_DIFFUSION_GENCODES)
# Generate SASS for each architecture
foreach(arch ${GPU_ARCHS})
    if (${arch} GREATER_EQUAL 70)
        set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
        set(STABLE_DIFFUSION_GENCODES "${STABLE_DIFFUSION_GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
    endif()
    set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

# Generate PTX for the last architecture in the list.
list(GET GPU_ARCHS -1 LATEST_SM)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
if (${LATEST_SM} GREATER_EQUAL 70)
    set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
    set(STABLE_DIFFUSION_GENCODES "${STABLE_DIFFUSION_GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
endif()

if(NOT MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")
else()
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler")
endif()

############################################################################################
# TensorRT

if(BUILD_PLUGINS)
    add_subdirectory(plugin)
else()
    find_library_create_target(nvinfer_plugin nvinfer_plugin SHARED ${TRT_OUT_DIR} ${TRT_LIB_DIR})
endif()

if(BUILD_PARSERS)
    add_subdirectory(parsers)
else()
    find_library_create_target(nvcaffeparser nvparsers SHARED ${TRT_OUT_DIR} ${TRT_LIB_DIR})
    find_library_create_target(nvonnxparser nvonnxparser SHARED ${TRT_OUT_DIR} ${TRT_LIB_DIR})
endif()

if(BUILD_SAMPLES)
    add_subdirectory(samples)
endif()
