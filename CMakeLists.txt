#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
set_ifndef(TRT_BIN_DIR ${CMAKE_BINARY_DIR})

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/NvInferVersion.h" VERSION_STRINGS REGEX "#define NV_TENSORRT_.*")

foreach(TYPE MAJOR MINOR PATCH BUILD)
    string(REGEX MATCH "NV_TENSORRT_${TYPE} [0-9]" TRT_TYPE_STRING ${VERSION_STRINGS})
    string(REGEX MATCH "[0-9]" TRT_${TYPE} ${TRT_TYPE_STRING})
endforeach(TYPE)

foreach(TYPE MAJOR MINOR PATCH)
    string(REGEX MATCH "NV_TENSORRT_SONAME_${TYPE} [0-9]" TRT_TYPE_STRING ${VERSION_STRINGS})
    string(REGEX MATCH "[0-9]" TRT_SO_${TYPE} ${TRT_TYPE_STRING})
endforeach(TYPE)

set(TRT_VERSION "${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}.${TRT_BUILD}" CACHE STRING "TRT project version")
set(TRT_SOVERSION "${TRT_SO_MAJOR}.${TRT_SO_MINOR}.${TRT_SO_PATCH}" CACHE STRING "TRT library so version")
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
option(NVPARTNER "Build partner repos from source" OFF)
option(NVINTERNAL "Build in NVIDIA internal source tree" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

############################# CROSS COMPILATION SETTINGS ##################################

set_ifndef(TRT_PLATFORM_ID "x86_64")
message(STATUS "Targeting TRT Platform: ${TRT_PLATFORM_ID}")

############################################################################################
set(TRT_DEBUG_POSTFIX _debug CACHE STRING "suffix for debug builds")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Building in debug mode ${DEBUG_POSTFIX}")
endif()

set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS} -DBUILD_SYSTEM=cmake_oss")

if (DEFINED GPU_ARCHS)
  message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
  separate_arguments(GPU_ARCHS)
else()
  list(APPEND GPU_ARCHS
      35
      53
      61
      70
      75
    )
  message(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS}")
endif()
set(BERT_GENCODES)
# Generate SASS for each architecture
foreach(arch ${GPU_ARCHS})
    if (${arch} GREATER_EQUAL 70)
        set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
    endif()
    set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
# Generate PTX for the last architecture in the list.
list(GET GPU_ARCHS -1 LATEST_SM)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
if (${LATEST_SM} GREATER_EQUAL 70)
    set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")

################################### DEPENDENCIES ##########################################
set(DEFAULT_CUDA_VERSION 10.2)
set(DEFAULT_CUDNN_VERSION 7.6)
set(DEFAULT_PROTOBUF_VERSION 3.0.0)
set(DEFAULT_PROTOBUF_INTERNAL_VERSION 10.0)
set(DEFAULT_CUB_VERSION 1.8.0)

# Dependency Version Resolution
set_ifndef(CUDA_VERSION ${DEFAULT_CUDA_VERSION})
message(STATUS "CUDA version set to ${CUDA_VERSION}")
set_ifndef(CUDNN_VERSION ${DEFAULT_CUDNN_VERSION})
message(STATUS "cuDNN version set to ${CUDNN_VERSION}")

if (NVINTERNAL)
    #TODO: Change this to set_ifndef(PROTOBUF_INTERNAL_VERSION ${DEFAULT_PROTOBUF_INTERNAL_VERSION}) once onnx-tensorrts build system is fixed
    set_ifndef(PROTOBUF_VERSION ${DEFAULT_PROTOBUF_VERSION})
    message(STATUS "Protobuf version set to ${PROTOBUF_INTERNAL_VERSION}")
    set_ifndef(CUB_VERSION ${DEFAULT_CUB_VERSION})
    message(STATUS "CUB version set to ${CUB_VERSION}")
    #TODO: Remove this once CMake is fully intergrated in the P4 build system
    set_ifndef(NVINTERNAL_SUFFIX "V2")
else()
    set_ifndef(PROTOBUF_VERSION ${DEFAULT_PROTOBUF_VERSION})
    message(STATUS "Protobuf version set to ${PROTOBUF_VERSION}")
endif()


find_package(Threads REQUIRED)
if (BUILD_PLUGINS OR BUILD_PARSERS)
    include(third_party/zlib.cmake)
    include(third_party/protobuf.cmake)
endif()
if (NVINTERNAL)
########################################### DEPENDENCIES FOR BUILDING IN NVIDIA's TREE ############################################
    set(EXTERNALS ${PROJECT_SOURCE_DIR}/../externals)
    set(CUB_ROOT_DIR ${EXTERNALS}/cub/${CUB_VERSION} CACHE STRING "directory of CUB installation")
    set(Protobuf_DIR ${EXTERNALS}/protobuf/${TRT_PLATFORM_ID} CACHE STRING "directory of PROTOBUF installation")

    ## This needs to be fixed to work with externals
    if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
        find_package(CUDA REQUIRED)
    endif()

    # Set this for ONNX Parser
    set(CUDNN_ROOT_DIR ${EXTERNALS}/cudnn/${TRT_PLATFORM_ID}/${CUDNN_VERSION}/cuda-${CUDA_VERSION})

    include_directories(
        ${CUDNN_ROOT_DIR}/include
        ${CUDA_TOOLKIT_ROOT_DIR}/include
        /usr/local/cuda-${CUDA_VERSION}/include
    )


    #Check externals before using system
    find_library(CUDNN_LIB cudnn HINTS
        ${CUDNN_ROOT_DIR}/lib64 NO_DEFAULT_PATH)
    find_library(CUDNN_LIB cudnn HINTS
        ${CUDNN_ROOT_DIR}/lib64)

    find_library(CUBLAS_LIB cublas HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib NO_DEFAULT_PATH)
    find_library(CUBLAS_LIB cublas HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib)

    if(BUILD_PARSERS)
        #TODO: Change this to configure_protobuf_internal(${PROTOBUF_INTERNAL_VERSION}) once onnx-tensorrts build system is fixed
        configure_protobuf(${PROTOBUF_VERSION})
    endif()
########################################### DEPENDENCIES FOR BUILDING IN NVIDIA's TREE ############################################
else()
########################################### DEPENDENCIES FOR BUILDING OUTSIDE OF NVIDIA ############################################
    if(NOT CUB_ROOT_DIR)
        set(CUB_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cub CACHE STRING "directory of CUB installation")
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
        ${CUDA_TOOLKIT_ROOT_DIR} ${CUDNN_ROOT_DIR} PATH_SUFFIXES lib64 lib)
    find_library(CUBLAS_LIB cublas HINTS
        ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
    # CUBLASLT libraries are only available in CUDA versions > 10. Check for CUDA version here and
    # remove dependency on the libarary and unset BERT_GENCODES.
    if (CUDA_VERSION VERSION_LESS_EQUAL 10.0)
        message(WARNING "Detected CUDA version is <= 10.0! Removing BERT plugins from compilation list.")
        unset(BERT_GENCODES)
    else()
        find_library(CUBLASLT_LIB cublasLt HINTS
            ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
    endif()
    if(BUILD_PARSERS)
        configure_protobuf(${PROTOBUF_VERSION})
    endif()
########################################### DEPENDENCIES FOR BUILDING OUTSIDE OF NVIDIA ############################################
endif()

find_library_create_target(nvinfer nvinfer SHARED ${TRT_LIB_DIR})

if (NOT (NVINTERNAL OR NVPARTNER))
    find_library_create_target(nvuffparser nvparsers SHARED ${TRT_LIB_DIR})
endif()

find_library(CUDART_LIB cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64)
find_library(RT_LIB rt)

set(CUDA_LIBRARIES ${CUDART_LIB})
############################################################################################
# TensorRT

if(BUILD_PLUGINS)
    add_subdirectory(plugin${NVINTERNAL_SUFFIX})
else()
    find_library_create_target(nvinfer_plugin nvinfer_plugin SHARED ${TRT_BIN_DIR} ${TRT_LIB_DIR})
endif()

if(BUILD_PARSERS)
    add_subdirectory(parsers${NVINTERNAL_SUFFIX})
else()
    if(NVPARTNER OR NVINTERNAL)
        find_library_create_target(nvuffparser nvparsers SHARED ${TRT_BIN_DIR} ${TRT_LIB_DIR})
    endif()

    find_library_create_target(nvcaffeparser nvparsers SHARED ${TRT_BIN_DIR} ${TRT_LIB_DIR})
    find_library_create_target(nvonnxparser nvonnxparser SHARED ${TRT_BIN_DIR} ${TRT_LIB_DIR})
endif()

if(BUILD_SAMPLES)
    add_subdirectory(samples${NVINTERNAL_SUFFIX})
endif()

