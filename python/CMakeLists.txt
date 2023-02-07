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

cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(PyTensorRT LANGUAGES CXX C)

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if(NOT DEFINED ${var})
        set(${var} ${val})
    endif()
endmacro()

function(message)
    if (VERBOSE)
        _message(${ARGN})
    endif()
endfunction()

# -------- CMAKE OPTIONS --------

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tensorrt/)
set(CPP_STANDARD 11 CACHE STRING "CPP Standard Version")
set(CMAKE_CXX_STANDARD ${CPP_STANDARD})

if (NOT MSVC)
    # This allows us to use TRT libs shipped with standalone wheels.
    set(CMAKE_SHARED_LINKER_FLAGS -Wl,-rpath=$ORIGIN)
endif()

set(PY_MODULE_NAME tensorrt)

# -------- PATHS --------
message(STATUS "EXT_PATH: ${EXT_PATH}")
message(STATUS "TENSORRT_BUILD: ${TENSORRT_BUILD}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "CUDA_ROOT: ${CUDA_ROOT}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set_ifndef(TENSORRT_ROOT ../)
message(STATUS "TENSORRT_ROOT: ${TENSORRT_ROOT}")

set_ifndef(WIN_EXTERNALS ${EXT_PATH})
message(STATUS "WIN_EXTERNALS: ${WIN_EXTERNALS}")

# Convert to an absolute path.
set_ifndef(ONNX_INC_DIR ${TENSORRT_ROOT}/parsers/)
find_path(PYBIND11_DIR pybind11/pybind11.h HINTS ${EXT_PATH} ${WIN_EXTERNALS} PATH_SUFFIXES pybind11/include)

message(STATUS "ONNX_INC_DIR: ${ONNX_INC_DIR}")
message(STATUS "PYBIND11_DIR: ${PYBIND11_DIR}")

# Source Files
file(GLOB_RECURSE SOURCE_FILES src/*.cpp)

set(PYTHON python${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION})
set(PYTHON_LIB_NAME python${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION})
message(STATUS "PYTHON: ${PYTHON}")

# Find headers
if (MSVC)
    find_path(PY_INCLUDE Python.h HINTS ${WIN_EXTERNALS}/${PYTHON} ${EXT_PATH}/${PYTHON} PATH_SUFFIXES include)
    find_path(PY_LIB_DIR ${PYTHON_LIB_NAME}.lib HINTS ${WIN_EXTERNALS}/${PYTHON} ${EXT_PATH}/${PYTHON} PATH_SUFFIXES lib)
    message(STATUS "PY_LIB_DIR: ${PY_LIB_DIR}")
else()
    find_path(PY_INCLUDE Python.h HINTS ${EXT_PATH}/${PYTHON} PATH_SUFFIXES include)
endif()

message(STATUS "PY_INCLUDE: ${PY_INCLUDE}")

if (MSVC)
    set(PY_TARGET_DIR win)
else()
    set(PY_TARGET_DIR ${TARGET}-linux-gnu)
    if (${TARGET} STREQUAL ppc64le)
        set(PY_TARGET_DIR powerpc64le-linux-gnu)
    endif()
endif()

find_path(PY_CONFIG_INCLUDE pyconfig.h HINTS ${PY_INCLUDE} PATH_SUFFIXES ${PY_TARGET_DIR}/${PYTHON} ${PY_TARGET_DIR}/${PYTHON}m)
message(STATUS "PY_CONFIG_INCLUDE: ${PY_CONFIG_INCLUDE}")

# -------- GLOBAL COMPILE OPTIONS --------

include_directories(${TENSORRT_ROOT}/include ${PROJECT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/docstrings ${ONNX_INC_DIR} ${PYBIND11_DIR})
link_directories(${TENSORRT_BUILD})

if (MSVC)
    include_directories(${MSVC_COMPILER_DIR}/include ${MSVC_COMPILER_DIR}/../ucrt/include ${NV_WDKSDK_INC}/um ${NV_WDKSDK_INC}/shared)
    link_directories(${PY_LIB_DIR} ${NV_WDKSDK_LIB}/um/x64 ${MSVC_COMPILER_DIR}/lib/amd64 ${MSVC_COMPILER_DIR}/../ucrt/lib/x64)
endif()


if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd /FS /bigobj")
    if (${NV_GEN_PDB})
        # PDB is only useful in release mode.
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi /bigobj")
        set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
    endif()
else()
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -std=c++${CPP_STANDARD} -Wno-deprecated-declarations")
endif()

# Update linker
if(${NV_TARGET_OS} MATCHES "wddm2")
    if(DEFINED W10_LINKER)
      set(CMAKE_LINKER ${W10_LINKER})
    endif()
endif()

# -------- START BUILD PROCESS --------
message(STATUS "Building in ${CMAKE_BUILD_TYPE} mode")
if (ENABLE_INETWORK_SERIALIZE)
    set(TRT_LIBS nvinfer nvonnxparser nvparsers nvinfer_plugin nvserialize)
else()
    set(TRT_LIBS nvinfer nvonnxparser nvparsers nvinfer_plugin)
endif()

# -------- BUILDING --------

set(LIB_NAME ${PY_MODULE_NAME})

# Set up target
add_library(${LIB_NAME} SHARED ${SOURCE_FILES})
target_include_directories(${LIB_NAME} BEFORE PUBLIC ${PY_CONFIG_INCLUDE} ${PY_INCLUDE})
if (MSVC)
    # For some reason, we must explicitly link against the Python library on Windows.
    target_link_libraries(${LIB_NAME} PRIVATE ${TRT_LIBS} ${PYTHON_LIB_NAME})
else()
    target_link_libraries(${LIB_NAME} PRIVATE ${TRT_LIBS})
endif()

# Note that we have to remove the `lib` prefix from the binding .so's
set_target_properties(${LIB_NAME} PROPERTIES PREFIX "")
