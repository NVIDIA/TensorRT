#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
# This allows us to use TRT libs shipped with the wheel.
set(CMAKE_SHARED_LINKER_FLAGS -Wl,-rpath=$ORIGIN)
set(PY_MODULE_NAME tensorrt)

# -------- PATHS --------

set_ifndef(TENSORRT_ROOT ../)
# Convert to an absolute path.
set_ifndef(ONNX_INC_DIR ${TENSORRT_ROOT}/parsers/)
set_ifndef(PYBIND11_DIR ${EXT_PATH}/pybind11/)

# Source Files
file(GLOB_RECURSE SOURCE_FILES src/*.cpp)

# Find headers
find_path(PY_INCLUDE Python.h HINTS ${EXT_PATH}/python${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION} PATH_SUFFIXES include)

set(PY_TARGET_DIR ${TARGET}-linux-gnu)
find_path(PY_CONFIG_INCLUDE pyconfig.h HINTS ${PY_INCLUDE} PATH_SUFFIXES ${PY_TARGET_DIR}/python${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION})

# -------- GLOBAL COMPILE OPTIONS --------

include_directories(${TENSORRT_ROOT}/include ${PROJECT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/docstrings ${ONNX_INC_DIR} ${PYBIND11_DIR}/include)
link_directories(${TENSORRT_BUILD})
set(CMAKE_CXX_FLAGS "-fvisibility=hidden -std=c++${CPP_STANDARD} -Wno-deprecated-declarations")

# -------- START BUILD PROCESS --------

message(STATUS "TENSORRT_ROOT: ${TENSORRT_ROOT}")
message(STATUS "TENSORRT_BUILD: ${TENSORRT_BUILD}")
message(STATUS "EXT_PATH: ${EXT_PATH}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "CUDA_ROOT: ${CUDA_ROOT}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "ONNX_INC_DIR: ${ONNX_INC_DIR}")
message(STATUS "PYBIND11_DIR: ${PYBIND11_DIR}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "PY_INCLUDE: ${PY_INCLUDE}")
message(STATUS "PY_CONFIG_INCLUDE: ${PY_CONFIG_INCLUDE}")

if (${CMAKE_BUILD_TYPE} STREQUAL Release)
    message(STATUS "Building in Release mode")
    set(TRT_LIBS nvinfer nvonnxparser nvparsers nvinfer_plugin)
else()
    message(STATUS "Building in Debug mode")
    set(TRT_LIBS nvinfer_debug nvonnxparser_debug nvparsers_debug nvinfer_plugin_debug)
endif()

# -------- BUILDING --------

set(LIB_NAME ${PY_MODULE_NAME})

# Set up target
add_library(${LIB_NAME} SHARED ${SOURCE_FILES})
target_include_directories(${LIB_NAME} BEFORE PUBLIC ${PY_INCLUDE} ${PY_CONFIG_INCLUDE})
target_link_libraries(${LIB_NAME} PRIVATE ${TRT_LIBS})

# Note that we have to remove the `lib` prefix from the binding .so's
set_target_properties(${LIB_NAME} PROPERTIES PREFIX "")
