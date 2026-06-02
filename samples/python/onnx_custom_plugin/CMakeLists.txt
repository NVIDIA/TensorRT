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

cmake_minimum_required(VERSION 3.31 FATAL_ERROR)
project(CustomHardMax LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

if(NOT MSVC)
    # Enable all compile warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-deprecated-declarations")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")
endif()

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if(NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
if(NOT MSVC)
    set_ifndef(TRT_LIB /usr/lib/x86_64-linux-gnu)
    set_ifndef(TRT_INCLUDE /usr/include/x86_64-linux-gnu)

endif()

# Find dependencies:
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

# TensorRT’s nvinfer lib
find_library(
    _NVINFER_LIB nvinfer
    HINTS ${TRT_LIB}
    PATH_SUFFIXES lib lib64)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

# -------- BUILDING --------

add_definitions(-DTENSORRT_BUILD_LIB)

# Add include directories
file(REAL_PATH ${CMAKE_SOURCE_DIR} CMAKE_SOURCE_DIR_REALPATH)

get_filename_component(SAMPLES_COMMON_DIR ${CMAKE_SOURCE_DIR_REALPATH}/../../common/ ABSOLUTE)
get_filename_component(SHARED_DIR ${CMAKE_SOURCE_DIR_REALPATH}/../../../shared ABSOLUTE)
get_filename_component(SAMPLES_DIR ${CMAKE_SOURCE_DIR_REALPATH}/../../ ABSOLUTE)
# Define Hardmax plugin library target
add_library(
    customHardmaxPlugin MODULE
    ${SAMPLES_COMMON_DIR}/logger.cpp ${SHARED_DIR}/utils/fileLock.cpp
    ${CMAKE_SOURCE_DIR_REALPATH}/plugin/customHardmaxPlugin.cpp ${CMAKE_SOURCE_DIR_REALPATH}/plugin/customHardmaxPlugin.h)

target_include_directories(customHardmaxPlugin PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS} ${TRT_INCLUDE} ${CMAKE_SOURCE_DIR_REALPATH}/plugin/
    ${SAMPLES_COMMON_DIR} ${SAMPLES_DIR} ${SHARED_DIR})

# Use C++11
target_compile_features(customHardmaxPlugin PUBLIC cxx_std_17)

# Link TensorRT’s nvinfer lib
target_link_libraries(customHardmaxPlugin PRIVATE ${NVINFER_LIB})
target_link_libraries(customHardmaxPlugin PRIVATE CUDA::cudart_static)
target_link_libraries(customHardmaxPlugin PRIVATE CUDA::cublas)
target_link_libraries(customHardmaxPlugin PRIVATE CUDA::cuda_driver)
