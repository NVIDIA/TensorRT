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
# We need cmake >= 3.8, since 3.8 introduced CUDA as a first class language
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ClipPlugin LANGUAGES CXX CUDA)

if(NOT MSVC)
  # Enable all compile warnings
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-deprecated-declarations")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")
endif()

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if (NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
if(NOT MSVC)
  set_ifndef(TRT_LIB /usr/lib/x86_64-linux-gnu)
  set_ifndef(TRT_INCLUDE /usr/include/x86_64-linux-gnu)
  set_ifndef(CUDA_INC_DIR /usr/local/cuda/include)
endif()

# Find dependencies:
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

# TensorRT's nvinfer lib
find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

# -------- BUILDING --------

add_definitions(-DTENSORRT_BUILD_LIB)

# Add include directories
include_directories(${CUDA_INC_DIR} ${TRT_INCLUDE} ${CMAKE_SOURCE_DIR}/plugin/)

# Define clip plugin library target
add_library(clipplugin MODULE
  ${CMAKE_SOURCE_DIR}/plugin/clipKernel.cu
  ${CMAKE_SOURCE_DIR}/plugin/customClipPlugin.cpp
  ${CMAKE_SOURCE_DIR}/plugin/clipKernel.h
  ${CMAKE_SOURCE_DIR}/plugin/customClipPlugin.h
)

# Use C++11
target_compile_features(clipplugin PUBLIC cxx_std_11)

# Link TensorRT's nvinfer lib
target_link_libraries(clipplugin PRIVATE ${NVINFER_LIB})

# We need to explicitly state that we need all CUDA files
# to be built with -dc as the member functions will be called by
# other libraries and executables (in our case, Python inference scripts)
set_target_properties(clipplugin PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
