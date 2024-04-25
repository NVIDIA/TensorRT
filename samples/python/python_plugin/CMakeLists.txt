#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(CircPadPlugin LANGUAGES CXX CUDA)

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
  set_ifndef(CUDA_LIB_DIR /usr/local/cuda)
endif()

message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

find_library(_CUDA_LIB cuda HINTS ${CUDA_LIB_DIR} PATH_SUFFIXES lib/stubs lib64/stubs)
set_ifndef(CUDA_LIB ${_CUDA_LIB})

# -------- BUILDING --------

add_library(circ_pad_plugin SHARED
  ${CMAKE_SOURCE_DIR}/circ_plugin_cpp/circ_pad_plugin.cu
)

target_include_directories(circ_pad_plugin
    PUBLIC ${CUDA_INC_DIR}
    PUBLIC ${TRT_INCLUDE}
)

set_property(TARGET circ_pad_plugin PROPERTY CUDA_STANDARD 14)

target_link_libraries(circ_pad_plugin PRIVATE ${NVINFER_LIB})
target_link_libraries(circ_pad_plugin PRIVATE ${CUDA_LIB})
