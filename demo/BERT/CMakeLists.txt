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

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(infer_c LANGUAGES CXX)
find_package(CUDA)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.2.3
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

include($ENV{TRT_OSSPATH}/cmake/modules/set_ifndef.cmake)
set_ifndef(TRT_INC_DIR $ENV{TRT_OSSPATH}/include)
set_ifndef(TRT_LIB_DIR $ENV{TRT_LIBPATH})
set_ifndef(TRT_OUT_DIR $ENV{TRT_OSSPATH}/build/out)

include_directories(
    infer_c
    ${CUDA_INCLUDE_DIRS}
    ${TRT_INC_DIR}
)

link_directories(
    ${TRT_OUT_DIR}
    ${TRT_LIB_DIR}
)

pybind11_add_module(infer_c
    infer_c/infer_c.cpp
    infer_c/logging.cpp
)
target_link_libraries(infer_c PRIVATE
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin
)

add_executable(perf
    infer_c/perf.cpp
    infer_c/logging.cpp
)

target_link_libraries(perf
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin
)
