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
    if(NOT ${var})
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

    find_program(NVCC_EXECUTABLE nvcc HINTS "${CUDA_LIB_DIR}/bin")

    # extract CUDA version
    if(NVCC_EXECUTABLE)
        execute_process(
            COMMAND "${NVCC_EXECUTABLE}" --version
            OUTPUT_VARIABLE NVCC_VERSION_OUTPUT
            ERROR_VARIABLE NVCC_VERSION_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        # Parse the version number from the output
        string(REGEX MATCH "release ([0-9]+)\\.([0-9]+)" CUDA_VERSION_MATCH "${NVCC_VERSION_OUTPUT}")
        if(CUDA_VERSION_MATCH)
            set(CUDA_VERSION_MAJOR "${CMAKE_MATCH_1}")
            set(CUDA_VERSION_MINOR "${CMAKE_MATCH_2}")
            set(CUDA_VER "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")
        else()
            message(FATAL_ERROR "Could not parse CUDA version from nvcc output.")
        endif()
    else()
        message(FATAL_ERROR "nvcc not found in ${CUDA_INST_DIR}/bin")
    endif()

    # Function to check if the current CUDA version is greater than or equal to a specified version
    function(cuda_ge major minor result_var)
        set(VERSION_TO_COMPARE "${major}.${minor}")
        if(CUDA_VER VERSION_GREATER_EQUAL "${VERSION_TO_COMPARE}")
            set(${result_var}
                1
                PARENT_SCOPE)
        else()
            set(${result_var}
                0
                PARENT_SCOPE)
        endif()
    endfunction()

    # Loop through minor versions from 0 to 9
    foreach(minor RANGE 0 9)
        set(result_var "CUDA_GE_11_${minor}")
        cuda_ge(11 ${minor} ${result_var})
    endforeach()

    # Add checks for CUDA 12.x versions
    foreach(minor RANGE 0 9)
        set(result_var "CUDA_GE_12_${minor}")
        cuda_ge(12 ${minor} ${result_var})
    endforeach()

    # Add checks for CUDA 13.x versions
    foreach(minor RANGE 0 9)
        set(result_var "CUDA_GE_13_${minor}")
        cuda_ge(13 ${minor} ${result_var})
    endforeach()

    # Define THOR_SM based on CUDA version
    if(${CUDA_GE_13_0})
        set(THOR_SM 110)
    else()
        set(THOR_SM 101)
    endif()

    set(SAMPLE_SMS "75")

    if(${CUDA_GE_11_0})
        list(APPEND SAMPLE_SMS "80")
    endif()

    if(${CUDA_GE_11_1})
        list(APPEND SAMPLE_SMS "86")
    endif()

    if(${CUDA_GE_11_4})
        list(APPEND SAMPLE_SMS "87")
    endif()

    if(${CUDA_GE_11_8})
        list(APPEND SAMPLE_SMS "89" "90")
    endif()

    # Blackwell support
    if(${CUDA_GE_12_8})
        list(APPEND SAMPLE_SMS "100" ${THOR_SM} "120")
    endif()

    set(NON_HFC_SMS "89" "90" "100" ${THOR_SM} "120")

    if(NOT DEFINED GENCODES)
        set(GENCODES "")

        # Add -gencode flags for each SM in SAMPLE_SMS
        foreach(sm ${SAMPLE_SMS})
            list(APPEND GENCODES "-gencode=arch=compute_${sm},code=sm_${sm}")
        endforeach()

        # Filter out NON_HFC_SMS from SAMPLE_SMS to get HFC_SMS
        set(HFC_SMS ${SAMPLE_SMS})
        foreach(sm ${NON_HFC_SMS})
            list(REMOVE_ITEM HFC_SMS "${sm}")
        endforeach()

        # Get the highest supported forward compatible SM
        if(HFC_SMS)
            list(SORT HFC_SMS)
            list(GET HFC_SMS -1 GEN_PTX_SM)
            # Add PTX generation flag
            list(APPEND GENCODES "-gencode=arch=compute_${GEN_PTX_SM},code=compute_${GEN_PTX_SM}")
        else()
            message(WARNING "No hardware forward compatible SMs found. PTX generation skipped.")
        endif()
    endif()
endif()

message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

find_library(
    _NVINFER_LIB nvinfer
    HINTS ${TRT_LIB}
    PATH_SUFFIXES lib lib64)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

find_library(
    _CUDA_LIB cuda
    HINTS ${CUDA_LIB_DIR}
    PATH_SUFFIXES lib/stubs lib64/stubs)
set_ifndef(CUDA_LIB ${_CUDA_LIB})

# -------- BUILDING --------

add_library(circ_pad_plugin SHARED ${CMAKE_SOURCE_DIR}/circ_plugin_cpp/circ_pad_plugin.cu)
target_compile_options(circ_pad_plugin PRIVATE ${GENCODES})

target_include_directories(
    circ_pad_plugin
    PUBLIC ${CUDA_INC_DIR}
    PUBLIC ${TRT_INCLUDE})

set_property(TARGET circ_pad_plugin PROPERTY CUDA_STANDARD 14)

target_link_libraries(circ_pad_plugin PRIVATE ${NVINFER_LIB})
target_link_libraries(circ_pad_plugin PRIVATE ${CUDA_LIB})
