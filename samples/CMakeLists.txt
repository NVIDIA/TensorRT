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

# Require the ONNX parser to build samples or trtexec.
if((${TRT_BUILD_SAMPLES} OR ${TRT_BUILD_TRTEXEC}) AND NOT TARGET nvonnxparser)
    message(FATAL_ERROR "Building trtexec (TRT_BUILD_TRTEXEC=${TRT_BUILD_TRTEXEC}) and/or the other samples (TRT_BUILD_SAMPLES=${TRT_BUILD_SAMPLES}) requires the nvonnxparser target")
endif()

# Setup aliases for ease of swapping between static/dynamic TRT.
if(${TRT_BUILD_SAMPLES_LINK_STATIC_TRT})
    add_library(TRT_SAMPLES::tensorrt ALIAS tensorrt_static)
    add_library(TRT_SAMPLES::onnxparser ALIAS nvonnxparser_static)

    # Limit linking jobs to reduce memory usage.
    # libnvinfer_static.a is multiple GB, so linking it into everything in parallel is a recipe for OOM kills.
    # Observation is peak ~6GB per thread on x86 Linux.
    set_property(GLOBAL PROPERTY JOB_POOLS three_jobs=3)
    set(CMAKE_JOB_POOL_LINK three_jobs)
else()
    add_library(TRT_SAMPLES::tensorrt ALIAS tensorrt)
    add_library(TRT_SAMPLES::onnxparser ALIAS nvonnxparser)
endif()

# OSS samples need the ONNX parser path to be included when each sample is built
if(NOT TRT_SAFETY_INFERENCE_ONLY)
    add_subdirectory(common)
endif()

# Target which holds all enabled samples, including trtexec.
# If TRT_BUILD_SAMPLES is disabled, this target may not build many things.
add_custom_target(tensorrt_samples)

set(trtSampleFolders "")

# \brief Adds one or more sample folders to the list of samples.
macro(add_sample)
    list(APPEND trtSampleFolders ${ARGN})
endmacro()


if(${TRT_BUILD_TRTEXEC})
    add_sample(trtexec)
endif()

if(${TRT_BUILD_SAMPLES})
    if (NOT ${TRT_BUILD_TRTEXEC})
        message(WARNING "TRT_BUILD_SAMPLES is enabled but TRT_BUILD_TRTEXEC is not enabled. This may be unintended.")
    endif()

    # Public (OSS) Samples
    add_sample(
        sampleEditableTimingCache
        sampleIOFormats
        sampleNamedDimensions
        sampleOnnxMNIST
        sampleProgressMonitor
    )

    if (NOT ${TRT_BUILD_WINML})
        add_sample(
            sampleDynamicReshape
            sampleNonZeroPlugin
        )
    endif()

    if(${TRT_BUILD_ENABLE_MULTIDEVICE})
        if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/workaround.cmake")
            add_sample(sampleDistCollective)
        endif()
    endif()

    set(CUDLA_SUPPORTED OFF)
    if(${TRT_BUILD_ENABLE_DLA} AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" AND NOT WIN32)
        if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
            message(STATUS "Skipping sampleCudla: cuDLA not supported on QNX (DriveOS 7)")
        else()
            set(CUDLA_SUPPORTED ON)
        endif()
    endif()
    if(CUDLA_SUPPORTED)
        if(TARGET CUDA::cudla)
            add_sample(sampleCudla)
        else()
            message(STATUS "Skipping sampleCudla: CUDA::cudla target not found")
        endif()
    endif()
endif() # TRT_BUILD_SAMPLES

if(${TRT_BUILD_ENABLE_UNIFIED_BUILDER} AND ${TRT_BUILD_TRTEXEC})
    add_sample(trtSafeExec)
    add_sample(sampleSafeMNIST)
    add_sample(sampleSafePluginV3)
elseif(BUILD_SAFE_SAMPLES)
    add_sample(sampleSafeMNIST)
    add_sample(sampleSafePluginV3)
    add_sample(trtSafeExec)
endif()

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/workaround.cmake")
    include(${CMAKE_CURRENT_LIST_DIR}/workaround.cmake)
endif()

foreach(FOLDER IN LISTS trtSampleFolders)
    add_subdirectory(${FOLDER})
endforeach()

