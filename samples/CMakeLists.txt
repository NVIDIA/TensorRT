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
if((${TRT_BUILD_SAMPLES} OR ${TRT_BUILD_TRTEXEC}) AND NOT ${TRT_BUILD_ONNX_PARSER})
    message(FATAL_ERROR "Building trtexec and/or the other samples requires that TRT_BUILD_ONNX_PARSER=ON")
endif()

if(${TRT_BUILD_ENABLE_NEW_SAMPLES_FLOW})

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
        sampleCharRNN
        sampleEditableTimingCache
        sampleIOFormats
        sampleNamedDimensions
        sampleOnnxMNIST
        sampleProgressMonitor
    )

    if (NOT ${TRT_BUILD_WINML})
        add_sample(
            sampleDynamicReshape
            sampleINT8API
            sampleNonZeroPlugin
        )
    endif()

    # This sample needs to link against nvinfer_plugin.
    if(${TRT_BUILD_PLUGINS})
        add_sample(sampleOnnxMnistCoordConvAC)
    endif()

    if(${TRT_BUILD_ENABLE_DLA} AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" AND NOT WIN32)
        if(TARGET CUDA::cudla)
            set(CUDLA_TARGET CUDA::cudla CACHE STRING "cuDLA library target")
        else()
            find_library(CUDLA_TARGET cudla HINTS
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib
                ${CUDA_ROOT}/lib64 ${CUDA_ROOT}/lib
                /usr/local/cuda/targets/aarch64-linux/lib)
        endif()
        if(CUDLA_TARGET)
            add_sample(sampleCudla)
        else()
            message(STATUS "Skipping sampleCudla: cudla library not found")
        endif()
    endif()
endif() # TRT_BUILD_SAMPLES

include(${CMAKE_CURRENT_LIST_DIR}/workaround.cmake)

foreach(FOLDER IN LISTS trtSampleFolders)
    add_subdirectory(${FOLDER})
endforeach()

else() # TRT_BUILD_ENABLE_NEW_SAMPLES_FLOW

add_custom_target(samples)
add_custom_target(tensorrt_samples)

# Setup aliases for OSS builds (similar to new samples flow)
if(TARGET nvinfer)
    add_library(TRT_SAMPLES::tensorrt ALIAS nvinfer)
endif()

# Define installLibraries as a no-op for OSS builds
function(installLibraries)
endfunction()

if(TRT_SAFETY_INFERENCE_ONLY)
    # Inference-only mode:
    # Only build the safety samples, skipping enterprise samples.
    add_subdirectory(sampleSafeMNIST)
    add_subdirectory(sampleSafePluginV3)
    add_subdirectory(trtSafeExec)
else()
    add_subdirectory(common)

    # Normal mode:
    # - BUILD_SAFE_SAMPLES controls whether safety samples are added.
    # - BUILD_SAMPLES (in the OSS entrypoint) controls enterprise samples.
    if(BUILD_SAFE_SAMPLES)
        add_subdirectory(sampleSafeMNIST)
        add_subdirectory(sampleSafePluginV3)
        add_subdirectory(trtSafeExec)
    endif()

    if(BUILD_SAMPLES)
        add_subdirectory(sampleCharRNN)
        add_subdirectory(sampleDynamicReshape)
        add_subdirectory(sampleEditableTimingCache)
        add_subdirectory(sampleINT8API)
        add_subdirectory(sampleIOFormats)
        add_subdirectory(sampleNamedDimensions)
        add_subdirectory(sampleNonZeroPlugin)
        add_subdirectory(sampleOnnxMNIST)
        add_subdirectory(sampleOnnxMnistCoordConvAC)
        add_subdirectory(sampleProgressMonitor)
        add_subdirectory(trtexec)
        if(TRT_BUILD_ENABLE_DLA AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" AND NOT CMAKE_SYSTEM_NAME STREQUAL "QNX" AND NOT WIN32)
            find_library(CUDLA_TARGET cudla HINTS
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib
                ${CUDA_ROOT}/lib64 ${CUDA_ROOT}/lib
                /usr/local/cuda/targets/aarch64-linux/lib)
            if(CUDLA_TARGET)
                add_subdirectory(sampleCudla)
            else()
                message(STATUS "Skipping sampleCudla: cudla library not found")
            endif()
        endif()
    endif()
endif()

endif()
