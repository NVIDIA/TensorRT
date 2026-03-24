#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_library(trt_samples_common STATIC
    argsParser.h
    BatchStream.h
    bfloat16.cpp
    bfloat16.h
    buffers.h
    common.h
    debugTensorWriter.cpp
    debugTensorWriter.h
    EntropyCalibrator.h
    ErrorRecorder.h
    getOptions.cpp
    getOptions.h
    getoptWin.h
    half.h
    logger.cpp
    logger.h
    logging.h
    parserOnnxConfig.h
    sampleConfig.h
    sampleDevice.cpp
    sampleDevice.h
    sampleEngines.cpp
    sampleEngines.h
    sampleEntrypoints.h
    sampleInference.cpp
    sampleInference.h
    sampleOptions.cpp
    sampleOptions.h
    sampleReporting.cpp
    sampleReporting.h
    sampleUtils.cpp
    sampleUtils.h
    streamReader.h
)

if(MSVC)
    enable_language(C)
    target_sources(trt_samples_common PRIVATE
        getOpt.c
    )
endif()

target_include_directories(trt_samples_common PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../shared
)

# OSS build mode
if(NOT TRT_BUILD_ENABLE_NEW_SAMPLES_FLOW)
    target_sources(trt_samples_common PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../shared/utils/cacheUtils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../shared/utils/fileLock.cpp
    )
    target_link_libraries(trt_samples_common PUBLIC
        Threads::Threads
        ${CUDA_LIBRARIES}
    )
    if(NOT WIN32 AND NOT ${CMAKE_SYSTEM_NAME} STREQUAL "QNX")
        target_link_libraries(trt_samples_common PUBLIC dl)
    endif()

    # Link nvinfer if available
    if(TARGET nvinfer)
        target_link_libraries(trt_samples_common PUBLIC nvinfer)
    endif()
else()
    # Internal build mode
    target_link_libraries(trt_samples_common PUBLIC
        trt_global_definitions
        trt_shared
        Threads::Threads
        dl
        TRT::cudart
        $<COMPILE_ONLY:TRT_SAMPLES::tensorrt> # Each sample individually must determine its linkage to TRT.
        TRT_SAMPLES::onnxparser
        $<COMPILE_ONLY:CUDA::cupti> # Samples use cuda_profiler_api.h but do not link cupti.
    )

    # For statically-linked samples, we need to upgrade the link to always link TRT rather than letting the samples decide.
    if(${TRT_BUILD_SAMPLES_LINK_STATIC_TRT})
        target_link_libraries(trt_samples_common PUBLIC
            $<LINK_LIBRARY:WHOLE_ARCHIVE,TRT_SAMPLES::tensorrt> # Has to be whole archive so we keep the builder resources correctly.
        )
    endif()

    if(${TRT_BUILD_ENABLE_DLA})
        target_link_libraries(trt_samples_common PUBLIC NVDLA::compiler)
    endif()
endif()
