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

if(NOT NLOHMANN_JSON_INCLUDE_DIRS)
    include(FetchNlohmannJson)
endif()

add_library(trt_samples_common STATIC
    argsParser.h
    BatchStream.h
    bfloat16.cpp
    bfloat16.h
    bigInt.cpp
    bigInt.h
    buffers.h
    common.cpp
    common.h
    debugTensorWriter.cpp
    debugTensorWriter.h
    ErrorRecorder.h
    getOptions.cpp
    getOptions.h
    getoptWin.h
    globalTimerKernel.cu
    globalTimerKernel.h
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
    sampleTuning.cpp
    sampleTuning.h
    sampleUtils.cpp
    sampleUtils.h
    safeCommon.h
    safeCudaAllocator.h
    safeErrorRecorder.h
    streamReader.h
)

if(MSVC)
    enable_language(C)
    target_sources(trt_samples_common PRIVATE
        getopt.c
    )
endif()

if (${TRT_BUILD_TESTING})
    include(GoogleTest)
    enable_testing()

    add_executable(trt_samples_common_test
        bfloat16.test.cpp
        getOptions.test.cpp
        half.test.cpp
        sampleOptions.test.cpp
        sampleUtils.test.cpp
    )

    target_link_libraries(trt_samples_common_test PRIVATE
        gtest_main
        trt_samples_common
    )

    gtest_discover_tests(trt_samples_common_test DISCOVERY_MODE ${TRT_GTEST_DISCOVERY_MODE})
endif() # TRT_BUILD_TESTING

target_include_directories(trt_samples_common PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${NLOHMANN_JSON_INCLUDE_DIRS}
)

target_link_libraries(trt_samples_common PUBLIC
    tensorrt_headers
    trt_shared
    trt_global_definitions
    Threads::Threads
    $<COMPILE_ONLY:TRT_SAMPLES::tensorrt> # Each sample individually must determine its linkage to TRT.
    TRT_SAMPLES::onnxparser
)

if(TARGET TRT::cudart)
    target_link_libraries(trt_samples_common PUBLIC TRT::cudart)
else()
    target_link_libraries(trt_samples_common PUBLIC CUDA::cudart_static)
endif()

if(NOT WIN32 AND NOT ${CMAKE_SYSTEM_NAME} STREQUAL "QNX")
    target_link_libraries(trt_samples_common PUBLIC dl)
endif()

target_link_libraries(trt_samples_common PUBLIC $<COMPILE_ONLY:CUDA::cupti>)

# For statically-linked samples, we need to upgrade the link to always link TRT rather than letting the samples decide.
if(${TRT_BUILD_SAMPLES_LINK_STATIC_TRT})
    target_link_libraries(trt_samples_common PUBLIC
        $<LINK_LIBRARY:WHOLE_ARCHIVE,TRT_SAMPLES::tensorrt> # Has to be whole archive so we keep the builder resources correctly.
    )
endif()
