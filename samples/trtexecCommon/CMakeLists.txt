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

add_library(trtexec_common STATIC
    bfloat16.cpp
    bfloat16.h
    bigInt.cpp
    bigInt.h
    buffers.h
    common.cpp
    common.h
    companionSo.h
    debugTensorWriter.cpp
    debugTensorWriter.h
    ErrorRecorder.h
    globalTimerKernel.cu
    globalTimerKernel.h
    half.h
    idleWait.h
    logger.cpp
    logger.h
    logging.h
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
    safeMemoryTracker.h
    safeErrorRecorder.h
    streamReader.h
)

if (${TRT_BUILD_TESTING})
    include(GoogleTest)
    enable_testing()

    add_executable(trtexec_common_test
        bfloat16.test.cpp
        half.test.cpp
        idleWait.test.cpp
        safeMemoryTracker.test.cpp
        sampleOptions.test.cpp
        sampleTuning.test.cpp
        sampleUtils.test.cpp
    )

    target_link_libraries(trtexec_common_test PRIVATE
        gtest_main
        trtexec_common
    )

    if(${CMAKE_SYSTEM_NAME} STREQUAL "QNX")
        # idleWait.test.cpp interposes TimerTimeout() to exercise kernel-call failures that cannot be
        # induced safely on a shared board; the real call stays reachable as __real_TimerTimeout().
        target_link_options(trtexec_common_test PRIVATE "LINKER:--wrap=TimerTimeout")
    endif()

    gtest_discover_tests(trtexec_common_test DISCOVERY_MODE ${TRT_GTEST_DISCOVERY_MODE})

    if(NOT ${TRT_BUILD_SAMPLES})
        set_target_properties(trtexec_common_test PROPERTIES
            EXCLUDE_FROM_ALL TRUE
        )
    endif()
endif() # TRT_BUILD_TESTING

target_include_directories(trtexec_common PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${NLOHMANN_JSON_INCLUDE_DIRS}
)

target_link_libraries(trtexec_common PUBLIC
    tensorrt_headers
    trt_shared
    trt_global_definitions
    Threads::Threads
)

if(${TRT_BUILD_SAMPLES_LINK_STATIC_TRT})
    target_link_libraries(trtexec_common PUBLIC
        $<COMPILE_ONLY:tensorrt_static>
        nvonnxparser_static
    )
else()
    target_link_libraries(trtexec_common PUBLIC
        $<COMPILE_ONLY:tensorrt>
        nvonnxparser
    )
endif()

if(TARGET TRT::cudart)
    target_link_libraries(trtexec_common PUBLIC TRT::cudart)
else()
    target_link_libraries(trtexec_common PUBLIC CUDA::cudart_static)
endif()

if(NOT WIN32 AND NOT ${CMAKE_SYSTEM_NAME} STREQUAL "QNX")
    target_link_libraries(trtexec_common PUBLIC dl)
endif()

target_link_libraries(trtexec_common PUBLIC $<COMPILE_ONLY:CUDA::cupti>)

# A statically linked build upgrades the link to always pull TRT in rather than leaving it to each tool.
if(${TRT_BUILD_SAMPLES_LINK_STATIC_TRT})
    target_link_libraries(trtexec_common PUBLIC
        $<LINK_LIBRARY:WHOLE_ARCHIVE,tensorrt_static> # Has to be whole archive so we keep the builder resources correctly.
    )
endif()
