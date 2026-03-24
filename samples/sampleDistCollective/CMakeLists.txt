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

if(${TRT_BUILD_SAMPLES_LINK_STATIC_TRT})
    message(FATAL_ERROR "sampleDistCollective does not support linking with static TensorRT libraries. Please set TRT_BUILD_SAMPLES_LINK_STATIC_TRT to OFF.")
endif()

include(InstallImportedLibraries)

if (${TRT_BUILD_ENABLE_NEW_SAMPLES_FLOW})
    if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/modules/FindNCCL.cmake" AND DEFINED NCCL_LIB_DIRS)
        find_package(NCCL REQUIRED)
    endif()
else()
    find_package(NCCL REQUIRED)
    if(NCCL_VERSION VERSION_LESS "2.19")
        message(FATAL_ERROR "NCCL version ${NCCL_VERSION} is too old. Minimum required: 2.19.0")
    endif()
    if(NCCL_VERSION VERSION_GREATER_EQUAL "3.0")
        message(FATAL_ERROR "NCCL version ${NCCL_VERSION} is not supported. Maximum supported: 2.x")
    endif()
endif()

add_executable(sample_dist_collective
    sampleDistCollective.cpp
)

if (NOT ${TRT_BUILD_ENABLE_NEW_SAMPLES_FLOW})
set_target_properties(sample_dist_collective PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
)
endif()

installImportedLibraries(
    TARGETS NCCL::nccl
    COMPONENT internal
)

target_link_libraries(sample_dist_collective PRIVATE
    trt_samples_common
    TRT_SAMPLES::tensorrt
    NCCL::nccl
    dl # For dlopen/dlsym
)
add_dependencies(tensorrt_samples sample_dist_collective)

installLibraries(
    TARGETS sample_dist_collective
    OPTIONAL
    COMPONENT internal
)
