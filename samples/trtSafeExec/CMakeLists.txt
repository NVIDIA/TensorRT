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

add_executable(trtexec_safe trtSafeExec.cpp delayStreamKernel.cu)

if(TRT_SAFETY_INFERENCE_ONLY)
    target_link_libraries(trtexec_safe PRIVATE trt_global_definitions)
    target_include_directories(trtexec_safe PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
else()
    target_link_libraries(trtexec_safe PRIVATE trt_samples_common TRTSAFE::nvinfer_safe_debug)
endif()

if(TRT_OUT_DIR)
    set_target_properties(trtexec_safe
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
    )
endif()

add_dependencies(tensorrt_samples trtexec_safe)

installLibraries(
    TARGETS trtexec_safe
    OPTIONAL
    COMPONENT external
    EXPORT TensorRT
)
