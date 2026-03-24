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

if(NOT TRT_SAFETY_INFERENCE_ONLY)
    add_executable(sample_mnist_safe_build sampleSafeMNISTBuild.cpp)
    target_link_libraries(sample_mnist_safe_build PRIVATE
        trt_samples_common
        TRT_SAMPLES::tensorrt
        TRTSAFE::nvinfer_safe_shared
    )
    # Link ONNX parser if available
    if(TARGET nvonnxparser)
        target_link_libraries(sample_mnist_safe_build PRIVATE nvonnxparser)
    endif()
    if(TRT_OUT_DIR)
        set_target_properties(sample_mnist_safe_build
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
        )
    endif()

    add_dependencies(tensorrt_samples sample_mnist_safe_build)

    installLibraries(
        TARGETS sample_mnist_safe_build
        OPTIONAL
        COMPONENT internal
    )
endif()

add_executable(sample_mnist_safe_infer sampleSafeMNISTInfer.cpp)

if(TRT_SAFETY_INFERENCE_ONLY)
    target_link_libraries(sample_mnist_safe_infer PRIVATE trt_global_definitions)
    target_include_directories(sample_mnist_safe_infer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    # Explicitly link CTK libraries
    if(CUDA_TOOLKIT_ROOT_DIR)
        target_link_directories(sample_mnist_safe_infer PRIVATE
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64
            ${CUDA_TOOLKIT_ROOT_DIR}/lib
            ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib
        )
    endif()
else()
    target_link_libraries(sample_mnist_safe_infer PRIVATE
        trt_samples_common
        TRTSAFE::nvinfer_safe_shared
    )
endif()

if(TRT_OUT_DIR)
    set_target_properties(sample_mnist_safe_infer
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
    )
endif()

add_dependencies(tensorrt_samples sample_mnist_safe_infer)

installLibraries(
    TARGETS sample_mnist_safe_infer
    OPTIONAL
    COMPONENT internal
)
