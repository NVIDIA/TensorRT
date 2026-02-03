# SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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

add_dependencies(tensorrt_samples sample_mnist_safe_infer)

installLibraries(
    TARGETS sample_mnist_safe_infer
    OPTIONAL
    COMPONENT internal
)
