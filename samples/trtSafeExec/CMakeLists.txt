# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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

add_dependencies(tensorrt_samples trtexec_safe)

installLibraries(
    TARGETS trtexec_safe
    OPTIONAL
    COMPONENT external
)
