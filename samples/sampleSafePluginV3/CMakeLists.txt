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
    add_executable(sample_plugin_safe_build
        sampleSafePluginBuild.cpp
        maxPoolPlugin.cpp
        maxPoolPluginCreator.cpp
        maxPoolPluginRuntime.cpp
        maxPoolPluginRuntimeCreator.cpp
        maxPoolKernel.cu
    )
    target_link_libraries(sample_plugin_safe_build PRIVATE
        trt_samples_common
        TRT_SAMPLES::tensorrt
        TRTSAFE::nvinfer_safe_shared
    )
    # Link ONNX parser if available
    if(TARGET nvonnxparser)
        target_link_libraries(sample_plugin_safe_build PRIVATE nvonnxparser)
    endif()
    if(TRT_OUT_DIR)
        set_target_properties(sample_plugin_safe_build
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
        )
    endif()

    add_dependencies(tensorrt_samples sample_plugin_safe_build)

    installLibraries(
        TARGETS sample_plugin_safe_build
        OPTIONAL
        COMPONENT internal
    )
endif()

add_executable(sample_plugin_safe_infer
    sampleSafePluginInfer.cpp
    maxPoolPluginRuntime.cpp
    maxPoolPluginRuntimeCreator.cpp
    maxPoolKernel.cu
)

if(TRT_SAFETY_INFERENCE_ONLY)
    target_link_libraries(sample_plugin_safe_infer PRIVATE trt_global_definitions)
    target_include_directories(sample_plugin_safe_infer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
else()
    target_link_libraries(sample_plugin_safe_infer PRIVATE
        trt_samples_common
        TRTSAFE::nvinfer_safe_shared
    )
endif()

if(TRT_OUT_DIR)
    set_target_properties(sample_plugin_safe_infer
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
    )
endif()

add_dependencies(tensorrt_samples sample_plugin_safe_infer)

installLibraries(
    TARGETS sample_plugin_safe_infer
    OPTIONAL
    COMPONENT internal
)

# ==============================================================================
# PLUGIN LIBRARIES FOR TRTEXEC INTEGRATION
# ==============================================================================

if(NOT TRT_SAFETY_INFERENCE_ONLY)
    # Builder Plugin Library - for trtexec --safe (includes all plugin components)
    add_library(sample_safe_plugin_build_lib SHARED
        maxPoolPlugin.cpp
        maxPoolPluginRuntime.cpp
        maxPoolPluginCreator.cpp
        maxPoolPluginRuntimeCreator.cpp
        maxPoolPluginCreatorInterface.cpp
        maxPoolKernel.cu
    )

    target_compile_definitions(sample_safe_plugin_build_lib PRIVATE
        GEN_PLUGIN_LIB=1  # Enable getSafetyPluginCreator export
    )

    target_link_libraries(sample_safe_plugin_build_lib PRIVATE
        trt_samples_common
        TRT_SAMPLES::tensorrt
        TRTSAFE::nvinfer_safe_shared
    )

    # Link ONNX parser if available
    if(TARGET nvonnxparser)
        target_link_libraries(sample_safe_plugin_build_lib PRIVATE nvonnxparser)
    endif()

    target_include_directories(sample_safe_plugin_build_lib PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    if(TRT_OUT_DIR)
        set_target_properties(sample_safe_plugin_build_lib PROPERTIES
            OUTPUT_NAME "sample_safe_plugin_v3"
            LIBRARY_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
        )
    else()
        set_target_properties(sample_safe_plugin_build_lib PROPERTIES
            OUTPUT_NAME "sample_safe_plugin_v3"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        )
    endif()

    add_dependencies(tensorrt_samples sample_safe_plugin_build_lib)

    installLibraries(
        TARGETS sample_safe_plugin_build_lib
        OPTIONAL
        COMPONENT internal
    )
endif()

# Safe Runtime Plugin Library - for trtexec_safe (runtime components only)
add_library(sample_safe_plugin_runtime_lib SHARED
    maxPoolPluginRuntime.cpp
    maxPoolPluginRuntimeCreator.cpp
    maxPoolPluginRuntimeCreatorInterface.cpp
    maxPoolKernel.cu
)

target_compile_definitions(sample_safe_plugin_runtime_lib PRIVATE
    GEN_PLUGIN_LIB=1  # Enable getSafetyPluginCreator export
)

if(TRT_SAFETY_INFERENCE_ONLY)
    target_link_libraries(sample_safe_plugin_runtime_lib PRIVATE trt_global_definitions)
    target_include_directories(sample_safe_plugin_runtime_lib PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
else()
    target_link_libraries(sample_safe_plugin_runtime_lib PRIVATE
        trt_samples_common
        TRTSAFE::nvinfer_safe_shared
    )
endif()

if(TRT_OUT_DIR)
    set_target_properties(sample_safe_plugin_runtime_lib PROPERTIES
        OUTPUT_NAME "sample_safe_plugin_v3_safe"
        LIBRARY_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
    )
else()
    set_target_properties(sample_safe_plugin_runtime_lib PROPERTIES
        OUTPUT_NAME "sample_safe_plugin_v3_safe"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )
endif()

add_dependencies(tensorrt_samples sample_safe_plugin_runtime_lib)

installLibraries(
    TARGETS sample_safe_plugin_runtime_lib
    OPTIONAL
    COMPONENT internal
)
