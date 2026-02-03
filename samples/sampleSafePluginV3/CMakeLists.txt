# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
    
    set_target_properties(sample_safe_plugin_build_lib PROPERTIES
        OUTPUT_NAME "sample_safe_plugin_v3"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )
    
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

set_target_properties(sample_safe_plugin_runtime_lib PROPERTIES
    OUTPUT_NAME "sample_safe_plugin_v3_safe"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

add_dependencies(tensorrt_samples sample_safe_plugin_runtime_lib)

installLibraries(
    TARGETS sample_safe_plugin_runtime_lib
    OPTIONAL
    COMPONENT internal
)
