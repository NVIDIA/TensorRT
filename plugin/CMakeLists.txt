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

include(ShouldCompileKernel)

option(TRT_BUILD_INCLUDE_BERT_QKV_PLUGIN "Build the BERT QKV to Context Plugin and related plugins." ON)

# Create the main object library, which is shared between plugin, plugin_internal, and plugin_static.
add_library(trt_plugins OBJECT)
function(add_plugin_source)
    target_sources(trt_plugins PRIVATE ${ARGN})
endfunction()

# Create the VC object lib, used by vc and vc_static.
add_library(trt_vc_plugins OBJECT)
function(add_vc_plugin_source)
    target_sources(trt_vc_plugins PRIVATE ${ARGN})
endfunction()

set(TRT_PLUGIN_NAMES
    cropAndResizePlugin
    decodeBbox3DPlugin
    detectionLayerPlugin
    disentangledAttentionPlugin
    efficientNMSPlugin
    flattenConcat
    generateDetectionPlugin
    gridAnchorPlugin
    groupNormalizationPlugin
    instanceNormalizationPlugin
    modulatedDeformConvPlugin
    multilevelCropAndResizePlugin
    multilevelProposeROI
    multiscaleDeformableAttnPlugin
    nvFasterRCNN
    pillarScatterPlugin
    priorBoxPlugin
    proposalLayerPlugin
    pyramidROIAlignPlugin
    regionPlugin
    reorgPlugin
    resizeNearestPlugin
    roiAlignPlugin
    scatterElementsPlugin
    scatterPlugin
    topkLastDimPlugin
    voxelGeneratorPlugin
)

if(${TRT_BUILD_INCLUDE_BERT_QKV_PLUGIN})
    list(APPEND TRT_PLUGIN_NAMES
        bertQKVToContextPlugin
        embLayerNormPlugin
        fcPlugin
        skipLayerNormPlugin
    )
endif()

add_subdirectory(api)
add_subdirectory(vc)
add_subdirectory(common)

foreach(PLUGIN_NAME IN LISTS TRT_PLUGIN_NAMES)
    add_subdirectory(${PLUGIN_NAME})
endforeach()

set(trt_plugin_include_dirs
    $<BUILD_LOCAL_INTERFACE:${TRT_EXTERNALS_DIR}>
    $<BUILD_LOCAL_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_include_directories(trt_plugins PUBLIC ${trt_plugin_include_dirs})
target_include_directories(trt_vc_plugins PUBLIC ${trt_plugin_include_dirs})

# Use the compile-time dependencies of TRT when compiling the objects before the link stage.
# The final targets will be responsible for selecting the target TRT distribution to use.
target_link_libraries(trt_plugins PRIVATE $<COMPILE_ONLY:tensorrt>)
target_link_libraries(trt_vc_plugins PRIVATE $<COMPILE_ONLY:tensorrt>)

# Use true link dependencies on the global definitions and cudart.
target_link_libraries(trt_plugins PRIVATE trt_global_definitions TRT::cudart)
target_link_libraries(trt_vc_plugins PRIVATE trt_global_definitions TRT::cudart)

foreach(SM IN LISTS CMAKE_CUDA_ARCHITECTURES)
    get_numeric_sm(${SM} "SM")
    target_compile_definitions(trt_plugins PUBLIC "ENABLE_SM${SM}")
    target_compile_definitions(trt_vc_plugins PUBLIC "ENABLE_SM${SM}")
endforeach()

target_compile_options(trt_plugins PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(trt_vc_plugins PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

target_compile_definitions(trt_vc_plugins PRIVATE
    COMPILE_VFC_PLUGIN=1
)

# Create all the library targets, reusing the objects we've compiled in the first step.
add_library(tensorrt_plugins SHARED)
target_link_libraries(tensorrt_plugins PRIVATE trt_plugins)

add_library(tensorrt_plugins_static STATIC)
target_bundle_libraries(tensorrt_plugins_static PRIVATE trt_plugins)

add_library(tensorrt_vc_plugins SHARED)
target_link_libraries(tensorrt_vc_plugins PRIVATE trt_vc_plugins)

add_library(tensorrt_vc_plugins_static STATIC)
target_bundle_libraries(tensorrt_vc_plugins_static PRIVATE trt_plugins)

if(NOT MSVC)
    set(trt_plugins_link_options
        "LINKER:-z,relro"
        "LINKER:-Bsymbolic"
        "LINKER:--no-undefined"
        "LINKER:--no-as-needed"
        "$<$<CONFIG:Release>:LINKER:--strip-all>"
    )
else()
    set(trt_plugins_link_options)
endif()

set(trt_plugin_dependencies
    tensorrt
    trt_global_definitions
)

if(NOT MSVC)
    list(APPEND trt_plugin_dependencies
        Threads::Threads
        CUDA::culibos
    )
endif()

### TRT Plugin Setup
target_link_libraries(tensorrt_plugins PRIVATE ${trt_plugin_dependencies})
target_link_options(tensorrt_plugins PRIVATE ${trt_plugins_link_options})

set_target_properties(
    tensorrt_plugins
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON
               OUTPUT_NAME nvinfer_plugin
               VERSION ${TensorRT_VERSION}
               SOVERSION ${TRT_MAJOR}
)

apply_export_map(
    TARGETS tensorrt_plugins
)

update_windows_output_name(tensorrt_plugins ${TRT_MAJOR} ${TRT_MINOR})

### Static Plugin Setup
set(trt_plugin_static_dependencies
    tensorrt_static
    $<BUILD_LOCAL_INTERFACE:trt_global_definitions>
)

target_include_directories(tensorrt_plugins_static PRIVATE ${trt_plugin_include_dirs})
target_link_libraries(tensorrt_plugins_static PRIVATE ${trt_plugin_static_dependencies})

set_target_properties(
    tensorrt_plugins_static
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON
               OUTPUT_NAME nvinfer_plugin_static
               VERSION ${TensorRT_VERSION}
               SOVERSION ${TRT_MAJOR}
               LINK_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/exports/nvinfer_plugin.map)

if(NOT ${TRT_BUILD_ENABLE_STATIC_LIBS})
    set_target_properties(tensorrt_plugins_static
        PROPERTIES EXCLUDE_FROM_ALL ON
    )
endif()

smoke_test_static_lib(tensorrt_plugins_static)

### VC Plugin Setup
if (NOT MSVC)
    set(trt_vc_plugins_link_options
        "LINKER:-z,relro"
        "LINKER:-Bsymbolic"
        "LINKER:--no-undefined"
        "LINKER:--no-as-needed"
        "$<$<CONFIG:Release>:LINKER:--strip-all>"
    )
else()
    set(trt_vc_plugins_link_options)
endif()

# Target properties for tensorrt_vc_plugins
# This library includes a minimal subset of the plugins used for version compatibility.
target_include_directories(tensorrt_vc_plugins PRIVATE ${trt_plugin_include_dirs})
target_link_libraries(tensorrt_vc_plugins PRIVATE trt_global_definitions)
target_link_options(tensorrt_vc_plugins PRIVATE ${trt_vc_plugins_link_options})

set_target_properties(
    tensorrt_vc_plugins
    PROPERTIES CXX_VISIBILITY_PRESET hidden
                VISIBILITY_INLINES_HIDDEN ON
                OUTPUT_NAME nvinfer_vc_plugin
                VERSION ${TensorRT_VERSION}
                SOVERSION ${TRT_MAJOR}
)

apply_export_map(
    TARGETS tensorrt_vc_plugins
)

update_windows_output_name(tensorrt_vc_plugins ${TRT_MAJOR} ${TRT_MINOR})

### VC Plugin Static Setup
target_include_directories(tensorrt_vc_plugins_static PRIVATE ${trt_plugin_include_dirs})
target_link_libraries(tensorrt_vc_plugins_static PRIVATE ${trt_plugin_static_dependencies})

set_target_properties(
    tensorrt_vc_plugins_static
    PROPERTIES CXX_VISIBILITY_PRESET hidden
                VISIBILITY_INLINES_HIDDEN ON
                OUTPUT_NAME nvinfer_vc_plugin_static
                VERSION ${TensorRT_VERSION}
                SOVERSION ${TRT_MAJOR}
                LINK_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/exports/nvinfer_vc_plugin.map)

if(NOT ${TRT_BUILD_ENABLE_STATIC_LIBS})
    set_target_properties(tensorrt_vc_plugins_static
        PROPERTIES EXCLUDE_FROM_ALL ON
    )
endif()

smoke_test_static_lib(tensorrt_vc_plugins_static)

if(${TRT_BUILD_STUB_LIBS} AND NOT MSVC)
    create_stub_lib(tensorrt_plugins)
    create_stub_lib(tensorrt_vc_plugins)
endif()

installLibraries(
    TARGETS tensorrt_plugins tensorrt_vc_plugins
    OPTIONAL
    COMPONENT external
    EXPORT TensorRT
)

installLibraries(
    TARGETS tensorrt_plugins_static tensorrt_vc_plugins_static
    OPTIONAL
    COMPONENT external
    EXPORT TensorRT_Static
)
