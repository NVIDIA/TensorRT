#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if (${TRT_BUILD_ENABLE_NEW_PLUGIN_FLOW})

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
    batchedNMSPlugin
    batchTilePlugin
    clipPlugin
    coordConvACPlugin
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
    leakyReluPlugin
    modulatedDeformConvPlugin
    multilevelCropAndResizePlugin
    multilevelProposeROI
    multiscaleDeformableAttnPlugin
    nmsPlugin
    normalizePlugin
    nvFasterRCNN
    pillarScatterPlugin
    priorBoxPlugin
    proposalLayerPlugin
    proposalPlugin
    pyramidROIAlignPlugin
    regionPlugin
    reorgPlugin
    resizeNearestPlugin
    roiAlignPlugin
    scatterElementsPlugin
    scatterPlugin
    specialSlicePlugin
    splitPlugin
    voxelGeneratorPlugin
)



if(${TRT_BUILD_INCLUDE_BERT_QKV_PLUGIN})
    list(APPEND TRT_PLUGIN_NAMES
        bertQKVToContextPlugin
        embLayerNormPlugin
        fcPlugin
        geluPlugin
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
    ${TensorRT_SOURCE_DIR}/externals
    ${CMAKE_CURRENT_LIST_DIR}
)

target_include_directories(trt_plugins PUBLIC ${trt_plugin_include_dirs})
target_include_directories(trt_vc_plugins PUBLIC ${trt_plugin_include_dirs})

# Use the compile-time dependencies of TRT when compiling the objects before the link stage.
# The final targets will be responsible for selecting the target TRT distribution to use.
target_link_libraries(trt_plugins PRIVATE $<COMPILE_ONLY:tensorrt>)
target_link_libraries(trt_vc_plugins PRIVATE $<COMPILE_ONLY:tensorrt>)

# Use true link dependencies on the global definitions and cudart.
target_link_libraries(trt_plugins PUBLIC trt_global_definitions TRT::cudart trt_cccl_headers)
target_link_libraries(trt_vc_plugins PUBLIC trt_global_definitions TRT::cudart)

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
add_library(tensorrt_plugins_internal SHARED)
add_library(tensorrt_plugins_static STATIC)
foreach(lib tensorrt_plugins tensorrt_plugins_internal tensorrt_plugins_static)
    target_link_libraries(${lib} PRIVATE trt_plugins)
endforeach()

add_library(tensorrt_vc_plugins SHARED)
add_library(tensorrt_vc_plugins_static STATIC)
foreach(lib tensorrt_vc_plugins tensorrt_vc_plugins_static)
    target_link_libraries(${lib} PRIVATE trt_vc_plugins)
endforeach()

if (NOT MSVC)
    set(trt_plugins_link_options
        "LINKER:--version-script=${CMAKE_CURRENT_LIST_DIR}/exports.map"
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
               LINK_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/exports.map
)

update_windows_output_name(tensorrt_plugins ${TRT_MAJOR} ${TRT_MINOR})

if (NOT MSVC)
    set(trt_plugins_internal_link_options
        "LINKER:--version-script=${TensorRT_SOURCE_DIR}/Exports-plugin_internal.map"
        "LINKER:-z,relro"
        "LINKER:-Bsymbolic"
        "LINKER:--no-undefined"
        "LINKER:--no-as-needed"
        "$<$<CONFIG:Release>:LINKER:--strip-all>"
    )
else()
    set(trt_plugins_internal_link_options)
endif()

### Internal Plugin Setup
# This library is effectively the same as tensorrt_plugins, but without stripped symbols.
target_include_directories(tensorrt_plugins_internal PUBLIC ${trt_plugin_include_dirs})
target_link_libraries(tensorrt_plugins_internal PRIVATE ${trt_plugin_dependencies})
target_link_options(tensorrt_plugins_internal PRIVATE ${trt_plugins_internal_link_options})

set_target_properties(
    tensorrt_plugins_internal
    PROPERTIES CXX_VISIBILITY_PRESET default
               VISIBILITY_INLINES_HIDDEN OFF
               WINDOWS_EXPORT_ALL_SYMBOLS ON # Allows tests to reach internal symbols without __declspec(dllexport)
               OUTPUT_NAME nvinfer_plugin_internal
               VERSION ${TensorRT_VERSION}
               SOVERSION ${TRT_MAJOR}
               LINK_DEPENDS ${TensorRT_SOURCE_DIR}/Exports-plugin_internal.map
)

update_windows_output_name(tensorrt_plugins_internal ${TRT_MAJOR} ${TRT_MINOR})

### Static Plugin Setup
set(trt_plugin_static_dependencies
    tensorrt_static
    trt_global_definitions
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
               LINK_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/exports.map)

if(NOT ${TRT_BUILD_ENABLE_STATIC_LIBS})
    set_target_properties(tensorrt_plugins_static
        PROPERTIES EXCLUDE_FROM_ALL ON
    )
endif()

smoke_test_static_lib(tensorrt_plugins_static)

### VC Plugin Setup
if (NOT MSVC)
    set(trt_vc_plugins_link_options
        "LINKER:--version-script=${CMAKE_CURRENT_LIST_DIR}/exports-vfc_plugin.map"
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
                LINK_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/exports-vfc_plugin.map
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
                LINK_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/exports-vfc_plugin.map)

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
    TARGETS tensorrt_plugins tensorrt_plugins_static tensorrt_vc_plugins tensorrt_vc_plugins_static
    OPTIONAL
    COMPONENT external
)

installLibraries(
    TARGETS tensorrt_plugins_internal
    OPTIONAL
    COMPONENT internal
)

else() # TRT_BUILD_ENABLE_NEW_PLUGIN_FLOW
add_custom_target(plugin)

set(TARGET_NAME ${nvinfer_plugin_lib_name})
set(SHARED_TARGET ${TARGET_NAME})
set(STATIC_TARGET ${TARGET_NAME}_static)
set(VFC_TARGET_NAME ${nvinfer_vc_plugin_lib_name})
set(VFC_SHARED_TARGET ${VFC_TARGET_NAME})

set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PLUGIN_EXPORT_MAP ${TARGET_DIR}/exports.map)
set(VFC_PLUGIN_EXPORT_MAP ${TARGET_DIR}/exports-vfc_plugin.map)

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

foreach(SM IN LISTS CMAKE_CUDA_ARCHITECTURES)
    get_numeric_sm(${SM} "SM")
    set(ENABLED_SMS "${ENABLED_SMS} -DENABLE_SM${SM}")
endforeach()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ENABLED_SMS}")
set(PLUGIN_SOURCES)
set(PLUGIN_CU_SOURCES)
set(VFC_PLUGIN_SOURCES)
set(VFC_PLUGIN_CU_SOURCES)

# Backwards compat versions of the functions used by the new flow that bind to the old flow.
# We have to add C++ sources to PLUGIN_SOURCES, and Cuda sources to PLUGIN_CU_SOURCES.
macro(add_plugin_source)
    foreach(SRC_FILE IN ITEMS ${ARGN})
        if(SRC_FILE MATCHES "^.*((\.h)|(\.cpp)|(\.hpp))$")
            list(APPEND PLUGIN_SOURCES ${CMAKE_CURRENT_LIST_DIR}/${SRC_FILE})
        else()
            list(APPEND PLUGIN_CU_SOURCES ${CMAKE_CURRENT_LIST_DIR}/${SRC_FILE})
        endif()
    endforeach()
    set(PLUGIN_SOURCES ${PLUGIN_SOURCES} PARENT_SCOPE)
    set(PLUGIN_CU_SOURCES ${PLUGIN_CU_SOURCES} PARENT_SCOPE)
endmacro()

macro(add_vc_plugin_source)
    foreach(SRC_FILE IN ITEMS ${ARGN})
        if(SRC_FILE MATCHES "^.*((\.h)|(\.cpp)|(\.hpp))$")
            list(APPEND VFC_PLUGIN_SOURCES ${CMAKE_CURRENT_LIST_DIR}/${SRC_FILE})
        else()
            list(APPEND VFC_PLUGIN_CU_SOURCES ${CMAKE_CURRENT_LIST_DIR}/${SRC_FILE})
        endif()
    endforeach()
    set(VFC_PLUGIN_SOURCES ${VFC_PLUGIN_SOURCES} PARENT_SCOPE)
    set(VFC_PLUGIN_CU_SOURCES ${VFC_PLUGIN_CU_SOURCES} PARENT_SCOPE)
endmacro()

set(PLUGIN_LISTS
    batchTilePlugin
    batchedNMSPlugin
    clipPlugin
    coordConvACPlugin
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
    leakyReluPlugin
    modulatedDeformConvPlugin
    multilevelCropAndResizePlugin
    multilevelProposeROI
    multiscaleDeformableAttnPlugin
    nmsPlugin
    normalizePlugin
    nvFasterRCNN
    pillarScatterPlugin
    priorBoxPlugin
    proposalLayerPlugin
    proposalPlugin
    pyramidROIAlignPlugin
    regionPlugin
    reorgPlugin
    resizeNearestPlugin
    roiAlignPlugin
    scatterElementsPlugin
    scatterPlugin
    specialSlicePlugin
    splitPlugin
    voxelGeneratorPlugin)

# Add BERT sources if ${BERT_GENCODES} was populated
if(BERT_GENCODES)
    set(BERT_CU_SOURCES)
    set(PLUGIN_LISTS ${PLUGIN_LISTS} bertQKVToContextPlugin embLayerNormPlugin fcPlugin geluPlugin skipLayerNormPlugin)
    include_directories(bertQKVToContextPlugin/fused_multihead_attention
                       /include bertQKVToContextPlugin/fused_multihead_attention_v2/include)
endif()

include_directories(common common/kernels ${CMAKE_SOURCE_DIR}/third_party)


foreach(PLUGIN_ITER ${PLUGIN_LISTS})
    include_directories(${PLUGIN_ITER})
    add_subdirectory(${PLUGIN_ITER})
endforeach(PLUGIN_ITER)

# Add common
add_subdirectory(common)
add_subdirectory(vc)

# Set gencodes
set_source_files_properties(${PLUGIN_CU_SOURCES} PROPERTIES COMPILE_FLAGS "${GENCODES} ${ENABLED_SMS}")
list(APPEND PLUGIN_SOURCES "${PLUGIN_CU_SOURCES}")
if(BERT_CU_SOURCES)
    set_source_files_properties(${BERT_CU_SOURCES} PROPERTIES COMPILE_FLAGS "${BERT_GENCODES} ${ENABLED_SMS}")
    list(APPEND PLUGIN_SOURCES "${BERT_CU_SOURCES}")
endif()

list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/api/inferPlugin.cpp")
list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../samples/common/logger.cpp")

################################## SHARED LIBRARY #######################################

add_library(${SHARED_TARGET} SHARED ${PLUGIN_SOURCES})

target_include_directories(
    ${SHARED_TARGET}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${PROJECT_SOURCE_DIR}/common
    PUBLIC ${CUDA_INSTALL_DIR}/include
    PRIVATE ${TARGET_DIR})

if(CUDA_VERSION VERSION_LESS 11.0)
    target_include_directories(${SHARED_TARGET} PUBLIC ${CUB_ROOT_DIR})
endif()

set_target_properties(
    ${SHARED_TARGET}
    PROPERTIES CXX_STANDARD "17"
               CXX_STANDARD_REQUIRED "YES"
               CXX_EXTENSIONS "NO"
               ARCHIVE_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
               LIBRARY_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}")

if(MSVC)
    set_target_properties(${SHARED_TARGET} PROPERTIES LINK_FLAGS "/DEF:${PLUGIN_EXPORT_DEF}")
else()
    set_target_properties(
        ${SHARED_TARGET}
        PROPERTIES LINK_FLAGS
                   "-Wl,--exclude-libs,ALL -Wl,-Bsymbolic -Wl,--version-script=${PLUGIN_EXPORT_MAP} -Wl,--no-undefined")
endif()

set_target_properties(${SHARED_TARGET} PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})

set_target_properties(${SHARED_TARGET} PROPERTIES VERSION ${TRT_VERSION} SOVERSION ${TRT_SOVERSION})

set_property(TARGET ${SHARED_TARGET} PROPERTY CUDA_STANDARD 17)

target_link_directories(${SHARED_TARGET} PUBLIC ${CUDA_ROOT}/lib)

target_link_libraries(${SHARED_TARGET} ${CUDART_LIB} ${${nvinfer_lib_name}_LIB_PATH} ${CMAKE_DL_LIBS})

# Needed when static linking CUDART
if(NOT MSVC)
    target_link_libraries(${SHARED_TARGET} Threads::Threads ${RT_LIB})
endif()

################################## STATIC LIBRARY #######################################

add_library(${STATIC_TARGET} STATIC ${PLUGIN_SOURCES})

target_include_directories(
    ${STATIC_TARGET}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${CUB_ROOT_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/common
    PUBLIC ${CUDA_INSTALL_DIR}/include
    PRIVATE ${TARGET_DIR})

set_target_properties(
    ${STATIC_TARGET}
    PROPERTIES CXX_STANDARD "17"
               CXX_STANDARD_REQUIRED "YES"
               CXX_EXTENSIONS "NO"
               ARCHIVE_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
               LIBRARY_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}")

set_target_properties(${STATIC_TARGET} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")

set_target_properties(${STATIC_TARGET} PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})

set_target_properties(${STATIC_TARGET} PROPERTIES VERSION ${TRT_VERSION} SOVERSION ${TRT_SOVERSION})

set_property(TARGET ${STATIC_TARGET} PROPERTY CUDA_STANDARD 17)

################################## VFC SHARED LIBRARY #######################################

set_source_files_properties(${VFC_PLUGIN_CU_SOURCES} PROPERTIES COMPILE_FLAGS "${GENCODES} ${ENABLED_SMS}")
list(APPEND VFC_PLUGIN_SOURCES "${VFC_PLUGIN_CU_SOURCES}")

add_library(${VFC_SHARED_TARGET} SHARED ${VFC_PLUGIN_SOURCES})

target_include_directories(
    ${VFC_SHARED_TARGET}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${PROJECT_SOURCE_DIR}/common
    PUBLIC ${CUDA_INSTALL_DIR}/include
    PRIVATE ${TARGET_DIR})

set_target_properties(
    ${VFC_SHARED_TARGET}
    PROPERTIES CXX_STANDARD "17"
               CXX_STANDARD_REQUIRED "YES"
               CXX_EXTENSIONS "NO"
               ARCHIVE_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
               LIBRARY_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}")

if(MSVC)
    set_target_properties(${VFC_SHARED_TARGET} PROPERTIES LINK_FLAGS "/DEF:${VFC_PLUGIN_EXPORT_DEF}")
else()
    set_target_properties(
        ${VFC_SHARED_TARGET}
        PROPERTIES
            LINK_FLAGS
            "-Wl,--exclude-libs,ALL -Wl,-Bsymbolic -Wl,--version-script=${VFC_PLUGIN_EXPORT_MAP} -Wl,--no-undefined")
endif()

set_target_properties(${VFC_SHARED_TARGET} PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})

set_target_properties(${VFC_SHARED_TARGET} PROPERTIES VERSION ${TRT_VERSION} SOVERSION ${TRT_SOVERSION})

set_property(TARGET ${VFC_SHARED_TARGET} PROPERTY CUDA_STANDARD 17)

target_link_directories(${VFC_SHARED_TARGET} PUBLIC ${CUDA_ROOT}/lib)

target_link_libraries(${VFC_SHARED_TARGET} ${CUDART_LIB} ${${nvinfer_lib_name}_LIB_PATH} ${CMAKE_DL_LIBS})

# Needed when static linking CUDART
if(NOT MSVC)
    target_link_libraries(${VFC_SHARED_TARGET} Threads::Threads ${RT_LIB})
endif()

#########################################################################################

add_dependencies(plugin ${SHARED_TARGET} ${STATIC_TARGET} ${VFC_SHARED_TARGET})

################################### INSTALLATION ########################################

install(
    TARGETS ${SHARED_TARGET} ${VFC_SHARED_TARGET}
)

endif()
