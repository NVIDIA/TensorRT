#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
set(ENABLED_SMS "-DENABLE_SM72 -DENABLE_SM75 -DENABLE_SM80 -DENABLE_SM86 -DENABLE_SM87 -DENABLE_SM89 -DENABLE_SM90")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ENABLED_SMS}")
set(PLUGIN_SOURCES)
set(PLUGIN_CU_SOURCES)

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
    efficientNMSPlugin/tftrt
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

# Add BERT sources if ${BERT_GENCODES} was populated
if(BERT_GENCODES)
    set(BERT_CU_SOURCES)
    set(PLUGIN_LISTS
        ${PLUGIN_LISTS}
        bertQKVToContextPlugin
        embLayerNormPlugin
        fcPlugin
        geluPlugin
        skipLayerNormPlugin
        )
    include_directories(bertQKVToContextPlugin/fused_multihead_attention/include bertQKVToContextPlugin/fused_multihead_attention_v2/include)
endif()

include_directories(common common/kernels ${CMAKE_SOURCE_DIR}/third_party)

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
    include_directories(${PLUGIN_ITER})
    add_subdirectory(${PLUGIN_ITER})
endforeach(PLUGIN_ITER)

# Add common
add_subdirectory(common)

# Set gencodes
set_source_files_properties(${PLUGIN_CU_SOURCES} PROPERTIES COMPILE_FLAGS "${GENCODES} ${ENABLED_SMS}")
list(APPEND PLUGIN_SOURCES "${PLUGIN_CU_SOURCES}")
if (BERT_CU_SOURCES)
    set_source_files_properties(${BERT_CU_SOURCES} PROPERTIES COMPILE_FLAGS "${BERT_GENCODES} ${ENABLED_SMS}")
    list(APPEND PLUGIN_SOURCES "${BERT_CU_SOURCES}")
endif()

list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/api/inferPlugin.cpp")
list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../samples/common/logger.cpp")

################################## SHARED LIBRARY #######################################

add_library(${SHARED_TARGET} SHARED
    ${PLUGIN_SOURCES}
)

target_include_directories(${SHARED_TARGET}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${PROJECT_SOURCE_DIR}/common
    PUBLIC ${CUDA_INSTALL_DIR}/include
    PRIVATE ${TARGET_DIR}
)

if (CUDA_VERSION VERSION_LESS 11.0)
    target_include_directories(${SHARED_TARGET} PUBLIC ${CUB_ROOT_DIR})
endif()

set_target_properties(${SHARED_TARGET} PROPERTIES
    CXX_STANDARD "14"
    CXX_STANDARD_REQUIRED "YES"
    CXX_EXTENSIONS "NO"
    ARCHIVE_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
)

if (MSVC)
    set_target_properties(${SHARED_TARGET} PROPERTIES LINK_FLAGS "/DEF:${PLUGIN_EXPORT_DEF}")
else()
    set_target_properties(${SHARED_TARGET} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL -Wl,-Bsymbolic -Wl,--version-script=${PLUGIN_EXPORT_MAP} -Wl,--no-undefined")
endif()

set_target_properties(${SHARED_TARGET} PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})

set_target_properties(${SHARED_TARGET} PROPERTIES VERSION ${TRT_VERSION} SOVERSION ${TRT_SOVERSION} )

set_property(TARGET ${SHARED_TARGET} PROPERTY CUDA_STANDARD 14)

target_link_libraries(${SHARED_TARGET}
    ${CUDART_LIB}
    ${${nvinfer_lib_name}_LIB_PATH}
    ${CMAKE_DL_LIBS}
)

# Needed when static linking CUDART
if (NOT MSVC)
    target_link_libraries(${SHARED_TARGET} Threads::Threads ${RT_LIB})
endif()

################################## STATIC LIBRARY #######################################

add_library(${STATIC_TARGET} STATIC
    ${PLUGIN_SOURCES}
)

target_include_directories(${STATIC_TARGET}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${CUB_ROOT_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/common
    PUBLIC ${CUDA_INSTALL_DIR}/include
    PRIVATE ${TARGET_DIR}
)

set_target_properties(${STATIC_TARGET} PROPERTIES
    CXX_STANDARD "14"
    CXX_STANDARD_REQUIRED "YES"
    CXX_EXTENSIONS "NO"
    ARCHIVE_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
)

set_target_properties(${STATIC_TARGET} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")

set_target_properties(${STATIC_TARGET} PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})

set_target_properties(${STATIC_TARGET} PROPERTIES VERSION ${TRT_VERSION} SOVERSION ${TRT_SOVERSION} )

set_property(TARGET ${STATIC_TARGET} PROPERTY CUDA_STANDARD 14)

################################## VFC SHARED LIBRARY #######################################

set_source_files_properties(${VFC_PLUGIN_CU_SOURCES} PROPERTIES COMPILE_FLAGS "${GENCODES} ${ENABLED_SMS}")
list(APPEND VFC_PLUGIN_SOURCES "${VFC_PLUGIN_CU_SOURCES}")

add_library(${VFC_SHARED_TARGET} SHARED
    ${VFC_PLUGIN_SOURCES}
)

target_include_directories(${VFC_SHARED_TARGET}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${PROJECT_SOURCE_DIR}/common
    PUBLIC ${CUDA_INSTALL_DIR}/include
    PRIVATE ${TARGET_DIR}
)

set_target_properties(${VFC_SHARED_TARGET} PROPERTIES
    CXX_STANDARD "14"
    CXX_STANDARD_REQUIRED "YES"
    CXX_EXTENSIONS "NO"
    ARCHIVE_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
)

if (MSVC)
    set_target_properties(${VFC_SHARED_TARGET} PROPERTIES LINK_FLAGS "/DEF:${VFC_PLUGIN_EXPORT_DEF}")
else()
    set_target_properties(${VFC_SHARED_TARGET} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL -Wl,-Bsymbolic -Wl,--version-script=${VFC_PLUGIN_EXPORT_MAP} -Wl,--no-undefined")
endif()

set_target_properties(${VFC_SHARED_TARGET} PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})

set_target_properties(${VFC_SHARED_TARGET} PROPERTIES VERSION ${TRT_VERSION} SOVERSION ${TRT_SOVERSION} )

set_property(TARGET ${VFC_SHARED_TARGET} PROPERTY CUDA_STANDARD 14)

target_link_libraries(${VFC_SHARED_TARGET}
    ${CUDART_LIB}
    ${${nvinfer_lib_name}_LIB_PATH}
    ${CMAKE_DL_LIBS}
)

# Needed when static linking CUDART
if (NOT MSVC)
    target_link_libraries(${VFC_SHARED_TARGET} Threads::Threads ${RT_LIB})
endif()

#########################################################################################

add_dependencies(plugin ${SHARED_TARGET} ${STATIC_TARGET} ${VFC_SHARED_TARGET})

################################### INSTALLATION ########################################

install(TARGETS ${SHARED_TARGET} ${VFC_SHARED_TARGET}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)
