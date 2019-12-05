#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
add_custom_target(plugin)

set(TARGET_NAME nvinfer_plugin)
set(SHARED_TARGET ${TARGET_NAME})
set(STATIC_TARGET ${TARGET_NAME}_static)

set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PLUGIN_EXPORT_MAP ${TARGET_DIR}/exports.map)

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

set(PLUGIN_SOURCES)
set(PLUGIN_CU_SOURCES)

set(PLUGIN_LISTS
    nmsPlugin
    normalizePlugin
    priorBoxPlugin
    reorgPlugin
    gridAnchorPlugin
    regionPlugin
    nvFasterRCNN
    batchedNMSPlugin
    flattenConcat
    cropAndResizePlugin
    proposalPlugin
    batchTilePlugin
    detectionLayerPlugin
    proposalLayerPlugin
    pyramidROIAlignPlugin
    resizeNearestPlugin
    specialSlicePlugin
    instanceNormalizationPlugin
    )

# Add BERT sources if ${BERT_GENCODES} was populated
if(BERT_GENCODES)
    set(BERT_CU_SOURCES)
    set(PLUGIN_LISTS
        ${PLUGIN_LISTS}
        embLayerNormPlugin
        fcPlugin
        geluPlugin
        bertQKVToContextPlugin
        skipLayerNormPlugin
        )
endif()

include_directories(common common/kernels ../samples/common)

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
    include_directories(${PLUGIN_ITER})
    add_subdirectory(${PLUGIN_ITER})
endforeach(PLUGIN_ITER)

# Add common
add_subdirectory(common)

# Set gencodes
set_source_files_properties(${PLUGIN_CU_SOURCES} PROPERTIES COMPILE_FLAGS ${GENCODES})
list(APPEND PLUGIN_SOURCES "${PLUGIN_CU_SOURCES}")
if (BERT_CU_SOURCES)
    set_source_files_properties(${BERT_CU_SOURCES} PROPERTIES COMPILE_FLAGS ${BERT_GENCODES})
    list(APPEND PLUGIN_SOURCES "${BERT_CU_SOURCES}")
endif()

list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/InferPlugin.cpp")
list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../samples/common/logger.cpp")

################################## SHARED LIBRARY #######################################

add_library(${SHARED_TARGET} SHARED
    ${PLUGIN_SOURCES}
)

target_include_directories(${SHARED_TARGET}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${CUB_ROOT_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/common
    PUBLIC ${CUDA_INSTALL_DIR}/include
    PRIVATE ${TARGET_DIR}
)

set_target_properties(${SHARED_TARGET} PROPERTIES
    CXX_STANDARD "11"
    CXX_STANDARD_REQUIRED "YES"
    CXX_EXTENSIONS "NO"
    ARCHIVE_OUTPUT_DIRECTORY "${TRT_BIN_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TRT_BIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${TRT_BIN_DIR}"
)

set_target_properties(${SHARED_TARGET} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL -Wl,--version-script=${PLUGIN_EXPORT_MAP} -Wl,--no-undefined")

set_target_properties(${SHARED_TARGET} PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})

set_target_properties(${SHARED_TARGET} PROPERTIES VERSION ${TRT_VERSION} SOVERSION ${TRT_SOVERSION} )

set_property(TARGET ${SHARED_TARGET} PROPERTY CUDA_STANDARD 11)

target_link_libraries(${SHARED_TARGET}
    ${CUBLAS_LIB}
    ${CUBLASLT_LIB}
    ${CUDART_LIB}
    ${CUDNN_LIB}
    nvinfer
)

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
    CXX_STANDARD "11"
    CXX_STANDARD_REQUIRED "YES"
    CXX_EXTENSIONS "NO"
    ARCHIVE_OUTPUT_DIRECTORY "${TRT_BIN_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TRT_BIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${TRT_BIN_DIR}"
)

set_target_properties(${STATIC_TARGET} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")

set_target_properties(${STATIC_TARGET} PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})

set_target_properties(${STATIC_TARGET} PROPERTIES VERSION ${TRT_VERSION} SOVERSION ${TRT_SOVERSION} )

set_property(TARGET ${STATIC_TARGET} PROPERTY CUDA_STANDARD 11)

#########################################################################################

add_dependencies(plugin ${SHARED_TARGET} ${STATIC_TARGET})

################################### INSTALLATION ########################################

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)
