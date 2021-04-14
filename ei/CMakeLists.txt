#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

set_ifndef(PLUGINS_NEEDED OFF)

set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})

get_filename_component(SAMPLES_DIR ../samples ABSOLUTE)
get_filename_component(SAMPLE_DIR_NAME ${TARGET_DIR} NAME)

set_ifndef(CUDA_INSTALL_DIR /usr/local/cuda)

# SAMPLES_COMMON_SOURCES
set(SAMPLES_COMMON_SOURCES
    ${SAMPLES_DIR}/common/logger.cpp
)

set(SAMPLE_NAME ei)# CACHE STRING "binary name of the sample")

set(TARGET_NAME ${SAMPLE_NAME})

add_library(${TARGET_NAME} 
    ${CMAKE_CURRENT_LIST_DIR}/libeitrt.cpp
    ${SAMPLES_COMMON_SOURCES}
)

# debug
set(CMAKE_BUILD_TYPE Debug)

set(DEPS_LIST "")

if(BUILD_PLUGINS)
    list(APPEND DEPS_LIST nvinfer_plugin)
endif()

if(BUILD_PARSERS)
    list(APPEND DEPS_LIST nvuffparser nvcaffeparser nvonnxparser)
endif()

if(BUILD_PLUGINS OR BUILD_PARSERS)
    add_dependencies(${TARGET_NAME}
        ${DEPS_LIST}
    )
endif()

set(ONNX_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/parsers/onnx CACHE STRING "ONNX include directory")

target_include_directories(${TARGET_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${ONNX_INCLUDE_DIR}
    PUBLIC ${CUDA_INSTALL_DIR}/include
    PRIVATE ${SAMPLES_DIR}/common
    PRIVATE ${TARGET_DIR}
)

target_compile_options(${TARGET_NAME} PUBLIC "-fno-rtti")

set(SAMPLE_DEP_LIBS
    ${CUDART_LIB}
    ${CUBLAS_LIB}
    ${CUDNN_LIB}
    nvinfer
    ${RT_LIB}
    ${CMAKE_DL_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
)

if(${PLUGINS_NEEDED})
    list(APPEND SAMPLE_DEP_LIBS nvinfer_plugin)
endif()

list(APPEND SAMPLE_DEP_LIBS nvonnxparser)

# Necessary to link nvinfer_plugin library.
target_link_libraries(${TARGET_NAME}
    ${SAMPLE_DEP_LIBS}
    -Wl,--unresolved-symbols=ignore-in-shared-libs
)


set_target_properties(${TARGET_NAME} PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})

set_target_properties(${TARGET_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
)


################################### INSTALLATION ########################################

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

##################################### SUMMARY ###########################################

get_filename_component(LICENSE_STATUS ../ ABSOLUTE)
get_filename_component(LICENSE_STATUS "${LICENSE_STATUS}" NAME)

message(STATUS "Adding new sample: ${TARGET_NAME}")
message(STATUS "    - InferPlugin Used: ${PLUGINS_NEEDED}")
