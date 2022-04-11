#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

############################# GENERATE C++ PROTO FILES ###################################
protobuf_generate_cpp(CAFFE_PROTO_SRC CAFFE_PROTO_HDR proto/trtcaffe.proto)
add_custom_target(caffe_proto
    DEPENDS
    ${CAFFE_PROTO_SRC} ${CAFFE_PROTO_HDR}
)
############################## BUILD CAFFE PARSER ########################################
add_custom_target(nvcaffeparserlibs)

set(TARGET_NAME nvcaffeparser)
set(SHARED_TARGET ${TARGET_NAME})
set(STATIC_TARGET ${TARGET_NAME}_static)

################################# DEFINE SOURCES ########################################
include(CaffeParserSources.txt)
#########################################################################################

################################## SHARED LIBRARY #######################################

add_library(${SHARED_TARGET} SHARED
    ${CAFFE_PARSER_SRCS}
)

add_dependencies(${SHARED_TARGET} caffe_proto)

target_include_directories(${SHARED_TARGET}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE .
    PRIVATE caffeParser
    PRIVATE caffeParser/opParsers
    PRIVATE caffeWeightFactory
    PRIVATE ../common
    PRIVATE ${Protobuf_INCLUDE_DIR}
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/proto
)

set_target_properties(${SHARED_TARGET}
    PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    ARCHIVE_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
)

target_link_libraries(${SHARED_TARGET}
    ${Protobuf_LIBRARY}
    nvinfer
)

# modify google namespace to avoid namespace collision.
set(GOOGLE google_private)
target_compile_definitions(${SHARED_TARGET}
    PRIVATE
    "-Dgoogle=${GOOGLE}"
    "-DGOOGLE_PROTOBUF_ARCH_64_BIT"
)

set_target_properties(${SHARED_TARGET} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")

set_target_properties(${SHARED_TARGET} PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})

set_target_properties(${SHARED_TARGET} PROPERTIES VERSION ${TRT_VERSION} SOVERSION ${TRT_SOVERSION} )

set_property(TARGET ${SHARED_TARGET} PROPERTY CUDA_STANDARD 11)

################################## STATIC LIBRARY #######################################

add_library(${STATIC_TARGET} STATIC
    ${CAFFE_PARSER_SRCS}
)

add_dependencies(${STATIC_TARGET} caffe_proto)

target_include_directories(${STATIC_TARGET}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE .
    PRIVATE caffeParser
    PRIVATE caffeParser/opParsers
    PRIVATE caffeWeightFactory
    PRIVATE ../common
    PRIVATE ${Protobuf_INCLUDE_DIR}
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/proto
)

set_target_properties(${STATIC_TARGET}
    PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    ARCHIVE_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
)

target_link_libraries(${STATIC_TARGET}
    ${Protobuf_LIBRARY}
)

# modify google namespace to avoid namespace collision.
set(GOOGLE google_private)
target_compile_definitions(${STATIC_TARGET}
    PRIVATE
    "-Dgoogle=${GOOGLE}"
    "-DGOOGLE_PROTOBUF_ARCH_64_BIT"
)

set_target_properties(${STATIC_TARGET} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")

set_target_properties(${STATIC_TARGET} PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})

set_target_properties(${STATIC_TARGET} PROPERTIES VERSION ${TRT_VERSION} SOVERSION ${TRT_SOVERSION} )

set_property(TARGET ${STATIC_TARGET} PROPERTY CUDA_STANDARD 11)

#########################################################################################

add_dependencies(nvcaffeparserlibs ${SHARED_TARGET} ${STATIC_TARGET})

################################### INSTALLATION ########################################

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)
