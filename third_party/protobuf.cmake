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

include(ExternalProject)
# Downloads and builds a given protobuf version, generating a protobuf target
# with the include dir and binaries imported
macro(configure_protobuf VERSION)
    set(protobufPackage "protobuf-cpp-${VERSION}.tar.gz")
    set(Protobuf_PKG_URL "https://github.com/google/protobuf/releases/download/v${VERSION}/${protobufPackage}")
    set(Protobuf_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR})
    set(Protobuf_TARGET third_party.protobuf)

    set(PROTOBUF_CFLAGS "-Dgoogle=google_private")
    set(PROTOBUF_CXXFLAGS "-Dgoogle=google_private")

    ExternalProject_Add(${Protobuf_TARGET}
        PREFIX ${Protobuf_TARGET}
        URL ${Protobuf_PKG_URL}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ${CMAKE_COMMAND} ${Protobuf_INSTALL_DIR}/${Protobuf_TARGET}/src/${Protobuf_TARGET}/cmake
            -G${CMAKE_GENERATOR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc
            -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++
            -DCMAKE_C_FLAGS=${PROTOBUF_CFLAGS}
            -DCMAKE_CXX_FLAGS=${PROTOBUF_CXXFLAGS}
            -DCMAKE_INSTALL_PREFIX=${Protobuf_INSTALL_DIR}/${Protobuf_TARGET}
            -Dprotobuf_BUILD_TESTS=OFF
        SOURCE_SUBDIR cmake
        BINARY_DIR ${Protobuf_INSTALL_DIR}/${Protobuf_TARGET}/src/${Protobuf_TARGET}
    )

    set(Protobuf_BIN_DIR "${CMAKE_BINARY_DIR}/${Protobuf_TARGET}/bin")
    find_file (CENTOS_FOUND centos-release PATHS /etc)
    if (CENTOS_FOUND)
        set(Protobuf_LIB_DIR "${CMAKE_BINARY_DIR}/${Protobuf_TARGET}/lib64")
    else (CENTOS_FOUND)
        set(Protobuf_LIB_DIR "${CMAKE_BINARY_DIR}/${Protobuf_TARGET}/lib")
    endif (CENTOS_FOUND)
    set(Protobuf_INCLUDE_DIR "${CMAKE_BINARY_DIR}/${Protobuf_TARGET}/include")
    set(Protobuf_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/${Protobuf_TARGET}/include")
    set(Protobuf_PROTOC_EXECUTABLE  "${Protobuf_BIN_DIR}/protoc")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(Protobuf_LIBRARY "${Protobuf_LIB_DIR}/libprotobufd.a")
        set(Protobuf_PROTOC_LIBRARY "${Protobuf_LIB_DIR}/libprotocd.a")
        set(Protobuf_LITE_LIBRARY "${Protobuf_LIB_DIR}/libprotobuf-lited.a")
    else()
        set(Protobuf_LIBRARY "${Protobuf_LIB_DIR}/libprotobuf.a")
        set(Protobuf_PROTOC_LIBRARY "${Protobuf_LIB_DIR}/libprotoc.a")
        set(Protobuf_LITE_LIBRARY "${Protobuf_LIB_DIR}/libprotobuf-lite.a")
    endif()
    set(protolibType STATIC)

    if ((${CMAKE_SYSTEM_NAME} STREQUAL "Linux") AND NOT(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64"))
        message(STATUS "Setting up another Protobuf build for cross compilation targeting ${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}")
        # In case of cross-compilation for QNX requires additional CXX flags
        if(${CMAKE_SYSTEM_NAME} STREQUAL "qnx")
            message("Conigure compilation flags for qnx")
            set(PROTOBUF_CXXFLAGS "-D__EXT_POSIX1_198808 -D_POSIX_C_SOURCE=200112L -D_QNX_SOURCE -D_FILE_OFFSET_BITS=64 ${PROTOBUF_CXXFLAGS}")
        endif()
        ExternalProject_Add(${Protobuf_TARGET}_${CMAKE_SYSTEM_PROCESSOR}
            PREFIX ${Protobuf_TARGET}_${CMAKE_SYSTEM_PROCESSOR}
            URL ${Protobuf_PKG_URL}
            UPDATE_COMMAND ""
            CONFIGURE_COMMAND ${CMAKE_COMMAND} ${Protobuf_INSTALL_DIR}/${Protobuf_TARGET}_${CMAKE_SYSTEM_PROCESSOR}/src/${Protobuf_TARGET}_${CMAKE_SYSTEM_PROCESSOR}/cmake
                -G${CMAKE_GENERATOR}
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
                -DCMAKE_C_FLAGS=${PROTOBUF_CFLAGS}
                -DCMAKE_CXX_FLAGS=${PROTOBUF_CXXFLAGS}
                -DCMAKE_INSTALL_PREFIX=${Protobuf_INSTALL_DIR}/${Protobuf_TARGET}_${CMAKE_SYSTEM_PROCESSOR}
                -Dprotobuf_BUILD_TESTS=OFF
            SOURCE_SUBDIR cmake
            BINARY_DIR ${Protobuf_INSTALL_DIR}/${Protobuf_TARGET}_${CMAKE_SYSTEM_PROCESSOR}/src/${Protobuf_TARGET}_${CMAKE_SYSTEM_PROCESSOR}/
        )

        set(Protobuf_LIB_DIR "${CMAKE_BINARY_DIR}/${Protobuf_TARGET}_${CMAKE_SYSTEM_PROCESSOR}/lib")
        set(Protobuf_INCLUDE_DIR "${CMAKE_BINARY_DIR}/${Protobuf_TARGET}/include")
        set(Protobuf_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/${Protobuf_TARGET}/include")
        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(Protobuf_LIBRARY "${Protobuf_LIB_DIR}/libprotobufd.a")
            set(Protobuf_PROTOC_LIBRARY "${Protobuf_LIB_DIR}/libprotocd.a")
            set(Protobuf_LITE_LIBRARY "${Protobuf_LIB_DIR}/libprotobuf-lited.a")
        else()
            set(Protobuf_LIBRARY "${Protobuf_LIB_DIR}/libprotobuf.a")
            set(Protobuf_PROTOC_LIBRARY "${Protobuf_LIB_DIR}/libprotoc.a")
            set(Protobuf_LITE_LIBRARY "${Protobuf_LIB_DIR}/libprotobuf-lite.a")
        endif()
        set(Protobuf_INSTALL_DIR "${CMAKE_BINARY_DIR}/${Protobuf_TARGET}")
        set(protolibType STATIC)
    endif()

    add_library(protobuf::libprotobuf ${protolibType} IMPORTED)
    set_target_properties(protobuf::libprotobuf PROPERTIES
        IMPORTED_LOCATION "${Protobuf_LIBRARY}"
    )

    add_library(protobuf::libprotobuf-lite ${protolibType} IMPORTED)
    set_target_properties(protobuf::libprotobuf-lite PROPERTIES
        IMPORTED_LOCATION "${Protobuf_LITE_LIBRARY}"
    )
    if ((${CMAKE_SYSTEM_NAME} STREQUAL "Linux") AND NOT(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64"))
        add_dependencies(protobuf::libprotobuf ${Protobuf_TARGET}_${CMAKE_SYSTEM_PROCESSOR})
        add_dependencies(protobuf::libprotobuf-lite ${Protobuf_TARGET}_${CMAKE_SYSTEM_PROCESSOR})
    else ()
        add_dependencies(protobuf::libprotobuf ${Protobuf_TARGET}_${CMAKE_SYSTEM_PROCESSOR})
        add_dependencies(protobuf::libprotobuf-lite ${Protobuf_TARGET})
    endif()

    add_library(protobuf::libprotoc ${protolibType} IMPORTED)
    add_dependencies(protobuf::libprotoc ${Protobuf_TARGET})
    set_target_properties(protobuf::libprotoc PROPERTIES
        IMPORTED_LOCATION "${Protobuf_PROTOC_LIBRARY}"
    )

    add_executable(protobuf::protoc IMPORTED)
    add_dependencies(protobuf::protoc ${Protobuf_TARGET})
    set_target_properties(protobuf::protoc PROPERTIES
        IMPORTED_LOCATION "${Protobuf_PROTOC_EXECUTABLE}"
    )

    add_library(Protobuf INTERFACE)
    target_include_directories(Protobuf INTERFACE "${Protobuf_INCLUDE_DIR}")
    target_link_libraries(Protobuf INTERFACE protobuf::libprotobuf)
    message(STATUS "Using libprotobuf ${Protobuf_LIBRARY}")
endmacro()

macro(configure_protobuf_internal VERSION)
    #Assuming building on a x86_64 Linux Box so use local protoc even for cross compiliation
    set(Protobuf_BIN_DIR "${Protobuf_DIR}/../x86_64/${VERSION}/bin")
    set(Protobuf_LIB_DIR "${Protobuf_DIR}/${VERSION}/lib")
    set(Protobuf_PROTOC_EXECUTABLE  "${Protobuf_BIN_DIR}/protoc")
    set(Protobuf_LIBRARY "${Protobuf_LIB_DIR}/libprotobuf.a")
    set(Protobuf_PROTOC_LIBRARY "${Protobuf_LIB_DIR}/libprotoc.a")
    set(Protobuf_LITE_LIBRARY "${Protobuf_LIB_DIR}/libprotobuf-lite.a")
    set(Protobuf_INCLUDE_DIR "${Protobuf_DIR}/${VERSION}/include")
    set(Protobuf_INCLUDE_DIRS "${Protobuf_DIR}/${VERSION}/include")
    set(Protobuf_INSTALL_DIR "${Protobuf_DIR}/${VERSION}")
    set(protolibType STATIC)

    add_library(protobuf::libprotobuf ${protolibType} IMPORTED)
    add_dependencies(protobuf::libprotobuf ${Protobuf_DIR})
    set_target_properties(protobuf::libprotobuf PROPERTIES
        IMPORTED_LOCATION "${Protobuf_LIBRARY}"
    )

    add_library(protobuf::libprotobuf-lite ${protolibType} IMPORTED)
    add_dependencies(protobuf::libprotobuf-lite ${Protobuf_DIR})
    set_target_properties(protobuf::libprotobuf-lite PROPERTIES
        IMPORTED_LOCATION "${Protobuf_LITE_LIBRARY}"
    )

    add_library(protobuf::libprotoc ${protolibType} IMPORTED)
    add_dependencies(protobuf::libprotoc ${Protobuf_DIR})
    set_target_properties(protobuf::libprotoc PROPERTIES
        IMPORTED_LOCATION "${Protobuf_PROTOC_LIBRARY}"
    )

    add_executable(protobuf::protoc IMPORTED)
    add_dependencies(protobuf::protoc ${Protobuf_DIR})
    set_target_properties(protobuf::protoc PROPERTIES
    IMPORTED_LOCATION "${Protobuf_PROTOC_EXECUTABLE}"
    )

    add_library(Protobuf INTERFACE)
    target_include_directories(Protobuf INTERFACE "${Protobuf_INCLUDE_DIR}")
    target_link_libraries(Protobuf INTERFACE protobuf::libprotobuf)
    message(STATUS "Using libprotobuf ${Protobuf_LIBRARY}")
endmacro()

function(protobuf_generate_cpp SRCS HDRS)
    set(PROTOS ${ARGN})

    foreach(proto ${PROTOS})
        get_filename_component(PROTO_NAME "${proto}" NAME_WE)
        get_filename_component(PROTO_DIR "${proto}" DIRECTORY)

        set(PROTO_HEADER "${PROTO_NAME}.pb.h")
        set(PROTO_SRC    "${PROTO_NAME}.pb.cc")

        message(STATUS "Protobuf ${proto} -> ${PROTO_DIR}/${PROTO_SRC} ${PROTO_DIR}/${PROTO_HEADER}")

        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PROTO_DIR})

        message(STATUS ${CMAKE_CURRENT_BINARY_DIR})

        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_DIR}/${PROTO_SRC}"
                   "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_DIR}/${PROTO_HEADER}"
            COMMAND LIBRARY_PATH=${Protobuf_LIB_DIR} ${Protobuf_PROTOC_EXECUTABLE}
            ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR}/${PROTO_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/${proto}
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${proto}" protobuf::libprotobuf Protobuf protobuf::protoc
            COMMENT "${proto} -> ${PROTO_DIR}/${PROTO_SRC} ${PROTO_DIR}/${PROTO_HEADER}"
        )

        list(APPEND SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_DIR}/${PROTO_SRC}")
        list(APPEND HEADERS "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_DIR}/${PROTO_HEADER}")
    endforeach()
    set(${SRCS} ${SOURCES} PARENT_SCOPE)
    set(${HDRS} ${HEADERS} PARENT_SCOPE)
endfunction()
