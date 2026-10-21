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

cmake_minimum_required(VERSION 3.31 FATAL_ERROR)

project(TRTPyBinds LANGUAGES CXX)

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(DEFAULT_PY_EXT_PATH "${TOOLS_BASE}/win32")
else()
    set(DEFAULT_PY_EXT_PATH "/externals")
endif()
set(TRT_BUILD_PYTHON_EXTERNALS_PATH ${DEFAULT_PY_EXT_PATH} CACHE PATH "Path to the parent folder of the many versioned python headers/libs.")

set(BUILD_PYTHON_PY_VERSIONS 3.8 3.9 3.10 3.11 3.12 3.13 3.14)


set(TRT_BUILD_PYTHON_PY_VERSIONS ${BUILD_PYTHON_PY_VERSIONS} CACHE STRING "The list of python versions to build TensorRT bindings for.")

message(STATUS "TRT_BUILD_PYTHON_PY_VERSIONS: ${TRT_BUILD_PYTHON_PY_VERSIONS}")

if(NOT ${TRT_PRODUCT_IS_RTX})
    set(TRT_PYTHON_MODULE_NAMES
        "tensorrt"
        "tensorrt_lean"
        "tensorrt_dispatch")
else()
    set(TRT_PYTHON_MODULE_NAMES "tensorrt_rtx")
endif()

# The "main" tensorrt bindings depend on the parser, so if we aren't building it, we need to skip it.
if(NOT ${TRT_BUILD_ONNX_PARSER})
    message(STATUS "Not building the tensorrt python bindings as the ONNX Parser was disabled.")
    list(REMOVE_ITEM TRT_PYTHON_MODULE_NAMES "tensorrt")
endif()

set(CMAKE_CXX_STANDARD 20 CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "")
set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "")

find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED
)

# Disable automatic python detection since we need to build bindings for many python versions in one go.
set(PYBIND11_NOPYTHON ON CACHE INTERNAL "")
include(FetchPyBind11)

# Pybind11 would normally enable this by default, but does not do so under NOPYTHON mode, so we do it manually.
# Skip LTO under Bullseye coverage: covc emits slim-LTO objects whose code generation never runs at
# link time, producing empty PyInit-less binding stubs.
if(${TRT_BUILD_PLATFORM} STREQUAL ${TRT_PLATFORM_X86} AND NOT TRT_BUILD_ENABLE_BULLSEYE_COVERAGE)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

add_custom_target(tensorrt_python_bindings)

# Creates the binding library for the specified module and python version.
#
# \param moduleName The module name to create the bindings for. One of "tensorrt", "tensorrt_dispatch", or "tensorrt_lean".
# \param pyVersion  The python version to create bindings for, i.e. "3.12".
function(createBindingLibrary moduleName pyVersion)

    set(libName tensorrt_bindings_${moduleName}_${pyVersion})

    add_library(${libName} MODULE)

    # Set options unique to the "full" bindings.
    # The subdirs will use the value of TRT_PYTHON_IS_FULL_BINDINGS to set sources appropriately.
    if(${moduleName} STREQUAL "tensorrt" OR ${moduleName} STREQUAL "tensorrt_rtx")
        target_compile_definitions(${libName} PRIVATE
            tensorrt_EXPORTS=1
        )
        set(TRT_PYTHON_IS_FULL_BINDINGS ON)
    else()
        set(TRT_PYTHON_IS_FULL_BINDINGS OFF)
    endif()

    function(add_${libName}_source)
        target_sources(${libName} PRIVATE ${ARGN})
    endfunction()

    # Create an indirect refernce to the add_${libName}_source function which can be called by the subdirectories.
    # This allows each subdir to add files to the individual targets with unique binary dirs on each call.
    set(ADD_SOURCES_FUNCTION add_${libName}_source)
    set(SUBDIR_BINARY_DIR_PREFIX subbuild/${libName})
    add_subdirectory(src ${SUBDIR_BINARY_DIR_PREFIX}/src)

    target_link_libraries(${libName} PRIVATE
        pybind11::module
    )

    if(MSVC)
        target_link_libraries(${libName} PRIVATE
            pybind11::windows_extras
        )
    else()
        # This allows us to use TRT libs shipped with standalone wheels.
        set_target_properties(${libName} PROPERTIES SKIP_BUILD_RPATH ON)
        target_link_options(${libName} PRIVATE "LINKER:--rpath=$ORIGIN,--disable-new-dtags")
    endif()

    # Find the main python headers in the relevant python<ver> subfolder in the externals.
    find_path(
        PYTHON_INCLUDES Python.h
        HINTS ${TRT_BUILD_PYTHON_EXTERNALS_PATH}/python${pyVersion}
        PATH_SUFFIXES include
        NO_CACHE
        REQUIRED
        NO_CMAKE_FIND_ROOT_PATH
    )

    # Most of the headers are in that path we just found, except "pyconfig.h", which is platform-specific
    # and in a platform-specific directory with an inconsistent naming scheme.
    # So... go hunt for that. It's "mostly" located at /externals/python<ver>/include/<triple>/python<ver>/
    # Except on windows, where instead of <triple> it's just "win".
    if(${TRT_BUILD_PLATFORM} STREQUAL ${TRT_PLATFORM_X86})
        set(PYCONFIG_H_PATH "x86_64-linux-gnu/python${pyVersion}")
    elseif(${TRT_BUILD_PLATFORM} STREQUAL ${TRT_PLATFORM_AARCH64})
        set(PYCONFIG_H_PATH "aarch64-linux-gnu/python${pyVersion}")
    else()
        message(FATAL_ERROR "The current platform \"${TRT_BUILD_PLATFORM}\" cannot be used to build the TRT Python Bindings.")
    endif()

    # Also accepts the Debian multiarch layout where pyconfig.h sits one level up.
    find_path(
        PYCONFIG_INCLUDE pyconfig.h
        HINTS
            ${PYTHON_INCLUDES}/${PYCONFIG_H_PATH}
            ${PYTHON_INCLUDES}/../${PYCONFIG_H_PATH}
            ${PYTHON_INCLUDES}
        NO_CACHE
        REQUIRED
        NO_CMAKE_FIND_ROOT_PATH
    )

    # Add the python headers as SYSTEM headers to silence warnings.
    target_include_directories(${libName} SYSTEM PRIVATE
        ${PYTHON_INCLUDES}
        ${PYCONFIG_INCLUDE}
    )

    target_include_directories(${libName} PRIVATE
        "include"
        "docstrings"
    )

    # Setup links against the TRT Libraries.
    if(${moduleName} STREQUAL "tensorrt")
        set(TRT_LIBS tensorrt nvonnxparser)
        if(${TRT_BUILD_PLUGINS})
            list(APPEND TRT_LIBS tensorrt_plugins)
        elseif(TARGET nvinfer_plugin)
            # Plugins aren't built in this graph (e.g. OSS BUILD_PLUGINS=OFF); link
            # the prebuilt nvinfer_plugin so the bindings resolve plugin symbols.
            list(APPEND TRT_LIBS nvinfer_plugin)
        endif()
    elseif(${moduleName} STREQUAL "tensorrt_rtx")
        set(TRT_LIBS tensorrt nvonnxparser)
    elseif(${moduleName} STREQUAL "tensorrt_lean")
        set(TRT_LIBS tensorrt_lean_runtime)
    elseif(${moduleName} STREQUAL "tensorrt_dispatch")
        set(TRT_LIBS tensorrt_dispatch_runtime)
    else()
        message(FATAL_ERROR "Unknown TensorRT module " ${moduleName})
    endif()

    target_link_libraries(${libName} PRIVATE
        ${TRT_LIBS}
        $<COMPILE_ONLY:TRT::cudart> # We need the cuda headers to compile, but we don't link against them.
        $<COMPILE_ONLY:trt_global_definitions>
    )

    # Tell the files what module they are currently building.
    target_compile_definitions(${libName} PRIVATE
        TENSORRT_MODULE=${moduleName}
    )
    # Remove the `lib` prefix from the binding .so's and correct the output name.
    set_target_properties(${libName}
        PROPERTIES PREFIX ""
                   CXX_VISIBILITY_PRESET hidden
                   VISIBILITY_INLINES_HIDDEN ON
                   LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${moduleName}_bindings-py${pyVersion}
                   RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${moduleName}_bindings-py${pyVersion}
                   ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${moduleName}_bindings-py${pyVersion}
                   OUTPUT_NAME ${moduleName}
    )

    add_dependencies(tensorrt_python_bindings ${libName})

    # LLD can't link GCC LTO objects, so link the Python bindings with BFD instead.
    if(CMAKE_INTERPROCEDURAL_OPTIMIZATION AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set_target_properties(${libName} PROPERTIES LINKER_TYPE BFD)
    endif()

    if(MSVC)
        # Prevent pybind11 from sharing resources with other, potentially ABI incompatible modules
        # https://github.com/pybind/pybind11/issues/2898
        add_definitions(-DPYBIND11_COMPILER_TYPE="_${PROJECT_NAME}_abi")

        # The python lib is python<maj><minor>.lib, but pyVersion is <maj>.<minor>, so we need to remove the dot.
        string(REPLACE "." "" pyVerStr ${pyVersion})

        if(NOT TARGET python${pyVerStr})

            if(${TRT_BUILD_PLATFORM} STREQUAL ${TRT_PLATFORM_WIN10_X86}
            )
                set(PYTHON_EXTERNALS_TARGET_ARCH_PATH_SUFFIX "lib/x86_64")
            endif()

            # Windows needs an explicit link against the python library.
            # Standard CPython installs put the import lib under `libs/`; the bundled
            # tools-base layout uses `lib/`. Search both.
            find_library(
                PYTHON${pyVerStr}_LIBRARY_PATH python${pyVerStr}.lib
                HINTS ${TRT_BUILD_PYTHON_EXTERNALS_PATH}/python${pyVersion}
                PATH_SUFFIXES ${PYTHON_EXTERNALS_TARGET_ARCH_PATH_SUFFIX} libs lib
                REQUIRED
                NO_CMAKE_FIND_ROOT_PATH
            )

            add_library(python${pyVerStr} STATIC IMPORTED)
            set_target_properties(python${pyVerStr} PROPERTIES IMPORTED_LOCATION "${PYTHON${pyVerStr}_LIBRARY_PATH}")
        endif()

        target_link_libraries(${libName} PRIVATE python${pyVerStr})
    endif()
endfunction()

# Enumerate all the combinations and create the per-python per-module targets.
foreach(moduleName IN LISTS TRT_PYTHON_MODULE_NAMES)
    foreach(pyVersion IN LISTS TRT_BUILD_PYTHON_PY_VERSIONS)
        createBindingLibrary(${moduleName} ${pyVersion})
    endforeach()
endforeach()

# Enter the packaging subdir to actually build the wheels.
add_subdirectory(packaging)
