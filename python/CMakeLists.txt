#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

project(TRTPyBinds LANGUAGES CXX)

option(TRT_BUILD_ENABLE_NEW_PYTHON_FLOW "Use new build logic based on the main CMake build." OFF)

if (${TRT_BUILD_ENABLE_NEW_PYTHON_FLOW})

if(MSVC)
    set(DEFAULT_PY_EXT_PATH "${TOOLS_BASE}/win32")
else()
    set(DEFAULT_PY_EXT_PATH "/externals")
endif()
set(TRT_BUILD_PYTHON_EXTERNALS_PATH ${DEFAULT_PY_EXT_PATH} CACHE PATH "Path to the parent folder of pybind11 and the many versioned python headers/libs.")

set(TRT_BUILD_PYTHON_PY_VERSIONS 3.8 3.9 3.10 3.11 3.12 3.13 CACHE STRING "The list of python versions to build bindings for.")


if(NOT ${TRT_BUILD_WINML})
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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED
)

# Disable automatic python detection since we need to build bindings for many python versions in one go.
set(PYBIND11_NOPYTHON ON CACHE INTERNAL "")
add_subdirectory(${TRT_BUILD_PYTHON_EXTERNALS_PATH}/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/externals/pybind11)

# Pybind11 would normally enable this by default, but does not do so under NOPYTHON mode, so we do it manually.
if(${TRT_BUILD_PLATFORM} STREQUAL ${TRT_PLATFORM_X86})
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
        pybind11::headers
        pybind11::opt_size
    )

    if(MSVC)
        target_link_libraries(${libName} PRIVATE
            pybind11::windows_extras
        )
    else()
        # This allows us to use TRT libs shipped with standalone wheels.
        target_link_options(${libName} PRIVATE "LINKER:-rpath=$ORIGIN:$ORIGIN/../${TENSORRT_MODULE}_libs)")
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
    elseif(${TRT_BUILD_PLATFORM} STREQUAL ${TRT_PLATFORM_WIN10})
        set(PYCONFIG_H_PATH "win/python${pyVersion}")
    else()
        message(FATAL_ERROR "The current platform \"${TRT_BUILD_PLATFORM}\" cannot be used to build the TRT Python Bindings.")
    endif()

    find_path(
        PYCONFIG_INCLUDE pyconfig.h
        HINTS ${PYTHON_INCLUDES}/${PYCONFIG_H_PATH}
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

    if(MSVC)
        # Prevent pybind11 from sharing resources with other, potentially ABI incompatible modules
        # https://github.com/pybind/pybind11/issues/2898
        add_definitions(-DPYBIND11_COMPILER_TYPE="_${PROJECT_NAME}_abi")

        # The python lib is python<maj><minor>.lib, but pyVersion is <maj>.<minor>, so we need to remove the dot.
        string(REPLACE "." "" pyVerStr ${pyVersion})

        if(NOT TARGET python${pyVerStr})
            # Windows needs an explicit link against the python library.
            find_library(
                PYTHON${pyVerStr}_LIBRARY_PATH python${pyVerStr}.lib
                HINTS ${TRT_BUILD_PYTHON_EXTERNALS_PATH}/python${pyVersion}
                PATH_SUFFIXES lib
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

else() # TRT_BUILD_ENABLE_NEW_PYTHON_FLOW - old flow is below this line

set(TRT_BUILD_WINML OFF)
if (${TENSORRT_MODULE} STREQUAL "tensorrt_rtx")
    # The "old" flow doesn't already have TRT_BUILD_WINML set, because it's
    # coming from the legacy make build. So if we get the tensorrt_rtx module,
    # we assume we're in the TensorRT-RTX build.
    set(TRT_BUILD_WINML ON)
endif()

# Ensure TRT_WINML is either 0 or 1
if (NOT TRT_WINML)
    set(TRT_WINML 0)
    if (TRT_BUILD_WINML)
        set(TRT_WINML 1)
    endif()
endif()

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if(NOT DEFINED ${var})
        set(${var} ${val})
    endif()
endmacro()

function(message)
    if(VERBOSE)
        _message(${ARGN})
    endif()
endfunction()

# -------- CMAKE OPTIONS --------

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${TENSORRT_MODULE}/)
set(CPP_STANDARD
    17
    CACHE STRING "CPP Standard Version")
set(CMAKE_CXX_STANDARD ${CPP_STANDARD})

if(NOT MSVC)
    # This allows us to use TRT libs shipped with standalone wheels.
    set(CMAKE_SHARED_LINKER_FLAGS -Wl,-rpath=$ORIGIN:$ORIGIN/../${TENSORRT_MODULE}_libs)
endif()

# -------- PATHS --------
message(STATUS "EXT_PATH: ${EXT_PATH}")
message(STATUS "TENSORRT_BUILD: ${TENSORRT_BUILD}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "CUDA_ROOT: ${CUDA_ROOT}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set_ifndef(TENSORRT_ROOT ../)
message(STATUS "TENSORRT_ROOT: ${TENSORRT_ROOT}")

set_ifndef(WIN_EXTERNALS ${EXT_PATH})
message(STATUS "WIN_EXTERNALS: ${WIN_EXTERNALS}")

# Convert to an absolute path.
set_ifndef(ONNX_INC_DIR ${TENSORRT_ROOT}/parsers/onnx)
find_path(
    PYBIND11_DIR pybind11/pybind11.h
    HINTS ${EXT_PATH} ${WIN_EXTERNALS}
    PATH_SUFFIXES pybind11/include
    REQUIRED
)

message(STATUS "ONNX_INC_DIR: ${ONNX_INC_DIR}")
message(STATUS "PYBIND11_DIR: ${PYBIND11_DIR}")

# Source Files
if(${TENSORRT_MODULE} STREQUAL "tensorrt" OR ${TENSORRT_MODULE} STREQUAL "tensorrt_rtx")
    # tensorrt full dependencies
    file(GLOB_RECURSE SOURCE_FILES src/*.cpp)
    set(_NEED_EXPORTS_DEF ON)
else()
    # tensorrt_lean and tensorrt_dispatch dependencies
    set(SOURCE_FILES src/pyTensorRT.cpp src/utils.cpp src/infer/pyCore.cpp src/infer/pyPlugin.cpp
                     src/infer/pyFoundationalTypes.cpp)
endif()

set(PYTHON python${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION})
set(PYTHON_LIB_NAME python${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION})
message(STATUS "PYTHON: ${PYTHON}")
message(STATUS "TENSORRT_MODULE: ${TENSORRT_MODULE}")

set(PY_MODULE_NAME ${TENSORRT_MODULE})

# Find headers
set(LIBPATH_SUFFIX lib)
if(MSVC)
    find_path(
        PY_INCLUDE Python.h
        HINTS ${WIN_EXTERNALS}/${PYTHON} ${EXT_PATH}/${PYTHON}
        PATH_SUFFIXES include
        REQUIRED
    )
    find_path(
        PY_LIB_DIR ${PYTHON_LIB_NAME}.lib
        HINTS ${WIN_EXTERNALS}/${PYTHON} ${EXT_PATH}/${PYTHON}
        PATH_SUFFIXES ${LIBPATH_SUFFIX}
        REQUIRED
    )
    message(STATUS "PY_LIB_DIR: ${PY_LIB_DIR}")
else()
    find_path(
        PY_INCLUDE Python.h
        HINTS ${EXT_PATH}/${PYTHON} /usr/include/${PYTHON}
        PATH_SUFFIXES include
        REQUIRED
    )
endif()

message(STATUS "PY_INCLUDE: ${PY_INCLUDE}")

if(MSVC)
    set(PY_TARGET_DIR win)
else()
    set(PY_TARGET_DIR ${TARGET}-linux-gnu)
    if(${TARGET} STREQUAL ppc64le)
        set(PY_TARGET_DIR powerpc64le-linux-gnu)
    endif()
endif()

# The per-platform pyconfig.h is located at /externals/python<ver>/<triple>/python<ver>/.
# Not sure why it's setup that way.
find_path(
    PY_CONFIG_INCLUDE pyconfig.h
    HINTS ${PY_INCLUDE}
    PATH_SUFFIXES ${PY_TARGET_DIR}/${PYTHON} ${PY_TARGET_DIR}/${PYTHON}m
    REQUIRED
)
message(STATUS "PY_CONFIG_INCLUDE: ${PY_CONFIG_INCLUDE}")

# -------- GLOBAL COMPILE OPTIONS --------

include_directories(${TENSORRT_ROOT}/include ${PROJECT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/docstrings ${ONNX_INC_DIR} ${PYBIND11_DIR})
link_directories(${TENSORRT_BUILD} ${TENSORRT_LIBPATH})

if(MSVC)
    # Prevent pybind11 from sharing resources with other, potentially ABI incompatible modules
    # https://github.com/pybind/pybind11/issues/2898
    add_definitions(-DPYBIND11_COMPILER_TYPE="_${PROJECT_NAME}_abi")
endif()

if(MSVC)
    include_directories(SYSTEM BEFORE ${ADDITIONAL_PLATFORM_INCLUDE_DIRS})
    link_libraries(${ADDITIONAL_PLATFORM_LIB_FLAGS})
    link_directories(${PY_LIB_DIR})

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
    if(${NV_GEN_PDB})
        # PDB is only useful in release mode.
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi /bigobj")
        set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
    endif()
else()
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} ${GLIBCXX_USE_CXX11_ABI_FLAG} -fvisibility=hidden -std=c++${CPP_STANDARD} -Wno-deprecated-declarations"
    )
endif()


# ---------- MODULE DEPENDENCIES ----------
add_compile_definitions(TENSORRT_MODULE=${TENSORRT_MODULE})

if("${TRT_VCAST}" STREQUAL "1" OR "${TRT_VCAST_SAFE}" STREQUAL "1")
    set(vfc_suffix "_static")
else()
    set(vfc_suffix "")
endif()


if(MSVC)
    set(nvinfer_lib_name "${TRT_NVINFER_NAME}_${TENSORRT_MAJOR_VERSION}${TRT_LIB_SUFFIX}")
    set(nvinfer_plugin_lib_name "nvinfer_plugin_${TENSORRT_MAJOR_VERSION}")
    set(nvonnxparser_lib_name "${TRT_ONNXPARSER_NAME}_${TENSORRT_MAJOR_VERSION}${TRT_LIB_SUFFIX}")
    set(nvinfer_lean_lib_name "nvinfer_lean_${TENSORRT_MAJOR_VERSION}${vfc_suffix}")
    set(nvinfer_dispatch_lib_name "nvinfer_dispatch_${TENSORRT_MAJOR_VERSION}${vfc_suffix}")
else()
    if(${TRT_BUILD_WINML})
        set(nvinfer_lib_name "-l:lib${TRT_NVINFER_NAME}.so")
    else()
        set(nvinfer_lib_name "${TRT_NVINFER_NAME}")
    endif()
    set(nvinfer_plugin_lib_name "nvinfer_plugin")
    set(nvonnxparser_lib_name "${TRT_ONNXPARSER_NAME}")
    set(nvinfer_lean_lib_name "nvinfer_lean${vfc_suffix}")
    set(nvinfer_dispatch_lib_name "nvinfer_dispatch${vfc_suffix}")
endif()
message(STATUS "NVInferLib = ${nvinfer_lib_name}")
message(STATUS "NVOnnxParserLib = ${nvonnx_parser_lib_name}")

if(${TENSORRT_MODULE} STREQUAL "tensorrt" OR ${TENSORRT_MODULE} STREQUAL "tensorrt_rtx")
    set(TRT_LIBS ${nvinfer_lib_name} ${nvonnxparser_lib_name})
        list(APPEND TRT_LIBS ${nvinfer_plugin_lib_name})
elseif(${TENSORRT_MODULE} STREQUAL "tensorrt_lean")
    set(TRT_LIBS ${nvinfer_lean_lib_name})
elseif(${TENSORRT_MODULE} STREQUAL "tensorrt_dispatch")
    set(TRT_LIBS ${nvinfer_dispatch_lib_name})
else()
    message(FATAL_ERROR "Unknown TensorRT module " ${TENSORRT_MODULE})
endif()

# -------- BUILDING --------
set(LIB_NAME ${PY_MODULE_NAME})

# Set up target
add_library(${LIB_NAME} SHARED ${SOURCE_FILES})
if(_NEED_EXPORTS_DEF)
    target_compile_definitions(${LIB_NAME} PRIVATE
        tensorrt_EXPORTS=1
    )
endif()
message(STATUS "Python library name ${LIB_NAME}")
target_include_directories(${LIB_NAME} BEFORE PUBLIC ${PY_CONFIG_INCLUDE} ${PY_INCLUDE})
if(MSVC)
    # For some reason, we must explicitly link against the Python library on Windows.
    message(STATUS "Dependencies ${TRT_LIBS} ${PYTHON_LIB_NAME}")
    target_link_libraries(${LIB_NAME} PRIVATE ${TRT_LIBS} ${PYTHON_LIB_NAME})
else()
    message(STATUS "Dependencies ${TRT_LIBS} ")
    target_link_libraries(${LIB_NAME} PRIVATE ${TRT_LIBS})
endif()

# Note that we have to remove the `lib` prefix from the binding .so's
set_target_properties(${LIB_NAME} PROPERTIES PREFIX "")

endif()
