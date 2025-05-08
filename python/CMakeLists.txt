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

if(MSVC)
    set(DEFAULT_PY_EXT_PATH "${TOOLS_BASE}/win32")
else()
    set(DEFAULT_PY_EXT_PATH "/externals")
endif()
set(TRT_BUILD_PYTHON_EXTERNALS_PATH ${DEFAULT_PY_EXT_PATH} CACHE PATH "Path to the parent folder of pybind11 and the many versioned python headers/libs.")

set(TRT_BUILD_PYTHON_PY_VERSIONS 3.8 3.9 3.10 3.11 3.12 3.13 CACHE STRING "The list of python versions to build bindings for.")

set(TRT_PYTHON_MODULE_NAMES
    "tensorrt"
    "tensorrt_lean"
    "tensorrt_dispatch"
)

if (${TRT_BUILD_ENABLE_NEW_PYTHON_FLOW})

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
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

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
    if(${moduleName} STREQUAL "tensorrt")
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
    set(SUBDIR_BINARY_DIR_PREFIX ${libName})
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
        set(TRT_LIBS tensorrt nvonnxparser tensorrt_plugins)
    elseif(${moduleName} STREQUAL "tensorrt_lean")
        set(TRT_LIBS tensorrt_lean_runtime)
    elseif(${moduleName} STREQUAL "tensorrt_dispatch")
        set(TRT_LIBS tensorrt_dispatch_runtime)
    else()
        message(FATAL_ERROR "Unknown TensorRT module " ${moduleName})
    endif()

    find_package(CUDAToolkit REQUIRED)

    target_link_libraries(${libName} PRIVATE
        ${TRT_LIBS}
        $<COMPILE_ONLY:CUDA::cudart_static>
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
                   OUTPUT_NAME ${moduleName}
    )

    add_dependencies(tensorrt_python_bindings ${libName})

    # Setup some final MSVC-exclusive settings
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

# Processes one or more wheel templates file, replacing any markers with concrete information and
# copying the result into the per-python per-module build dir.
#
# \param moduleName The module name to create the bindings for. One of "tensorrt", "tensorrt_dispatch", or "tensorrt_lean".
# \param pyVersion  The python version to create bindings for, i.e. "3.12".
# \param ARGN       A list of file paths, relative to './packaging/bindings_wheel/tensorrt', of the file(s) to copy.
# \returns generatedWheelFiles A list containing paths to all generated files, which can be used to create a custom target.
function(processWheelTemplates moduleName pyVersion)
    foreach(filePath IN LISTS ARGN)
        set(outputDir ${CMAKE_CURRENT_BINARY_DIR}/${moduleName}_bindings-py${pyVersion}/$<CONFIG>)
        set(outputFile ${outputDir}/${filePath})

        add_custom_command(
            OUTPUT ${outputFile}
            COMMAND
                ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/scripts/process_wheel_template.py
                --src-dir ${CMAKE_CURRENT_LIST_DIR}/packaging/bindings_wheel/tensorrt
                --dst-dir ${outputDir}
                --filepath ${filePath}
                --trt-module ${moduleName}
                --trt-py-version ${TensorRT_VERSION}
                --cuda-version ${TRT_CUDA_VERSION}
                --trt-version ${TensorRT_VERSION}
            DEPENDS
                scripts/process_wheel_template.py
                ${CMAKE_CURRENT_LIST_DIR}/packaging/bindings_wheel/tensorrt/${filePath}
            VERBATIM)

        list(APPEND generatedFiles ${outputFile})
    endforeach()
    set(generatedWheelFiles ${generatedFiles} PARENT_SCOPE)
endfunction()

add_custom_target(trt_python_wheel_files)

# Expands all template files for the bindings for the target module and python version.
# Each call to this function creates a new custom target, and makes trt_python_wheel_files depend on the new target.
#
# \param moduleName The module name to create the bindings for. One of "tensorrt", "tensorrt_dispatch", or "tensorrt_lean".
# \param pyVersion  The python version to create bindings for, i.e. "3.12".
function(expandWheelTemplates moduleName pyVersion)
    set(wheelTarget trt_wheel_files_${moduleName}_${pyVersion})

    processWheelTemplates(${moduleName} ${pyVersion}
        __init__.py
        plugin/__init__.py
        plugin/_autotune.py
        plugin/_export.py
        plugin/_lib.py
        plugin/_plugin_class.py
        plugin/_tensor.py
        plugin/_top_level.py
        plugin/_utils.py
        plugin/_validate.py
    )

    add_custom_target(${wheelTarget} DEPENDS ${generatedWheelFiles})
    add_dependencies(trt_python_wheel_files ${wheelTarget})
endfunction()

# Enumerate all the combinations and create the per-python per-module targets.
foreach(moduleName IN LISTS TRT_PYTHON_MODULE_NAMES)
    foreach(pyVersion IN LISTS TRT_BUILD_PYTHON_PY_VERSIONS)
        createBindingLibrary(${moduleName} ${pyVersion})
        expandWheelTemplates(${moduleName} ${pyVersion})
        # TODO: Add custom targets to run copy_standalone_libs.py and build the wheels.
        # We should be able to generate a pyproject.toml from the current config and setup a custom command to build the wheel directly.
    endforeach()
endforeach()

else() # TRT_BUILD_ENABLE_NEW_PYTHON_FLOW - old flow is below this line

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
if(${TENSORRT_MODULE} STREQUAL "tensorrt")
    # tensorrt full dependencies
    file(GLOB_RECURSE SOURCE_FILES src/*.cpp)
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
        PATH_SUFFIXES lib
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
    set(nvinfer_lib_name "nvinfer_${TENSORRT_MAJOR_VERSION}")
    set(nvinfer_plugin_lib_name "nvinfer_plugin_${TENSORRT_MAJOR_VERSION}")
    set(nvonnxparser_lib_name "nvonnxparser_${TENSORRT_MAJOR_VERSION}")
    set(nvinfer_lean_lib_name "nvinfer_lean_${TENSORRT_MAJOR_VERSION}${vfc_suffix}")
    set(nvinfer_dispatch_lib_name "nvinfer_dispatch_${TENSORRT_MAJOR_VERSION}${vfc_suffix}")
else()
    set(nvinfer_lib_name "nvinfer")
    set(nvinfer_plugin_lib_name "nvinfer_plugin")
    set(nvonnxparser_lib_name "nvonnxparser")
    set(nvinfer_lean_lib_name "nvinfer_lean${vfc_suffix}")
    set(nvinfer_dispatch_lib_name "nvinfer_dispatch${vfc_suffix}")
endif()

if(${TENSORRT_MODULE} STREQUAL "tensorrt")
    set(TRT_LIBS ${nvinfer_lib_name} ${nvonnxparser_lib_name} ${nvinfer_plugin_lib_name})
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
target_include_directories(${LIB_NAME} BEFORE PUBLIC ${PY_CONFIG_INCLUDE} ${PY_INCLUDE})
if(MSVC)
    # For some reason, we must explicitly link against the Python library on Windows.
    target_link_libraries(${LIB_NAME} PRIVATE ${TRT_LIBS} ${PYTHON_LIB_NAME})
else()
    target_link_libraries(${LIB_NAME} PRIVATE ${TRT_LIBS})
endif()

# Note that we have to remove the `lib` prefix from the binding .so's
set_target_properties(${LIB_NAME} PROPERTIES PREFIX "")

endif()
