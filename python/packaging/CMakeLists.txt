# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include(FlagToInt)

# Processes one or more wheel templates file, replacing any markers with concrete information and
# copying the result into the per-python per-module build dir.
#
# \param moduleName The module name to create the bindings for. One of "tensorrt", "tensorrt_dispatch", or "tensorrt_lean".
# \param pyVersion  The python version to create bindings for, i.e. "3.12".
# \param ARGN       A list of file paths, relative to './packaging/bindings_wheel/tensorrt', of the file(s) to copy.
# \returns generatedWheelFiles A list containing paths to all generated files, which can be used to create a custom target.
# \returns generatedFileOutDir A string containing the output directory for the generated files. Contains generator expressions.
function(processWheelTemplates wheelType moduleName pyVersion)
    set(outputDir ${CMAKE_CURRENT_BINARY_DIR}/${moduleName}_${wheelType}-py${pyVersion}/$<CONFIG>)
    foreach(filePath IN LISTS ARGN)
        # Target files that start with tensorrt/ need to be expanded into ${moduleName}/ instead of tensorrt/
        # So we need to rewrite the relative path, the source directory, *and* the output directory to match.
        # Note that we do not update the "outputDir" variable, since we want to report it back based on the original file path.
        if(filePath MATCHES "^tensorrt/")
            string(REGEX REPLACE "^tensorrt/" "" adjustedFilePath ${filePath})
            set(__srcDir ${CMAKE_CURRENT_LIST_DIR}/tensorrt)
            set(__srcFilePath ${adjustedFilePath})
            # Bit of a hack branch for standalone binding wheels
            if(wheelType STREQUAL "binding_standalone")
                set(__outDir ${outputDir}/${moduleName}_bindings)
                set(__outputFile ${outputDir}/${moduleName}_bindings/${adjustedFilePath})
            else()
                set(__outDir ${outputDir}/${moduleName})
                set(__outputFile ${outputDir}/${moduleName}/${adjustedFilePath})
            endif()
        elseif(filePath MATCHES "^tensorrt_libs/") # Standalone lib wheels use <module>_libs/ instead of <module>/
            string(REGEX REPLACE "^tensorrt_libs/" "" adjustedFilePath ${filePath})
            set(__srcDir ${CMAKE_CURRENT_LIST_DIR}/tensorrt_libs)
            set(__srcFilePath ${adjustedFilePath})
            set(__outDir ${outputDir}/${moduleName}_libs)
            set(__outputFile ${outputDir}/${moduleName}_libs/${adjustedFilePath})
        else()
            set(__srcDir ${CMAKE_CURRENT_LIST_DIR})
            set(__srcFilePath ${filePath})
            set(__outDir ${outputDir})
            set(__outputFile ${outputDir}/${filePath})
        endif()

        flagToInt(TRT_BUILD_WINML)

        get_target_property(TRT_OUTPUT_NAME tensorrt OUTPUT_NAME)
        string(REGEX MATCH "[a-zA-Z_]+[a-zA-Z]" TRT_OUTPUT_NAME ${TRT_OUTPUT_NAME})
        get_target_property(PARSER_OUTPUT_NAME nvonnxparser OUTPUT_NAME)
        string(REGEX MATCH "[a-zA-Z_]+[a-zA-Z]" PARSER_OUTPUT_NAME ${PARSER_OUTPUT_NAME})

        add_custom_command(
            OUTPUT ${__outputFile}
            COMMAND
                ${Python3_EXECUTABLE} ${TensorRT_SOURCE_DIR}/python/scripts/process_wheel_template.py
                --src-dir ${__srcDir}
                --dst-dir ${__outDir}
                --filepath ${__srcFilePath}
                --trt-module ${moduleName}
                --trt-py-version ${TensorRT_PACKAGE_VERSION}
                --cuda-version ${TRT_CUDA_VERSION}
                --trt-version ${TensorRT_VERSION}
                --plugin-disabled ${TRT_BUILD_WINML_INT}
                --trt-nvinfer-name ${TRT_OUTPUT_NAME}
                --trt-onnxparser-name ${PARSER_OUTPUT_NAME}
            DEPENDS
                ${TensorRT_SOURCE_DIR}/python/scripts/process_wheel_template.py
                ${CMAKE_CURRENT_LIST_DIR}/${filePath}
            COMMENT "Expanding wheel template ${filePath} for ${moduleName}"
            VERBATIM
        )

        list(APPEND generatedFiles ${__outputFile})
    endforeach()
    set(generatedWheelFiles ${generatedFiles} PARENT_SCOPE)
    set(generatedFileOutDir ${outputDir} PARENT_SCOPE)
endfunction()

# Create meta targets used to group the post-processed wheel files and all the python wheels.
add_custom_target(tensorrt_python_wheels)

# Specify a single output directory to hold all the generated wheels.
# This is similar to CMAKE_ARCHIVE_OUTPUT_DIRECTORY, but for wheels.
set(TRT_WHEEL_OUTPUT_DIR "${CMAKE_BINARY_DIR}/$<CONFIG>/dist" CACHE PATH "")

# Add a custom command to run pip install for the requirements.txt file.
# We need to run this only once prior to any calls to setup.py.
add_custom_command(
    OUTPUT package_install_cache.txt
    COMMAND ${Python3_EXECUTABLE} -m pip install --user -r ${TensorRT_SOURCE_DIR}/python/packaging/requirements.txt
    COMMAND ${CMAKE_COMMAND} -E touch package_install_cache.txt
    DEPENDS ${TensorRT_SOURCE_DIR}/python/packaging/requirements.txt
    VERBATIM
)
# Custom commands that call setup.py should depend on this target so that the requirements are installed first.
add_custom_target(trt_packaging_requirements_installed DEPENDS package_install_cache.txt)

add_subdirectory(bindings_wheel)
add_subdirectory(libs_wheel)
add_subdirectory(frontend_sdist)
add_subdirectory(metapackage)
