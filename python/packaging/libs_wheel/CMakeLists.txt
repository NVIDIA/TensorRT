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

# Adding a custom target to the all target requires passing the "ALL" keyword to the initial call.
# We use this variable so we can conditionally add the keyword based on the build options.
if(${TRT_BUILD_PYTHON_STANDALONE_WHEELS})
    set(ENABLE_ALL "ALL")
else()
    set(ENABLE_ALL "")
endif()

add_custom_target(tensorrt_libs_wheels ${ENABLE_ALL})

# \brief Creates a target named tensorrt_libs_wheel_${moduleName} which will build the Standalone Libs Wheel for that module.
#
# \details The wheel is created by expanding all template files (from this directory) into the per-module build directory.
# Then, the relevant TRT libraries (e.g. libnvinfer.so) are copied into the same directory as the generated files.
# Finally, the wheel is built by running setup.py with the appropriate arguments in the binary directory.
#
# \param moduleName The module name to create the libs wheel for. One of "tensorrt", "tensorrt_dispatch", or "tensorrt_lean".
function(buildLibsWheel moduleName)
    set(filesTarget trt_wheel_files_libs_${moduleName})

    set(wheelTemplateFiles
        tensorrt_libs/__init__.py
        LICENSE.txt
        setup.cfg
        setup.py
    )

    # Expands all template files for the libs for the target module. The python version is unused (and set to 0).
    # File paths starting with "tensorrt_libs/" are expanded into "${moduleName}_libs/".
    processWheelTemplates(libs ${moduleName} 0 ${wheelTemplateFiles})

    # Determine which targets are packed into the wheel.
    if(${moduleName} STREQUAL "tensorrt_dispatch")
        set(moduleLibraryTargets tensorrt_dispatch_runtime)
    elseif(${moduleName} STREQUAL "tensorrt_lean")
        set(moduleLibraryTargets tensorrt_lean_runtime)
        if(${TRT_BUILD_PLUGINS})
            list(APPEND moduleLibraryTargets tensorrt_vc_plugins)
        endif()
    elseif(${moduleName} STREQUAL "tensorrt" OR ${moduleName} STREQUAL "tensorrt_rtx")
        set(moduleLibraryTargets tensorrt)
        get_all_fatbin_archs(KLIB_ARCHS KLIB_ARCHS_CROSS)
        if(NOT ${TRT_BUILD_WINML})
            if(NOT ${TRT_BUILD_SPLIT_KLIB})
                list(APPEND moduleLibraryTargets tensorrt_builder_resource)
            else()
                foreach(ARCH IN LISTS KLIB_ARCHS)
                    list(APPEND moduleLibraryTargets tensorrt_builder_resource_${ARCH})
                endforeach()
            endif()
        endif()
        if(${TRT_BUILD_PLUGINS})
            list(APPEND moduleLibraryTargets tensorrt_plugins)
        endif()
        if(${TRT_BUILD_ONNX_PARSER})
            list(APPEND moduleLibraryTargets nvonnxparser)
        endif()
        if(NOT ${TRT_BUILD_SKIP_WIN_BUILDER_RESOURCE})
            foreach(ARCH IN LISTS KLIB_ARCHS_CROSS)
                list(APPEND moduleLibraryTargets tensorrt_builder_resource_win_${ARCH})
            endforeach()
        endif()
    else()
        message(FATAL_ERROR "Unknown module name: ${moduleName}. Expected 'tensorrt', 'tensorrt_dispatch', or 'tensorrt_lean'.")
    endif()

    # copy_standalone_libs.py needs a space-separated string of the files to copy.
    set(moduleLibraryPaths "")
    foreach(tgt IN LISTS moduleLibraryTargets)
        list(APPEND moduleLibraryPaths "$<TARGET_FILE:${tgt}>")
    endforeach()
    string(JOIN " " moduleLibraryFiles ${moduleLibraryPaths})

    # CMake needs a list of files output by the custom command.
    # But it needs it at configure time, so we can't use generator expressions here.
    set(moduleLibraryOutputFiles)
    foreach(tgt IN LISTS moduleLibraryTargets)
        # Get the library filename (configure time)
        get_target_property(lib_name ${tgt} OUTPUT_NAME)
        if(NOT lib_name)
            set(lib_name ${tgt})
        endif()

        # Add extension based on platform
        if(MSVC)
            set(lib_filename "${lib_name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
        else()
            set(lib_filename "${CMAKE_SHARED_LIBRARY_PREFIX}${lib_name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
        endif()

        # Add to output files list
        list(APPEND moduleLibraryOutputFiles "${generatedFileOutDir}/${moduleName}_libs/${lib_filename}")
    endforeach()

    # ARM Linux requires that we specify the page size for the patchelf call.
    if(${TRT_BUILD_PLATFORM} STREQUAL ${TRT_PLATFORM_AARCH64})
        set(pageSizeArgs "--page-size=65536")
    else()
        set(pageSizeArgs "")
    endif()

    # Copies the TRT libraries (nvinfer.so, nvinfer_dispatch.so, etc) into the same directory as the generated files.
    add_custom_command(
        OUTPUT ${moduleLibraryOutputFiles}
        COMMAND
            ${Python3_EXECUTABLE} ${TensorRT_SOURCE_DIR}/scripts/wheel/copy_standalone_libs.py
            --update
            --follow-symlinks
            --mode=$<CONFIG>
            ${pageSizeArgs}
            --build-dir=${CMAKE_LIBRARY_OUTPUT_DIRECTORY} # Since we give absolute paths, this is mostly unused, but we'll keep it for consistency.
            --output=${generatedFileOutDir}/${moduleName}_libs/
            --trim-version
            ${moduleLibraryPaths}
        COMMENT "Copying TRT libraries for ${moduleName} to ${generatedFileOutDir}/${moduleName}_libs/"
        DEPENDS ${moduleLibraryTargets}
        VERBATIM
    )

    # Creates a new custom target, and makes trt_standalone_wheel_files depend on the new target.
    add_custom_target(${filesTarget} DEPENDS ${generatedWheelFiles} ${moduleLibraryOutputFiles})

    if(MSVC)
        set(wheelPlatform win_amd64)
    else()
        execute_process(
            COMMAND ldd --version
            OUTPUT_VARIABLE LDD_OUTPUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(LDD_OUTPUT MATCHES "([0-9]+\\.[0-9]+)")
            set(GLIBC_VERSION ${CMAKE_MATCH_1})
            string(REPLACE "." "_" GLIBC_VERSION ${GLIBC_VERSION})
        else()
            message(FATAL_ERROR "Failed to determine glibc version from ldd output: ${LDD_OUTPUT}")
        endif()

        set(wheelPlatform manylinux_${GLIBC_VERSION}_${TRT_CONFIG_ARCH})
    endif()

    # Define the output directory for the wheel
    set(wheelOutDir ${TRT_WHEEL_OUTPUT_DIR}/libs)
    set(wheelOutputFile ${wheelOutDir}/${moduleName}_cu${CUDAToolkit_VERSION_MAJOR}_libs-${TensorRT_PACKAGE_VERSION}-py2.py3-none-${wheelPlatform}.whl)

    # Add a custom command to build the wheel
    add_custom_command(
        OUTPUT ${wheelOutputFile}
        COMMAND ${Python3_EXECUTABLE} setup.py -q bdist_wheel --plat-name=${wheelPlatform} --dist-dir=${wheelOutDir}
        WORKING_DIRECTORY ${generatedFileOutDir}
        DEPENDS ${filesTarget} trt_packaging_requirements_installed
        VERBATIM
    )

    set(wheelTarget tensorrt_libs_wheel_${moduleName})

    # Add a custom target for the wheel
    add_custom_target(
        ${wheelTarget}
        ${ENABLE_ALL}
        DEPENDS ${wheelOutputFile}
    )

    add_dependencies(tensorrt_libs_wheels ${wheelTarget})

    # Standalone wheels are only published to the internal tarfile as they are released separately.
    install(FILES
        ${wheelOutputFile}
        DESTINATION python_standalone
        COMPONENT internal
        OPTIONAL
    )
endfunction()

foreach(moduleName IN LISTS TRT_PYTHON_MODULE_NAMES)
    buildLibsWheel(${moduleName})
endforeach()

add_dependencies(tensorrt_python_wheels tensorrt_libs_wheels)
