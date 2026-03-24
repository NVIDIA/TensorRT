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

add_custom_target(tensorrt_bindings_wheels ALL)
add_custom_target(trt_bindings_wheel_files)

define_property(TARGET
    PROPERTY TRT_WHEEL_STAGING_DIR
    BRIEF_DOCS "The directory containing all the files that were packaged into this wheel."
)


# \brief Creates a target named tensorrt_bindings_wheel_${moduleName}_${pyVersion} which will build the Bindings Wheel for that combination.
#
# \details The wheel is created by expanding all template files (from this directory) into the per-module per-python build directory.
# Then, the binding library (tensorrt.so) is copied into the same directory as the generated files.
# Finally, the wheel is built by running setup.py with the appropriate arguments in the binary directory.
#
# \param moduleName   The module name to create the bindings for. One of "tensorrt", "tensorrt_dispatch", or "tensorrt_lean".
# \param pyVersion    The python version to create bindings for, i.e. "3.12".
# \param isStandalone If we are building the wheels for distribution with the standalone libs wheels.
#                     If true, the binding wheels' top-level folder will be <module_name>_bindings instead of <module_name>
function(buildBindingsWheel moduleName pyVersion isStandalone)
    if (isStandalone)
        set(standaloneMarker "_standalone")
        set(bindingSuffix "_bindings")
    else()
        set(standaloneMarker "")
        set(bindingSuffix "")
    endif()

    set(filesTarget trt_wheel_files_binding_${moduleName}_${pyVersion}${standaloneMarker})

    string(REPLACE "." "" pyVerStr ${pyVersion})

    get_wheel_platform(${isStandalone} wheelPlatform)

    set(wheelTemplateFiles
        tensorrt/__init__.py
        LICENSE.txt
        pyproject.toml
    )

    if(${TRT_BUILD_PLUGINS})
        list(APPEND wheelTemplateFiles
            tensorrt/plugin/__init__.py
            tensorrt/plugin/_autotune.py
            tensorrt/plugin/_export.py
            tensorrt/plugin/_lib.py
            tensorrt/plugin/_plugin_class.py
            tensorrt/plugin/_tensor.py
            tensorrt/plugin/_top_level.py
            tensorrt/plugin/_utils.py
            tensorrt/plugin/_validate.py
        )
    endif()

    # Expands all template files for the bindings for the target module and python version.
    # File paths starting with "tensorrt/" are expanded into "${moduleName}/".
    processWheelTemplates(binding${standaloneMarker} ${moduleName} ${pyVersion} ${wheelTemplateFiles})
    copyTRTBuildBackend()

    # We need to copy the binding library into the directory that the wheel is created from.
    set(copiedBindingLibrary ${generatedFileOutDir}/${moduleName}${bindingSuffix}/${moduleName}${CMAKE_SHARED_LIBRARY_SUFFIX})

    # I bet you thought you were free from more windows exceptions, I certainly did.
    # But here we are again, with yet another naming convention™
    if(MSVC)
        set(copiedBindingLibrary ${generatedFileOutDir}/${moduleName}${bindingSuffix}/${moduleName}.cp${pyVerStr}-${wheelPlatform}.pyd)
    endif()

    # Creates a new custom target, and makes trt_bindings_wheel_files depend on the new target.
    add_custom_target(${filesTarget} DEPENDS ${generatedWheelFiles} ${copiedBindingLibrary} ${copiedBuildBackend})
    add_dependencies(trt_bindings_wheel_files ${filesTarget})

    # Copies the binding library (tensorrt.so) into the same directory as the generated files.
    add_custom_command(
        OUTPUT ${copiedBindingLibrary}
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:tensorrt_bindings_${moduleName}_${pyVersion}>
            ${copiedBindingLibrary}
        DEPENDS tensorrt_bindings_${moduleName}_${pyVersion}
        COMMENT "Copying bindings library for ${moduleName} to ${copiedBindingLibrary}"
        VERBATIM
    )

    # Define the output directory for the wheel
    set(wheelOutDir ${TRT_WHEEL_OUTPUT_DIR}/bindings${standaloneMarker})

    # Wheel type determines how the build backend computes the package name
    if(isStandalone)
        set(wheelType "binding_standalone")
        set(wheelOutputFile ${wheelOutDir}/${moduleName}_cu${CUDAToolkit_VERSION_MAJOR}_bindings-${TensorRT_PACKAGE_VERSION}-cp${pyVerStr}-none-${wheelPlatform}.whl)
    else()
        set(wheelType "binding")
        set(wheelOutputFile ${wheelOutDir}/${moduleName}-${TensorRT_PACKAGE_VERSION}-cp${pyVerStr}-none-${wheelPlatform}.whl)
    endif()

    # Add a custom command to build the wheel using PEP 517 compliant build frontend.
    # The custom tensorrt_build_backend handles package naming and wheel tags via config settings.
    add_custom_command(
        OUTPUT ${wheelOutputFile}
        COMMAND
            ${Python3_EXECUTABLE} -m build --wheel --no-isolation
                --config-setting=--quiet
                --config-setting=wheel-type=${wheelType}
                --config-setting=python-tag=cp${pyVerStr}
                --config-setting=plat-name=${wheelPlatform}
                --outdir ${wheelOutDir}
        WORKING_DIRECTORY ${generatedFileOutDir}
        DEPENDS ${copiedBindingLibrary} ${copiedBuildBackend} ${generatedWheelFiles} trt_packaging_requirements_installed
        VERBATIM
    )

    set(wheelTarget tensorrt_bindings_wheel_${moduleName}_${pyVersion}${standaloneMarker})

    # Add a custom target for the wheel
    add_custom_target(
        ${wheelTarget}
        ALL
        DEPENDS ${wheelOutputFile}
    )

    add_dependencies(${wheelTarget} trt_bindings_wheel_files)
    add_dependencies(tensorrt_bindings_wheels ${wheelTarget})

    set_target_properties(${wheelTarget}
        PROPERTIES TRT_WHEEL_STAGING_DIR "${generatedFileOutDir}"
    )

    if(isStandalone)
        set(bindingsInstallComponent internal)
    else()
        set(bindingsInstallComponent external)
    endif()

    install(FILES
        ${wheelOutputFile}
        DESTINATION python${standaloneMarker}
        COMPONENT ${bindingsInstallComponent}
        OPTIONAL
    )
endfunction()

foreach(moduleName IN LISTS TRT_PYTHON_MODULE_NAMES)
    foreach(pyVersion IN LISTS TRT_BUILD_PYTHON_PY_VERSIONS)
        buildBindingsWheel(${moduleName} ${pyVersion} FALSE)
    endforeach()
endforeach()

add_dependencies(tensorrt_python_wheels tensorrt_bindings_wheels)

# For the standalone wheels, we need to build an additional copy of the wheels that uses <moduleName>_bindings as the package directory.
# This prevents clashes with the metapackage, which uses the "tensorrt" module name for the standalone path.
# TODO TRT-11.0: Can we drop this and change the name of the metapackage?
if(${TRT_BUILD_PYTHON_STANDALONE_WHEELS})
    foreach(moduleName IN LISTS TRT_PYTHON_MODULE_NAMES)
        foreach(pyVersion IN LISTS TRT_BUILD_PYTHON_PY_VERSIONS)
            buildBindingsWheel(${moduleName} ${pyVersion} TRUE)
        endforeach()
    endforeach()
endif()
