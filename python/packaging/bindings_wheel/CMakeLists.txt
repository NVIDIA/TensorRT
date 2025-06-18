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

# \brief Creates a target named tensorrt_bindings_wheel_${moduleName}_${pyVersion} which will build the Bindings Wheel for that combination.
# 
# \details The wheel is created by expanding all template files (from this directory) into the per-module per-python build directory.
# Then, the binding library (tensorrt.so) is copied into the same directory as the generated files.
# Finally, the wheel is built by running setup.py with the appropriate arguments in the binary directory.
#
# \param moduleName The module name to create the bindings for. One of "tensorrt", "tensorrt_dispatch", or "tensorrt_lean".
# \param pyVersion  The python version to create bindings for, i.e. "3.12".
function(buildBindingsWheel moduleName pyVersion)
    set(filesTarget trt_wheel_files_binding_${moduleName}_${pyVersion})

    set(wheelTemplateFiles
        tensorrt/__init__.py
        LICENSE.txt
        poetry.lock
        pyproject.toml
        setup.cfg
        setup.py
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
    processWheelTemplates(binding ${moduleName} ${pyVersion} ${wheelTemplateFiles})

    # Creates a new custom target, and makes trt_python_wheel_files depend on the new target.
    add_custom_target(${filesTarget} DEPENDS ${generatedWheelFiles})
    add_dependencies(trt_python_wheel_files ${filesTarget})

    # Copies the binding library (tensorrt.so) into the same directory as the generated files.
    add_custom_command(
        TARGET trt_wheel_files_binding_${moduleName}_${pyVersion}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:tensorrt_bindings_${moduleName}_${pyVersion}>
            ${generatedFileOutDir}/${moduleName}
        COMMENT "Copying bindings library to wheel directory"
        VERBATIM
    )

    string(REPLACE "." "" pyVerStr ${pyVersion})

    if (MSVC)
        set(wheelPlatform win_${TRT_CONFIG_ARCH})
    else()
        set(wheelPlatform linux_${TRT_CONFIG_ARCH})
    endif()

    # Define the output directory for the wheel
    set(wheelOutDir ${TRT_WHEEL_OUTPUT_DIR}/bindings)
    set(wheelOutputFile ${wheelOutDir}/${moduleName}-${TensorRT_PACKAGE_VERSION}-cp${pyVerStr}-none-${wheelPlatform}.whl)

    # Add a custom command to build the wheel
    add_custom_command(
        OUTPUT ${wheelOutputFile}
        COMMAND ${Python3_EXECUTABLE} setup.py -q bdist_wheel --python-tag=cp${pyVerStr} --plat-name=${wheelPlatform} --dist-dir=${wheelOutDir}
        WORKING_DIRECTORY ${generatedFileOutDir}
        DEPENDS tensorrt_bindings_${moduleName}_${pyVersion} ${generatedWheelFiles} trt_packaging_requirements_installed
        VERBATIM
    )

    set(wheelTarget tensorrt_bindings_wheel_${moduleName}_${pyVersion})

    # Add a custom target for the wheel
    add_custom_target(
        ${wheelTarget}
        ALL
        DEPENDS ${wheelOutputFile}
    )

    add_dependencies(${wheelTarget} trt_python_wheel_files)
    add_dependencies(tensorrt_bindings_wheels ${wheelTarget})

    install(FILES
        ${wheelOutputFile}
        DESTINATION wheels
        COMPONENT release
        OPTIONAL
    )
endfunction()

foreach(moduleName IN LISTS TRT_PYTHON_MODULE_NAMES)
    foreach(pyVersion IN LISTS TRT_BUILD_PYTHON_PY_VERSIONS)
        buildBindingsWheel(${moduleName} ${pyVersion})
    endforeach()
endforeach()

add_dependencies(tensorrt_python_wheels tensorrt_bindings_wheels)
