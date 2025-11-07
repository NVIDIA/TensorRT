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

add_custom_target(tensorrt_metapackage_sdist ${ENABLE_ALL})

# \brief Creates a target named tensorrt_metapackage_sdist_${moduleName} which will build the metapackage SDist for that module.
#
# \param moduleName The module name to create the metapackage for. One of "tensorrt", "tensorrt_dispatch", or "tensorrt_lean".
function(buildMetapackageWheel moduleName)
    set(filesTarget trt_wheel_files_metapackage_${moduleName})

    set(wheelTemplateFiles
        LICENSE.txt
        setup.cfg
        setup.py
    )

    # Expands all template files for the metapackage for the target module. The python version is unused (and set to 0).
    # File paths starting with "tensorrt/" are expanded into "${moduleName}/".
    processWheelTemplates(metapackage ${moduleName} 0 ${wheelTemplateFiles})

    # Creates a new custom target, and makes trt_standalone_wheel_files depend on the new target.
    add_custom_target(${filesTarget} DEPENDS ${generatedWheelFiles})

    # Define the output directory for the wheel
    set(sdistOutDir ${TRT_WHEEL_OUTPUT_DIR}/metapackage)
    set(sdistOutputFile ${sdistOutDir}/${moduleName}-${TensorRT_PACKAGE_VERSION}.tar.gz)

    # Add a custom command to build the wheel
    add_custom_command(
        OUTPUT ${sdistOutputFile}
        COMMAND ${Python3_EXECUTABLE} setup.py -q sdist --dist-dir=${sdistOutDir}
        WORKING_DIRECTORY ${generatedFileOutDir}
        DEPENDS ${filesTarget} trt_packaging_requirements_installed
        VERBATIM
    )

    set(sdistTarget tensorrt_metapackage_sdist_${moduleName})

    # Add a custom target for the wheel
    add_custom_target(
        ${sdistTarget}
        ${ENABLE_ALL}
        DEPENDS ${sdistOutputFile}
    )

    add_dependencies(tensorrt_metapackage_sdist ${sdistTarget})

    # Standalone wheels are only published to the internal tarfile as they are released separately.
    install(FILES
        ${sdistOutputFile}
        DESTINATION python_standalone
        COMPONENT internal
        OPTIONAL
    )
endfunction()

foreach(moduleName IN LISTS TRT_PYTHON_MODULE_NAMES)
    buildMetapackageWheel(${moduleName})
endforeach()

add_dependencies(tensorrt_python_wheels tensorrt_metapackage_sdist)
