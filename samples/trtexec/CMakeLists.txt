#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
add_executable(trtexec trtexec_main.cpp trtexec.cpp)
target_link_libraries(trtexec PRIVATE trt_samples_common)
if (TRT_BUILD_SAMPLES)
    add_dependencies(tensorrt_samples trtexec)
endif()

# Windows-specific configuration for long path support
if(MSVC)
    # For long path support to work on Windows, users must
    # additionally enable a system-level setting.
    # Adding manifest as a source file works on CMake 3.4+.
    if(TRT_IS_LINUX_TO_WINDOWS_CROSS_COMPILATION)
        # Remove all content in the manifest file before the <?xml ...?> tag
        # to ensure lld-link is able to parse the file.
        file(READ trtexec.manifest trtexec_manifest_content)
        string(REGEX REPLACE "<!--.*-->\n\n" "" trtexec_manifest_content "${trtexec_manifest_content}")
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/trtexec.manifest "${trtexec_manifest_content}")

        target_sources(trtexec PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/trtexec.manifest)
    else()
        target_sources(trtexec PRIVATE trtexec.manifest)
    endif()
endif()

installLibraries(
    TARGETS trtexec
    OPTIONAL
    COMPONENT external
)

# When statically linked, trtexec requires the plugins library.
# Otherwise it will resolve it via dlopen and doesn't need a link-time dependency.
if(${TRT_BUILD_SAMPLES_LINK_STATIC_TRT})
    target_link_libraries(trtexec PRIVATE
        $<LINK_LIBRARY:WHOLE_ARCHIVE,tensorrt_plugins_static>
    )
endif()

# Change the file name if TRT_WINML variable is set
set(sample_name "trtexec")
if (${TRT_PRODUCT_IS_RTX})
    set(sample_name "tensorrt_rtx")
endif()

set_target_properties(trtexec
    PROPERTIES
    OUTPUT_NAME ${sample_name}
)

# In this mode, we build an additional binary trtexec_static that always links tensorrt_static.
if(${TRT_BUILD_TRTEXEC_STATIC})
    add_executable(trtexec_static trtexec_main.cpp trtexec.cpp)
    target_link_libraries(trtexec_static PRIVATE trt_samples_common)
    if (TRT_BUILD_SAMPLES)
        add_dependencies(tensorrt_samples trtexec_static)
    endif()

    installLibraries(
        TARGETS trtexec_static
        OPTIONAL
        COMPONENT internal
    )

    target_link_libraries(trtexec_static PRIVATE
        $<LINK_LIBRARY:WHOLE_ARCHIVE,tensorrt_static>
        $<LINK_LIBRARY:WHOLE_ARCHIVE,tensorrt_plugins_static>
    )
    # Link as PIE and the binary will be loaded at a high address.
    # Then, avoid conflicts with ASan's shadow memory range.
    # MSVC does not support -pie.
    if(NOT MSVC)
        target_link_options(trtexec_static PRIVATE -pie)
    endif()
endif()

# In this mode, we build an additional binary tensorrt_rtx_internal with additional macro.
if(TRT_BUILD_WINML_PLUGIN)
    set(internal_sample_name "tensorrt_rtx_internal")
    add_executable(${internal_sample_name} trtexec_main.cpp trtexec.cpp)
    target_link_libraries(${internal_sample_name} PRIVATE trt_samples_common)
    target_compile_definitions(${internal_sample_name} PRIVATE TRT_WINML_PLUGIN=1)

    installLibraries(
        TARGETS ${internal_sample_name}
        OPTIONAL
        COMPONENT internal
    )
endif()
