#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
if (${TRT_BUILD_ENABLE_NEW_SAMPLES_FLOW})

add_executable(trtexec trtexec.cpp)
target_link_libraries(trtexec PRIVATE trt_samples_common)
if (TRT_BUILD_SAMPLES)
    add_dependencies(tensorrt_samples trtexec)
endif()

# Windows-specific configuration for long path support
if(MSVC)
    # For long path support to work on Windows, users must 
    # additionally enable a system-level setting.
    # Adding manifest as a source file works on CMake 3.4+.
    target_sources(trtexec PRIVATE trtexec.manifest)
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
if (${TRT_BUILD_WINML})
    set(sample_name "tensorrt_rtx")
endif()

set_target_properties(trtexec
    PROPERTIES
    OUTPUT_NAME ${sample_name}
)

# In this mode, we build an additional binary trtexec_static that always links tensorrt_static.
if(${TRT_BUILD_TRTEXEC_STATIC})
    add_executable(trtexec_static trtexec.cpp)
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
endif()

else() # TRT_BUILD_ENABLE_NEW_SAMPLES_FLOW - old flow below.

set(SAMPLE_SOURCES
    ../common/sampleDevice.cpp
    ../common/sampleEngines.cpp
    ../common/sampleInference.cpp
    ../common/sampleOptions.cpp
    ../common/sampleReporting.cpp
    ../common/sampleUtils.cpp
    ../common/bfloat16.cpp
    ../common/debugTensorWriter.cpp
    trtexec.cpp)

set(SAMPLE_PARSERS "onnx")

include(../CMakeSamplesTemplate.txt)

endif()
