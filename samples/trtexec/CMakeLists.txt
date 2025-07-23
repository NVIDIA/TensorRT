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

installLibraries(
    TARGETS trtexec
    OPTIONAL
    COMPONENT release
)

else()

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

# Change the file name if TRT_WINML variable is set
if (${TRT_BUILD_WINML})
    set_target_properties(trtexec PROPERTIES
        OUTPUT_NAME tensorrt_rtx)
endif()

endif()
