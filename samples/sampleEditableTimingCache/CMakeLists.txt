#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_executable(sample_editable_timing_cache sampleEditableTimingCache.cpp)
target_link_libraries(sample_editable_timing_cache PRIVATE trt_samples_common TRT_SAMPLES::tensorrt)
add_dependencies(tensorrt_samples sample_editable_timing_cache)

installLibraries(
    TARGETS sample_editable_timing_cache
    OPTIONAL
    COMPONENT internal
)

else()

set(SAMPLE_SOURCES sampleEditableTimingCache.cpp)
set(SAMPLE_PARSERS "onnx")
include(../CMakeSamplesTemplate.txt)

endif()
