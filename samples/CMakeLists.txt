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

# This target will build all of the samples.
add_custom_target(tensorrt_samples)

# Build the common submodule, which creates the trt_samples_common target.
add_subdirectory(common)

# Public (OSS) Samples
set(SAMPLE_FOLDERS
    sampleCharRNN
    sampleDynamicReshape
    sampleEditableTimingCache
    sampleINT8API
    sampleIOFormats
    sampleNamedDimensions
    sampleNonZeroPlugin
    sampleOnnxMNIST
    sampleProgressMonitor
    trtexec
)

# This sample needs to link against nvinfer_plugin.
if(${TRT_BUILD_PLUGINS})
    list(APPEND SAMPLE_FOLDERS sampleOnnxMnistCoordConvAC)
endif()

include(workaround.cmake)

foreach(FOLDER IN LISTS SAMPLE_FOLDERS)
    add_subdirectory(${FOLDER})
endforeach()

else()

add_custom_target(samples)

set(OPENSOURCE_SAMPLES_LIST
    sampleCharRNN
    sampleDynamicReshape
    sampleEditableTimingCache
    sampleINT8API
    sampleNonZeroPlugin
    sampleOnnxMNIST
    sampleIOFormats
    sampleOnnxMnistCoordConvAC
    sampleNamedDimensions
    sampleProgressMonitor
    trtexec)

foreach(SAMPLE_ITER ${OPENSOURCE_SAMPLES_LIST})
    add_subdirectory(${SAMPLE_ITER})
endforeach(SAMPLE_ITER)

endif()
