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

add_executable(sample_onnx_mnist_coord_conv_ac sampleOnnxMnistCoordConvAC.cpp)
target_link_libraries(sample_onnx_mnist_coord_conv_ac PRIVATE trt_samples_common TRT_SAMPLES::tensorrt)
add_dependencies(tensorrt_samples sample_onnx_mnist_coord_conv_ac)

if(${TRT_BUILD_SAMPLES_LINK_STATIC_TRT})
    target_link_libraries(sample_onnx_mnist_coord_conv_ac PRIVATE
        tensorrt_plugins_static
    )
else()
    target_link_libraries(sample_onnx_mnist_coord_conv_ac PRIVATE
        tensorrt_plugins
    )
endif()

installLibraries(
    TARGETS sample_onnx_mnist_coord_conv_ac
    OPTIONAL
    COMPONENT internal
)

else()

set(SAMPLE_SOURCES sampleOnnxMnistCoordConvAC.cpp)

set(SAMPLE_PARSERS "onnx")
set(PLUGINS_NEEDED ON)

include(../CMakeSamplesTemplate.txt)

endif()
