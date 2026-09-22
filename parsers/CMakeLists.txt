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

if(NOT MSVC)
    # Since we repackage protobuf (and abseil) into libnvonnxparser_static.a, we need to re-namespace them.
    # This works by doing a global string replacement on the namespaces.
    add_definitions(
        -Dgoogle=google_trtrepack
        -Dabsl=absl_trtrepack
    )
endif() # NOT MSVC

if(BUILD_PARSERS AND NOT TARGET protobuf::libprotobuf-lite)
    include(${TensorRT_SOURCE_DIR}/third_party/protobuf.cmake)
    set(ONNX_USE_LITE_PROTO ON CACHE INTERNAL "Forces the onnx parser to use libprotobuf-lite")
endif()

set(TENSORRT_ROOT ${TensorRT_SOURCE_DIR})
set(TENSORRT_INCLUDE_DIR "${TensorRT_SOURCE_DIR}/include")
set(TENSORRT_PYTHON_INCLUDE_DIR "${TensorRT_SOURCE_DIR}/include/impl")

add_subdirectory(onnx)
