# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include_guard()

# This is set and immediately overwritten intentionally. It's here to document the public repo, and to provide the boilerplate we'd use if we ever ship it publicly:
set(_onnx_default_repo "https://github.com/onnx/onnx.git")

set(ONNX_REPO ${_onnx_default_repo} CACHE STRING "The base project URL to FetchContent_Declare for onnx" )
set(ONNX_TAG "v1.22.0" CACHE STRING "The commit hash to FetchContent_Declare for onnx")

FetchContent_Declare(
    onnx
    PREFIX         "${CMAKE_BINARY_DIR}/third_party/onnx"
    GIT_REPOSITORY ${ONNX_REPO}
    GIT_TAG        ${ONNX_TAG}
    GIT_SHALLOW    TRUE
    SOURCE_DIR     "${TensorRT_BINARY_DIR}/onnx"
    EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(onnx)
