# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
set(_pybind11_default_repo "https://github.com/pybind/pybind11.git")


set(PYBIND11_REPO ${_pybind11_default_repo} CACHE STRING "The base project URL to FetchContent_Declare for pybind11" )
set(PYBIND11_TAG "v3.0.1" CACHE STRING "The commit hash to FetchContent_Declare for pybind11")

# We use this directory to ensure we only fetch a single copy of dependencies, even between builds.
# $HOME/storage is expected to be mounted from the host for developers.
set(TRT_THIRD_PARTY_DL_DIR "$ENV{HOME}/storage" CACHE PATH "Directory to download third party dependencies to")
file(TO_CMAKE_PATH "${TRT_THIRD_PARTY_DL_DIR}" TRT_THIRD_PARTY_DL_DIR)

FetchContent_Declare(
    pybind11
    PREFIX              "${CMAKE_BINARY_DIR}/third_party/pybind11"
    GIT_REPOSITORY      ${PYBIND11_REPO}
    GIT_TAG             ${PYBIND11_TAG}
    GIT_SHALLOW         TRUE
    SOURCE_DIR          "${TRT_THIRD_PARTY_DL_DIR}/pybind11/${PYBIND11_TAG}"
    EXCLUDE_FROM_ALL
    UPDATE_DISCONNECTED ${TRT_FETCH_CONTENT_UPDATES_DISCONNECTED}
    OVERRIDE_FIND_PACKAGE    # ONNX is going to try and look for pybind11, so we redirect it to here.
)
FetchContent_MakeAvailable(pybind11)
