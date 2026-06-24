# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include_guard()

set(_cccl_default_repo "https://github.com/NVIDIA/cccl.git")


set(CCCL_REPO ${_cccl_default_repo} CACHE STRING "The base project URL to FetchContent_Declare for CCCL" )
set(CCCL_TAG "v3.4.0-rc0" CACHE STRING "The commit hash to FetchContent_Declare for CCCL")

# We use this directory to ensure we only fetch a single copy of dependencies, even between builds.
# $HOME/storage is expected to be mounted from the host for developers.
set(TRT_THIRD_PARTY_DL_DIR "$ENV{HOME}/storage" CACHE PATH "Directory to download third party dependencies to")

FetchContent_Declare(
    cccl
    PREFIX         "${CMAKE_BINARY_DIR}/third_party/cccl"
    GIT_REPOSITORY ${CCCL_REPO}
    GIT_TAG        ${CCCL_TAG}
    GIT_SHALLOW    TRUE
    SOURCE_DIR     "${TRT_THIRD_PARTY_DL_DIR}/cccl/${CCCL_TAG}"
    EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(cccl)
