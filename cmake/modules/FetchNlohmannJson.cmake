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

set(NLOHMANN_JSON_REPO "https://github.com/nlohmann/json.git" CACHE STRING "The base project URL to FetchContent_Declare for nlohmann/json")
set(NLOHMANN_JSON_TAG "v3.11.3" CACHE STRING "The git tag to FetchContent_Declare for nlohmann/json")

# We use this directory to ensure we only fetch a single copy of dependencies, even between builds.
# $HOME/storage is expected to be mounted from the host for developers.
set(TRT_THIRD_PARTY_DL_DIR "$ENV{HOME}/storage" CACHE PATH "Directory to download third party dependencies to")

# nlohmann/json's own CMake builds tests and install rules by default. Disable both
# so our configure stays fast and our install tree doesn't pick up unrelated targets.
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")

include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    PREFIX              "${CMAKE_BINARY_DIR}/third_party/nlohmann_json"
    GIT_REPOSITORY      ${NLOHMANN_JSON_REPO}
    GIT_TAG             ${NLOHMANN_JSON_TAG}
    GIT_SHALLOW         TRUE
    SOURCE_DIR          "${TRT_THIRD_PARTY_DL_DIR}/nlohmann_json/${NLOHMANN_JSON_TAG}"
    EXCLUDE_FROM_ALL
    UPDATE_DISCONNECTED ${TRT_FETCH_CONTENT_UPDATES_DISCONNECTED}
    OVERRIDE_FIND_PACKAGE   # downstream find_package(nlohmann_json) is satisfied from here.
)
FetchContent_MakeAvailable(nlohmann_json)

# Populate NLOHMANN_JSON_INCLUDE_DIRS for samples/common.
if(nlohmann_json_SOURCE_DIR)
    set(NLOHMANN_JSON_INCLUDE_DIRS "${nlohmann_json_SOURCE_DIR}/include"
        CACHE PATH "nlohmann/json include directory (set by FetchNlohmannJson.cmake)")
endif()
