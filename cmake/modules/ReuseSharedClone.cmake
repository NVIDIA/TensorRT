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

# Reuse an already-populated shared FetchContent clone when it is checked out at the pinned ref.
#
# FetchContent's population step performs git update and stamp work against SOURCE_DIR on every
# fresh build tree (and, for direct-mode population, deletes and re-clones it). That is slow on
# network filesystems and races between concurrent builds sharing the clone. When the clone's HEAD
# already matches the pinned ref, setting FETCHCONTENT_SOURCE_DIR_<NAME> makes
# FetchContent_MakeAvailable use the directory as-is and skip population entirely. A missing or
# mismatched clone leaves the variable unset, so normal population creates or repairs it. A value
# already provided by the user (e.g. -DFETCHCONTENT_SOURCE_DIR_<NAME>=...) is respected.
#
# The check compares commits only; local modifications to the clone are not detected, since
# verifying cleanliness would read the whole working tree, which is the very cost being avoided.
function(trt_reuse_shared_clone name source_dir git_ref)
    string(TOUPPER "${name}" _name_upper)
    if(DEFINED FETCHCONTENT_SOURCE_DIR_${_name_upper} OR NOT EXISTS "${source_dir}/.git")
        return()
    endif()
    find_package(Git QUIET)
    if(NOT Git_FOUND)
        return()
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" rev-parse HEAD "${git_ref}^{commit}"
        OUTPUT_VARIABLE _revs
        ERROR_QUIET
        RESULT_VARIABLE _result
    )
    if(NOT _result EQUAL 0)
        return()
    endif()
    string(STRIP "${_revs}" _revs)
    string(REPLACE "\n" ";" _revs "${_revs}")
    list(GET _revs 0 _head)
    list(GET _revs 1 _pinned)
    if(_head STREQUAL _pinned)
        set(FETCHCONTENT_SOURCE_DIR_${_name_upper} "${source_dir}" PARENT_SCOPE)
        message(STATUS "Reusing shared clone for ${name}: ${source_dir} @ ${git_ref}")
    endif()
endfunction()
