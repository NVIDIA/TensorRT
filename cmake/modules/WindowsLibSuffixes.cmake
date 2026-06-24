# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# \brief Handles setting output names for libraries on Windows to include version information.
#        We have two conventions:
#        1. For enterprise, shared libraries have a suffix _${major_version}
#        2. For TRT-RTX, shared libraries have a suffix _${major_version}_${minor_version}
# \note This is a no-op for non-Windows platforms.
#
# \param target_name The name of the target to update.
# \param major_version The major version of the library.
# \param minor_version The minor version of the library.
function(update_windows_output_name target_name major_version minor_version)
    if(MSVC)
        get_target_property(tgt_output_name ${target_name} OUTPUT_NAME)
        if(NOT tgt_output_name)
            set(tgt_output_name ${target_name})
        endif()

        if(${TRT_BUILD_WINML})
            set(tgt_output_name "${tgt_output_name}_${major_version}_${minor_version}")
        else()
            set(tgt_output_name "${tgt_output_name}_${major_version}")
        endif()

        set_target_properties(${target_name} PROPERTIES OUTPUT_NAME ${tgt_output_name})
        message(STATUS "Updated output name for target '${target_name}' to '${tgt_output_name}'")
    endif()
endfunction()
