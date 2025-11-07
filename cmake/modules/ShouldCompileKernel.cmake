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

# \brief Converts a SM string (i.e. 86+abc) into the numeric SM version (i.e. 86).
# \returns the sm in the name specified by OUT_VAR.
function(get_numeric_sm SM OUT_VAR)
    # Convert the SM string to a numeric value
    if(${SM} MATCHES "^([0-9]+).*$")
        set(${OUT_VAR} ${CMAKE_MATCH_1} PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Invalid SM version: ${SM}")
    endif()
endfunction()

# \brief Converts the CMAKE_CUDA_ARCHITECTURES list into a list of numeric SM values.
# \returns the list in the name specified by OUT_VAR.
function(get_all_numeric_sms OUT_VAR)
    set(ALL_NUMERIC_SMS "")
    foreach(SM IN LISTS CMAKE_CUDA_ARCHITECTURES)
        get_numeric_sm(${SM} "SM")
        list(APPEND ALL_NUMERIC_SMS ${SM})
    endforeach()
    set(${OUT_VAR} ${ALL_NUMERIC_SMS} PARENT_SCOPE)
endfunction()

# \brief Converts the list returned by get_all_numeric_sms into a list of arch values.
# \returns the list in the name specified by OUT_VAR for native platform and OUT_VAR_CROSS for cross OS support. e.g. ptx, sm75, sm80, sm86, sm89, sm100, sm120.
function(get_all_fatbin_archs OUT_VAR OUT_VAR_CROSS)
    # Use get_all_numeric_sms to get SM values and convert them to sm-prefixed format
    set(ARCH_LIST "")
    set(ARCH_LIST_CROSS "")
    get_all_numeric_sms(NUMERIC_SMS)
    foreach(SM IN LISTS NUMERIC_SMS)
        list(APPEND ARCH_LIST "sm${SM}")
    endforeach()
    
    # Note: sm89 it is missing in NUMERIC_SMS since TRT treats sm89 as sm86.
    # We should add sm89 to the list to generate the builder resource for sm89.
    # If only sm86 is in the list, it means this build only supports sm86,
    # so no need to add sm89.
    list(FIND ARCH_LIST "sm86" SM86_INDEX)
    list(FIND ARCH_LIST "sm89" SM89_INDEX)
    list(LENGTH ARCH_LIST ARCH_LIST_COUNT)
    if(${SM86_INDEX} GREATER_EQUAL 0 AND ${SM89_INDEX} EQUAL -1 AND ${ARCH_LIST_COUNT} GREATER 1)
        list(APPEND ARCH_LIST "sm89")
    endif()


    # There is also a klib which only contains PTX code.
    list(APPEND ARCH_LIST "ptx")
    
    set(ARCH_LIST_CROSS ${ARCH_LIST})
    # Cask5 does not include sm100 cubins. Exclude sm100 for both
    # cross-OS support and native Windows build.
    list(FILTER ARCH_LIST_CROSS EXCLUDE REGEX "sm100")
    if(${TRT_BUILD_PLATFORM} STREQUAL ${TRT_PLATFORM_WIN10})
        list(FILTER ARCH_LIST EXCLUDE REGEX "sm100")
    endif()
    set(${OUT_VAR} ${ARCH_LIST} PARENT_SCOPE)
    set(${OUT_VAR_CROSS} ${ARCH_LIST_CROSS} PARENT_SCOPE)
endfunction()

# Certain cubins are binary compatible between different SM versions, so they are reused.
# This function checks if a SM-named file should be compiled based on current SM enablement.
# Specifically, the SM80 files are compiled if either 80, 86, or 89 are enabled.
function(should_compile_kernel SM OUT_VAR)
    get_all_numeric_sms(__TRT_NUMERIC_CUDA_ARCHS)
    # If the target SM is any of 80/86/89, we need to check if any of those are enabled in __TRT_NUMERIC_CUDA_ARCHS.
    if((${SM} EQUAL 80) OR (${SM} EQUAL 86) OR (${SM} EQUAL 89))
        list(FIND __TRT_NUMERIC_CUDA_ARCHS 80 SM80_INDEX)
        list(FIND __TRT_NUMERIC_CUDA_ARCHS 86 SM86_INDEX)
        list(FIND __TRT_NUMERIC_CUDA_ARCHS 89 SM89_INDEX)
        if((NOT ${SM80_INDEX} EQUAL -1) OR
           (NOT ${SM86_INDEX} EQUAL -1) OR
           (NOT ${SM89_INDEX} EQUAL -1)
        )
            set(${OUT_VAR} TRUE PARENT_SCOPE)
        else()
            set(${OUT_VAR} FALSE PARENT_SCOPE)
        endif()
    else()
        list(FIND __TRT_NUMERIC_CUDA_ARCHS ${SM} SM_INDEX)
        if (NOT ${SM_INDEX} EQUAL -1)
            set(${OUT_VAR} TRUE PARENT_SCOPE)
        else()
            set(${OUT_VAR} FALSE PARENT_SCOPE)
        endif()
    endif()
endfunction()
