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

add_plugin_source(
    mhaRunner.cu
    mhaRunner.h
    qkvToContextInt8InterleavedPlugin.cpp
    qkvToContextInt8InterleavedPlugin.h
    qkvToContextInt8InterleavedPluginLegacy.cpp
    qkvToContextInt8InterleavedPluginLegacy.h
    qkvToContextPlugin.cpp
    qkvToContextPlugin.h
    qkvToContextPluginLegacy.cpp
    qkvToContextPluginLegacy.h
    zeroPadding2d.cu
    zeroPadding2d.h
)

set(BERT_QKV_SUPPORTED_SMS 
    75
    80
    86
    87
    89
    90
    100
    120
)

# Certain cubins are binary compatible between different SM versions, so they are reused.
# This function checks if a SM-named file should be compiled based on current SM enablement.
# Specifically, the SM80 files are compiled if either 80, 86, or 89 are enabled.
function(should_compile_kernel SM OUT_VAR)
    # If the target SM is any of 80/86/89, we need to check if any of those are enabled in CMAKE_CUDA_ARCHITECTURES.
    if((${SM} EQUAL 80) OR (${SM} EQUAL 86) OR (${SM} EQUAL 89))
        list(FIND CMAKE_CUDA_ARCHITECTURES 80 SM80_INDEX)
        list(FIND CMAKE_CUDA_ARCHITECTURES 86 SM86_INDEX)
        list(FIND CMAKE_CUDA_ARCHITECTURES 89 SM89_INDEX)
        if((NOT ${SM80_INDEX} EQUAL -1) OR
           (NOT ${SM86_INDEX} EQUAL -1) OR
           (NOT ${SM89_INDEX} EQUAL -1)
        )
            set(${OUT_VAR} TRUE PARENT_SCOPE)
        else()
            set(${OUT_VAR} FALSE PARENT_SCOPE)
        endif()
    else()
        list(FIND CMAKE_CUDA_ARCHITECTURES ${SM} SM_INDEX)
        if (NOT ${SM_INDEX} EQUAL -1)
            set(${OUT_VAR} TRUE PARENT_SCOPE)
        else()
            set(${OUT_VAR} FALSE PARENT_SCOPE)
        endif()
    endif()
endfunction()

add_subdirectory(fused_multihead_attention)
add_subdirectory(fused_multihead_attention_v2)

# Promote PLUGIN_SOURCES and PLUGIN_CU_SOURCES added by `add_subdirectory` to this file's parent.
if (NOT DEFINED ${TRT_BUILD_ENABLE_NEW_PLUGIN_FLOW})
    set(PLUGIN_SOURCES ${PLUGIN_SOURCES} PARENT_SCOPE)
    set(PLUGIN_CU_SOURCES ${PLUGIN_CU_SOURCES} PARENT_SCOPE)
endif()
