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

# This folder contains a bunch of source files holding cubins. Since the files are huge, we only want to include and compile them if used.
# Usage is indicated by the SM being set in CMAKE_CUDA_ARCHITECTURES.

function(add_plugin_source_if_exists)
    foreach(SRC_FILE IN LISTS ARGN)
        if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/${SRC_FILE})
            add_plugin_source(${SRC_FILE})
        endif()
    endforeach()
endfunction()

foreach(SM IN LISTS BERT_QKV_SUPPORTED_SMS)
    should_compile_kernel(${SM} SHOULD_COMPILE)
    if (${SHOULD_COMPILE})
        # Not every file exists for each SM, so we list all of the candidates and add them if present.
        add_plugin_source_if_exists(
            fused_multihead_attention_v2_fp16_64_64_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_fp16_96_64_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_fp16_128_32_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_fp16_128_64_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_fp16_256_32_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_fp16_256_64_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_fp16_384_64_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_fp16_512_32_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_fp16_512_64_kernel.sm${SM}.cpp

            fused_multihead_attention_v2_il_int8_64_64_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_il_int8_96_64_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_il_int8_128_32_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_il_int8_128_64_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_il_int8_192_64_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_il_int8_256_64_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_il_int8_384_64_kernel.sm${SM}.cpp

            fused_multihead_attention_v2_int8_64_64_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_int8_96_64_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_int8_128_32_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_int8_128_64_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_int8_192_64_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_int8_256_32_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_int8_256_64_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_int8_384_64_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_int8_512_32_kernel.sm${SM}.cpp
            fused_multihead_attention_v2_int8_512_64_kernel.sm${SM}.cpp
        )
    endif()
endforeach()

add_plugin_source(
    fused_multihead_attention_v2.h
)
