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
    efficientNMSInference.cu
    efficientNMSInference.cuh
    efficientNMSInference.h
    efficientNMSParameters.h
    efficientNMSPlugin.cpp
    efficientNMSPlugin.h  
)

add_subdirectory(tftrt)

# Promote PLUGIN_SOURCES and PLUGIN_CU_SOURCES added by `add_subdirectory` to this file's parent.
if (NOT DEFINED ${TRT_BUILD_ENABLE_NEW_PLUGIN_FLOW})
    set(PLUGIN_SOURCES ${PLUGIN_SOURCES} PARENT_SCOPE)
    set(PLUGIN_CU_SOURCES ${PLUGIN_CU_SOURCES} PARENT_SCOPE)
endif()

