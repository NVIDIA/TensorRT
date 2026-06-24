#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Base source files
set(PLUGIN_COMMON_SOURCES
    bboxUtils.h
    bertCommon.h
    checkMacrosPlugin.cpp
    checkMacrosPlugin.h
    common.cuh
    cub_helper.h
    cublasLtWrapper.cpp
    cublasLtWrapper.h
    cublasWrapper.cpp
    cublasWrapper.h
    cudaDriverWrapper.cpp
    cudaDriverWrapper.h
    cudnnWrapper.cpp
    cudnnWrapper.h
    dimsHelpers.h
    half.h
    mrcnn_config.h
    nmsHelper.cpp
    plugin.cpp
    plugin.h
    reducedMathPlugin.cpp
    scopedCudaStream.h
    serialize.hpp
    templates.h
)

add_plugin_source(${PLUGIN_COMMON_SOURCES})

add_subdirectory(kernels)
