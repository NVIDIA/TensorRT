# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Note: This subdirectory is added multiple times, once for each binding library target.

# Indirectly call the current ADD_SOURCES_FUNCTION to populate target sources on the bindings lib that currently being setup.
# The parser is only included by the full "tensorrt" or "tensorrt_rtx" binding module.
if(${TRT_PYTHON_IS_FULL_BINDINGS})
    cmake_language(CALL ${ADD_SOURCES_FUNCTION}
        pyOnnx.cpp
    )
endif()
