#
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
#

from polygraphy import config, mod

def tensorrt_module_and_version_string():
    """
    Returns the name of the TensorRT module to import. This selects between
    TensorRT and TensorRT-RTX based on the value of config.USE_TENSORRT_RTX,
    and ensures that a consistent version of the module is imported.
    """
    if config.USE_TENSORRT_RTX:
        return "tensorrt_rtx>=1.0"
    else:
        return "tensorrt>=8.5"

def lazy_import_trt():
    """
    Returns either tensorrt or tensorrt_rtx based on config.USE_TENSORRT_RTX.
    
    Prefer to use this function instead of mod.lazy_import("tensorrt>=8.5") to
    import TensorRT, so that your code can use TensorRT-RTX.
    """

    return mod.lazy_import(tensorrt_module_and_version_string())
