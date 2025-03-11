#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tensorrt as trt
from types import ModuleType
import importlib

def public_api(module: ModuleType = None, symbol: str = None):
    def export_impl(obj):
        nonlocal module, symbol

        module = module or importlib.import_module(__package__)
        symbol = symbol or obj.__name__

        if not hasattr(module, "__all__"):
            module.__all__ = []

        module.__all__.append(symbol)
        setattr(module, symbol, obj)

        return obj

    return export_impl

IS_AOT_ENABLED = hasattr(trt, "QuickPluginCreationRequest")
