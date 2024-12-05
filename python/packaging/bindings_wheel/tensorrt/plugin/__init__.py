#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

logger = trt.Logger()
logger.log(trt.Logger.WARNING, "Functionality provided through tensorrt.plugin module is experimental.")

# export.public_api() will expose things here. To make sure that happens, we just need to
# import all the submodules so that the decorator is actually executed (__discover_modules() below).
__all__ = []

def __discover_modules():
    import importlib
    import pkgutil

    mods = [importlib.import_module(__package__)]
    while mods:
        mod = mods.pop(0)

        yield mod

        if hasattr(mod, "__path__"):
            mods.extend(
                [
                    importlib.import_module(f"{mod.__name__}.{submod.name}")
                    for submod in pkgutil.iter_modules(mod.__path__)
                ]
            )


_ = list(__discover_modules())
