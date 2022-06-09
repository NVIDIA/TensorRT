#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from polygraphy import mod
from polygraphy.tools.args.base import BaseRunnerArgs
from polygraphy.tools.args.backend.onnx import OnnxLoadArgs
from polygraphy.tools.script import make_invocable


@mod.export()
class PluginRefRunnerArgs(BaseRunnerArgs):
    """
    Plugin Reference Runner Inference: running inference with the plugin reference runner.

    Depends on:

        - OnnxLoadArgs
    """

    def get_name_opt_impl(self):
        return "Plugin CPU Reference", "pluginref"

    def add_to_script_impl(self, script):
        script.add_import(imports=["GsFromOnnx"], frm="polygraphy.backend.onnx")
        script.add_import(imports=["PluginRefRunner"], frm="polygraphy.backend.pluginref")

        onnx_name = self.arg_groups[OnnxLoadArgs].add_to_script(script)
        loader_name = script.add_loader(make_invocable("GsFromOnnx", onnx_name), "pluginref")
        script.add_runner(make_invocable("PluginRefRunner", loader_name))
