#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from polygraphy import mod
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.script import make_invocable


@mod.export()
class PluginRefArgs(BaseArgs):
    def register(self, maker):
        from polygraphy.tools.args.model import ModelArgs
        from polygraphy.tools.args.onnx import OnnxLoaderArgs

        if isinstance(maker, OnnxLoaderArgs):
            self.onnx_loader_args = maker
        if isinstance(maker, ModelArgs):
            self.model_args = maker

    def check_registered(self):
        assert self.onnx_loader_args is not None, "OnnxLoaderArgs is required!"
        assert self.model_args is not None, "ModelArgs is required!"

    def add_to_script(self, script):
        script.add_import(imports=["GsFromOnnx"], frm="polygraphy.backend.onnx")
        script.add_import(imports=["PluginRefRunner"], frm="polygraphy.backend.pluginref")

        onnx_name = self.onnx_loader_args.add_onnx_loader(script)
        loader_name = script.add_loader(make_invocable("GsFromOnnx", onnx_name), "pluginref")
        script.add_runner(make_invocable("PluginRefRunner", loader_name))
