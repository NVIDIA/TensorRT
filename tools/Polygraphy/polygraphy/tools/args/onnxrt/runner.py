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
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.util.script import Script


class OnnxrtRunnerArgs(BaseArgs):
    def register(self, maker):
        from polygraphy.tools.args.onnx.loader import OnnxLoaderArgs

        if isinstance(maker, OnnxLoaderArgs):
            self.onnx_loader_args = maker


    def check_registered(self):
        assert self.onnx_loader_args is not None, "OnnxLoaderArgs is required!"


    def add_to_script(self, script):
        script.add_import(imports=["OnnxrtRunner"], frm="polygraphy.backend.onnxrt")
        onnx_name = self.onnx_loader_args.add_serialized_onnx_loader(script)

        script.add_import(imports=["SessionFromOnnxBytes"], frm="polygraphy.backend.onnxrt")
        loader_name = script.add_loader(Script.invoke("SessionFromOnnxBytes", onnx_name), "build_onnxrt_session")

        runner_name = script.add_loader(Script.invoke("OnnxrtRunner", loader_name), "onnxrt_runner")
        script.add_runner(runner_name)
        return runner_name
