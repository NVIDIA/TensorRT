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
class TrtRunnerArgs(BaseArgs):
    def register(self, maker):
        from polygraphy.tools.args.model import ModelArgs
        from polygraphy.tools.args.trt.loader import TrtEngineLoaderArgs

        if isinstance(maker, ModelArgs):
            self.model_args = maker
        elif isinstance(maker, TrtEngineLoaderArgs):
            self.trt_engine_loader_args = maker

    def check_registered(self):
        assert self.model_args is not None, "ModelArgs is required!"
        assert self.trt_engine_loader_args is not None, "TrtEngineLoaderArgs is required!"

    def add_to_script(self, script):
        script.add_import(imports=["TrtRunner"], frm="polygraphy.backend.trt")

        if self.model_args.model_type == "engine":
            loader_name = self.trt_engine_loader_args.add_trt_serialized_engine_loader(script)
        else:
            loader_name = self.trt_engine_loader_args.add_trt_build_engine_loader(script)

        script.add_runner(make_invocable("TrtRunner", loader_name))
