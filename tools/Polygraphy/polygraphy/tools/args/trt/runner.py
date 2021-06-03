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
from polygraphy.tools.util import misc as tools_util
from polygraphy.tools.util.script import Script


class TrtRunnerArgs(BaseArgs):
    def __init__(self, write=True):
        self._write = write


    def add_to_parser(self, parser):
        trt_args = parser.add_argument_group("TensorRT Inference", "Options for TensorRT Inference")
        if self._write:
            trt_args.add_argument("--save-engine", help="Path to save a TensorRT engine file", default=None)


    def register(self, maker):
        from polygraphy.tools.args.model import ModelArgs
        from polygraphy.tools.args.trt.loader import TrtLoaderArgs

        if isinstance(maker, ModelArgs):
            self.model_args = maker
        elif isinstance(maker, TrtLoaderArgs):
            self.trt_loader_args = maker


    def check_registered(self):
        assert self.model_args is not None, "ModelArgs is required!"
        assert self.trt_loader_args is not None, "TrtLoaderArgs is required!"


    def parse(self, args):
        self.save_engine = tools_util.get(args, "save_engine")


    def add_to_script(self, script, data_loader_name):
        script.add_import(imports=["TrtRunner"], frm="polygraphy.backend.trt")

        if self.model_args.model_type == "engine":
            loader_name = self.trt_loader_args.add_trt_serialized_engine_loader(script)
        else:
            script.add_import(imports=["EngineFromNetwork"], frm="polygraphy.backend.trt")
            loader_name = self.trt_loader_args.add_trt_network_loader(script)
            config_loader_name = self.trt_loader_args.add_trt_config_loader(script, data_loader_name)
            loader_str = Script.invoke("EngineFromNetwork", loader_name, config=config_loader_name)
            loader_name = script.add_loader(loader_str, "build_engine")

        SAVE_ENGINE = "SaveEngine"
        save_engine = Script.invoke(SAVE_ENGINE, loader_name, path=self.save_engine)
        if save_engine != Script.invoke(SAVE_ENGINE, loader_name):
            script.add_import(imports=[SAVE_ENGINE], frm="polygraphy.backend.trt")
            loader_name = script.add_loader(save_engine, "save_engine")

        runner_name = script.add_loader(Script.invoke("TrtRunner", loader_name), "trt_runner")
        script.add_runner(runner_name)
        return runner_name
