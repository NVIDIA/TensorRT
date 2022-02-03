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
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.script import make_invocable


@mod.export()
class OnnxrtSessionArgs(BaseArgs):
    def register(self, maker):
        from polygraphy.tools.args.model import ModelArgs
        from polygraphy.tools.args.onnx.loader import OnnxLoaderArgs

        if isinstance(maker, OnnxLoaderArgs):
            self.onnx_loader_args = maker
        if isinstance(maker, ModelArgs):
            self.model_args = maker

    def add_to_parser(self, parser):
        self.group = parser.add_argument_group(
            "ONNX-Runtime Session Loader", "Options for the ONNX-Runtime Session Loader"
        )
        self.group.add_argument(
            "--providers",
            "--execution-providers",
            dest="providers",
            help="A list of execution providers to use in order of priority. "
            "Each provider may be either an exact match or a case-insensitive partial match "
            "for the execution providers available in ONNX-Runtime. For example, a value of 'cpu' would "
            "match the 'CPUExecutionProvider'",
            nargs="+",
            default=None,
        )

    def check_registered(self):
        assert self.onnx_loader_args is not None, "OnnxLoaderArgs is required!"
        assert self.model_args is not None, "ModelArgs is required!"

    def parse(self, args):
        self.providers = args_util.get(args, "providers")

    def add_onnxrt_session(self, script):
        if self.onnx_loader_args.should_use_onnx_loader():
            onnx_name = self.onnx_loader_args.add_serialized_onnx_loader(script)
        else:
            onnx_name = self.model_args.model_file

        script.add_import(imports=["SessionFromOnnx"], frm="polygraphy.backend.onnxrt")
        loader_name = script.add_loader(
            make_invocable("SessionFromOnnx", onnx_name, providers=self.providers), "build_onnxrt_session"
        )
        return loader_name

    def load_onnxrt_session(self):
        loader = args_util.run_script(self.add_onnxrt_session)
        return loader()
