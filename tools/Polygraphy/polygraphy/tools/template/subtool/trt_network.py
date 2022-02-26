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
import argparse

from polygraphy.tools.args import (
    ModelArgs,
    OnnxLoaderArgs,
    Tf2OnnxLoaderArgs,
    TfLoaderArgs,
    TrtNetworkLoaderArgs,
    TrtPluginLoaderArgs,
)
from polygraphy.tools.base import Tool
from polygraphy.tools.script import Script, inline, safe


class TrtNetwork(Tool):
    """
    Generate a template script to create a TensorRT network using the TensorRT network API,
    optionally starting from an existing model.
    """

    def __init__(self):
        super().__init__("trt-network")
        self.subscribe_args(ModelArgs(model_required=False, inputs=None))
        self.subscribe_args(TfLoaderArgs(artifacts=False))
        self.subscribe_args(Tf2OnnxLoaderArgs())
        self.subscribe_args(OnnxLoaderArgs())
        self.subscribe_args(TrtPluginLoaderArgs())
        self.subscribe_args(TrtNetworkLoaderArgs())

    def add_parser_args(self, parser):
        parser.add_argument(
            "-o", "--output", help="Path to save the generated script.", type=argparse.FileType("w"), required=True
        )

    def run(self, args):
        script = Script(
            summary="Creates a TensorRT Network using the Network API.", always_create_runners=False
        )
        script.add_import(imports=["func"], frm="polygraphy")
        script.add_import(imports=["tensorrt as trt"])

        if self.arg_groups[ModelArgs].model_file is not None:
            loader_name = self.arg_groups[TrtNetworkLoaderArgs].add_trt_network_loader(script)
            params = safe("builder, network, parser")
        else:
            script.add_import(imports=["CreateNetwork"], frm="polygraphy.backend.trt")
            loader_name = safe("CreateNetwork()")
            params = safe("builder, network")

        script.append_suffix(safe("@func.extend({:})", inline(loader_name)))
        script.append_suffix(safe("def load_network({:}):", inline(params)))
        script.append_suffix(safe("\tpass # TODO: Set up the network here. This function should not return anything."))

        script.save(args.output)
