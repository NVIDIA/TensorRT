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
    DataLoaderArgs,
    TrtConfigArgs,
)
from polygraphy.tools.base import Tool
from polygraphy.tools.script import Script, inline, safe


class TrtConfig(Tool):
    """
    Generate a template script to create a TensorRT builder configuration.
    """

    def __init__(self):
        super().__init__("trt-config")
        self.subscribe_args(ModelArgs(model_required=False))
        self.subscribe_args(DataLoaderArgs())
        self.subscribe_args(TrtConfigArgs())

    def add_parser_args(self, parser):
        parser.add_argument(
            "-o", "--output", help="Path to save the generated script.", type=argparse.FileType("w"), required=True
        )

    def run(self, args):
        script = Script(summary="Creates a TensorRT Builder Configuration.", always_create_runners=False)
        script.add_import(imports=["func"], frm="polygraphy")
        script.add_import(imports=["tensorrt as trt"])

        loader_name = self.arg_groups[TrtConfigArgs].add_trt_config_loader(script)
        if not loader_name:
            script.add_import(imports=["CreateConfig"], frm="polygraphy.backend.trt")
            loader_name = script.add_loader(safe("CreateConfig()"), "create_trt_config")
        params = safe("config")

        script.append_suffix(safe("@func.extend({:})", inline(loader_name)))
        script.append_suffix(safe("def load_config({:}):", inline(params)))
        script.append_suffix(
            safe("\tpass # TODO: Set up the builder configuration here. This function should not return anything.")
        )

        script.save(args.output)
