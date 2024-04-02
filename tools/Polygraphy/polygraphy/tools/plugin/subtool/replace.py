#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Replaces a subgraph in an onnx model with a plugin.
"""

from polygraphy import mod
from polygraphy.logger import G_LOGGER
from polygraphy.tools import Tool
from polygraphy.tools.args import (
    DataLoaderArgs,
    OnnxLoadArgs,
    ModelArgs,
    OnnxInferShapesArgs,
)
import os

# Your tool should lazily import any external dependencies. By doing so,
# we avoid creating hard dependencies on other packages.
# Additionally, this allows Polygraphy to automatically install required packages
# as they are needed, instead of requiring the user to do so up front.

gs = mod.lazy_import("onnx_graphsurgeon>=0.5.0")
onnx = mod.lazy_import("onnx")
yaml = mod.lazy_import("yaml", pkg_name="pyyaml")


def replace_with_plugin(graph, op, inputs, outputs, attrs=None):
    """
    replaces a subgraph with a plugin
    """

    # Disconnect output nodes of all input tensors
    for inp in inputs:
        inp.outputs.clear()

    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()

    # Insert the new node.
    new_node = graph.layer(op=op, inputs=inputs, outputs=outputs, attrs=attrs)

    # Remove the now-dangling subgraph.
    graph.cleanup().toposort()

    return new_node


class Replace(Tool):
    # Polygraphy will use the docstring of the tool child class to generate
    # the summary for the command-line help output.
    """
    Replace a subgraph in an onnx model with a plugin.
    """

    def __init__(self):
        super().__init__("replace")

    def get_subscriptions_impl(self):
        return [
            ModelArgs(model_opt_required=True),
            OnnxInferShapesArgs(),
            OnnxLoadArgs(),
            DataLoaderArgs(),
        ]

    def add_parser_args_impl(self, parser):
        parser.add_argument("--plugin-dir", help="Plugin directory.", required=True)
        parser.add_argument(
            "-o", "--output", help="Where to save the modified model", required=True
        )
        parser.add_argument("--config", help="location of config.yaml.")

    def run_impl(self, args):
        graph = gs.import_onnx(self.arg_groups[OnnxLoadArgs].load_onnx())
        tmap = graph.tensors()
        config_yaml = os.path.join(os.path.dirname(args.model_file), "config.yaml")
        if args.config:
            config_yaml = args.config

        with open(config_yaml, "r") as stream:
            in_yaml = yaml.safe_load_all(stream)

            for plugin in in_yaml:
                plugin_name = plugin["name"]
                for instance in plugin["instances"]:
                    inputs = [tmap[tensor_name] for tensor_name in instance["inputs"]]
                    outputs = [tmap[tensor_name] for tensor_name in instance["outputs"]]
                    attrs = instance["attributes"]

                    replace_with_plugin(
                        graph=graph,
                        op=plugin_name,
                        inputs=inputs,
                        outputs=outputs,
                        attrs=attrs,
                    )

        onnx.save(gs.export_onnx(graph), args.output)
