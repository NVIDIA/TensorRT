#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
common_backend = mod.lazy_import("polygraphy.backend.common")

def default_replace_with_plugin(graph, input_tensors: list, output_tensors: list, attrs=None, op=None):
    """
    replaces a subgraph (set of nodes) with a single plugin node
    default method to be used when the plugin does not specify a custom replacement method
    """

    def issubset_unhashable(list_a: list, list_b: list) -> bool:
        """
        Return whether list_a is a subset (or equal to) list_b
        The objects in list_a and list_b are unhashable, otherwise set(list_a) <= set(list_b) is enough
        """
        return len(list_a) <= len(list_b) and all(a in list_b for a in list_a)

    # Disconnect those output nodes of the input tensors whose inputs are a subset of the input tensors
    for in_tensor in input_tensors:
        to_remove_nodes = []
        for out_node in in_tensor.outputs:
            if issubset_unhashable(out_node.inputs, input_tensors):
                to_remove_nodes.append(out_node)
        for node in to_remove_nodes:
            in_tensor.outputs.remove(node)

    # Disconnet input nodes of all output tensors
    for out_tensor in output_tensors:
        to_remove_nodes = []
        for in_node in out_tensor.inputs:
            to_remove_nodes.append(in_node)
        for node in to_remove_nodes:
            out_tensor.inputs.remove(node)

    # Insert the new node
    new_node = graph.layer(op=op, inputs=input_tensors, outputs=output_tensors, attrs=attrs)
    graph.cleanup().toposort()
    return new_node[0].inputs[0]

class Replace(Tool):
    # Polygraphy will use the docstring of the tool child class to generate
    # the summary for the command-line help output.
    """
    Replace a subgraph in an onnx model with a plugin.
    """
    GRAPH_PATTERN_FILE_NAME="pattern.py"

    def __init__(self):
        super().__init__(name="replace")

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
            "-o", "--output", help="Where to save the modified model", required=False
        )
        parser.add_argument("--config", help="location of config.yaml.")

    def run_impl(self, args):
        self.replace_plugin(
            model_file=args.model_file,
            plugin_dir=args.plugin_dir,
            output=args.output,
            config=args.config
        )
    def replace_plugin(self, model_file, plugin_dir, output=None, config=None):
        graph = gs.import_onnx(self.arg_groups[OnnxLoadArgs].load_onnx()) if self.arg_groups else gs.import_onnx(onnx.load(model_file))

        tensor_map = graph.tensors()
        config_yaml = config or os.path.join(os.path.dirname(model_file), "config.yaml")            
        
        plugin_dir = os.path.abspath(plugin_dir)
        
        with open(config_yaml, "r") as stream:
            in_yaml = yaml.safe_load_all(stream)

            for plugin in in_yaml:
                plugin_name = plugin["name"]
                plugin_op = plugin["op"]
                G_LOGGER.ultra_verbose(f"replacing {plugin_name}...")
                plugin_pattern_loc = os.path.join(plugin_dir, plugin_name, self.GRAPH_PATTERN_FILE_NAME)
                # if the plugin provides a custom replacement method, use that                    
                replace_fn = default_replace_with_plugin
                try:
                    replace_fn = mod.import_from_script(
                        plugin_pattern_loc,
                        "replace_with_plugin"
                    )
                except:
                    pass

                replace_cnt = 0
                for instance in plugin["instances"]:
                    attrs = instance.get("attributes", None)
                    if replace_fn(
                            graph=graph,
                            input_tensors=[tensor_map[ip_tensor_name] for ip_tensor_name in instance["inputs"]],
                            output_tensors=[tensor_map[op_tensor_name] for op_tensor_name in instance["outputs"]],
                            attrs=attrs,
                            op=plugin_op
                    ):
                        replace_cnt += 1
                G_LOGGER.info(f"replaced {replace_cnt} instances of {plugin_name} plugin")
                if replace_cnt != len(plugin['instances']):
                    G_LOGGER.warning(f"Warning: not all instances of {plugin_name} were replaced!")

        output_onnx = output or os.path.join(os.path.dirname(model_file), "replaced.onnx")

        onnx.save(gs.export_onnx(graph), output_onnx)
