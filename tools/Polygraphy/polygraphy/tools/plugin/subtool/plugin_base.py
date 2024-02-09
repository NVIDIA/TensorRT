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
Analyzes onnx model for potential plugin substitutions.
"""

import glob
from polygraphy import mod
from polygraphy.logger import G_LOGGER
from polygraphy.tools import Tool
from polygraphy.tools.args import DataLoaderArgs, OnnxLoadArgs, ModelArgs, OnnxInferShapesArgs
import os

# Your tool should lazily import any external dependencies. By doing so,
# we avoid creating hard dependencies on other packages.
# Additionally, this allows Polygraphy to automatically install required packages
# as they are needed, instead of requiring the user to do so up front.
common_backend = mod.lazy_import("polygraphy.backend.common")
gs = mod.lazy_import("onnx_graphsurgeon>=0.5.0")
np = mod.lazy_import("numpy")
onnx = mod.lazy_import("onnx")
yaml = mod.lazy_import("yaml", pkg_name="pyyaml")

class PluginBase(Tool):
    """
    Analyze an onnx model for potential plugin substitutions.
    """
    GRAPH_PATTERN_FILE_NAME="pattern.py"

    def __init__(self, name=None):
        super().__init__(name)
        self.plugin_dir = None

    def get_subscriptions_impl(self):
        return [
            ModelArgs(model_opt_required=True),
            OnnxInferShapesArgs(),
            OnnxLoadArgs(),
            DataLoaderArgs(),
        ]

    def add_parser_args_impl(self, parser):
        parser.add_argument("--plugin-dir", help="Plugin directory.", required=True)
        include_exclude = parser.add_mutually_exclusive_group()
        include_exclude.add_argument("--include", help="Names of plugins to include. Format: `--include <plugin_name0> <plugin_name1> ...`", required=False, nargs="+", type=str, default=[])
        include_exclude.add_argument("--exclude", help="Names of plugins to exclude. Format: `--exclude <plugin_name0> <plugin_name1> ...`", required=False, nargs="+", type=str, default=[])

    def run_impl(self, args):
        raise NotImplementedError("run_impl() must be implemented by child classes")

    def match_plugin(self, args, list_plugins=False):

        self.plugin_dir = os.path.abspath(args.plugin_dir)
        full_pattern = os.path.join(self.plugin_dir, "*", self.GRAPH_PATTERN_FILE_NAME)

        plugin_set = {os.path.basename(os.path.dirname(x)) for x in glob.glob(pathname=full_pattern, recursive=False)}

        if args.include:
            plugin_set.intersection_update(set(args.include))

        if args.exclude:
            plugin_set.difference_update(set(args.exclude))

        graph = gs.import_onnx(self.arg_groups[OnnxLoadArgs].load_onnx())

        # list of plugin substitution instances (conent of config.yaml)
        out_yaml = []
        plugin_frequency = dict.fromkeys(plugin_set, 0)

        # for each plugin, see if there is any match in the onnx model
        for plugin in plugin_set:
            G_LOGGER.info(f"checking {plugin} in model")
            plugin_yaml = {}

            #build pattern from plugin
            plugin_pattern_loc = os.path.join(self.plugin_dir, plugin, self.GRAPH_PATTERN_FILE_NAME)
            graph_pattern = common_backend.invoke_from_script(plugin_pattern_loc, "get_plugin_pattern")

            matched_subgraphs = graph_pattern.match_all(graph)
            if matched_subgraphs:
                plugin_frequency[plugin] += len(matched_subgraphs)

            plugin_yaml["name"] = plugin
            plugin_yaml["instances"] = []

            for sg in matched_subgraphs:
                def get_names(tensors):
                    return [tensor.name for tensor in tensors]

                inputs = get_names(sg.inputs)
                outputs = get_names(sg.outputs)
                attributes = common_backend.invoke_from_script(plugin_pattern_loc, "get_plugin_attributes", sg)
                plugin_yaml["instances"].append({
                    "inputs": inputs,
                    "outputs": outputs,
                    "attributes": attributes
                })

            out_yaml.append(plugin_yaml)

        if list_plugins:
            G_LOGGER.info("the following plugins would be used:")
            G_LOGGER.info(plugin_frequency)
            return

        config_yaml = os.path.join(os.path.dirname(args.model_file),"config.yaml")
        if args.output:
            config_yaml = args.output

        with open(config_yaml, "w") as stream:
            yaml.dump_all(
                out_yaml,
                stream,
                default_flow_style=False,
                sort_keys=False
            )
