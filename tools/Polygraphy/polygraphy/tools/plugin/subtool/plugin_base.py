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
Analyzes onnx model for potential plugin substitutions.
"""

import glob
from polygraphy import mod
from polygraphy.logger import G_LOGGER
from polygraphy.tools import Tool
from polygraphy.tools.args import util as args_util
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
common_backend = mod.lazy_import("polygraphy.backend.common")
gs = mod.lazy_import("onnx_graphsurgeon>=0.5.0")
np = mod.lazy_import("numpy")
onnx = mod.lazy_import("onnx")
yaml = mod.lazy_import("yaml", pkg_name="pyyaml")


class PluginBase(Tool):
    """
    Analyze an onnx model for potential plugin substitutions.
    """

    GRAPH_PATTERN_FILE_NAME = "pattern.py"

    def __init__(self, list_plugins:bool, name=None):
        super().__init__(name)
        self.list_plugins = list_plugins

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
        include_exclude.add_argument(
            "--include",
            help="Names of plugins to include. Format: `--include <plugin_name0> <plugin_name1> ...`",
            required=False,
            nargs="+",
            type=str,
            default=[],
        )
        include_exclude.add_argument(
            "--exclude",
            help="Names of plugins to exclude. Format: `--exclude <plugin_name0> <plugin_name1> ...`",
            required=False,
            nargs="+",
            type=str,
            default=[],
        )

    def run_impl(self, args):
        self.match_plugin(
            model_file=args.model_file,
            plugin_dir=args.plugin_dir,
            output_file=args_util.get(args,"output"),
            include_list=args.include,
            exclude_list=args.exclude,
            list_plugins=self.list_plugins
        )

    def match_plugin(self, model_file, plugin_dir, output_file=None, include_list=None, exclude_list=None, list_plugins=False):
        """
        find matching subgraphs based on plugin pattern
        """

        plugin_dir = os.path.abspath(plugin_dir)
        full_pattern = os.path.join(plugin_dir, "*", self.GRAPH_PATTERN_FILE_NAME)

        plugin_set = {
            os.path.basename(os.path.dirname(x))
            for x in glob.glob(pathname=full_pattern, recursive=False)
        }

        if include_list:
            plugin_set.intersection_update(set(include_list))

        if exclude_list:
            plugin_set.difference_update(set(exclude_list))

        # list of plugin substitution instances (conent of config.yaml)
        out_yaml = []
        plugin_frequency = dict.fromkeys(plugin_set, 0)

        # for each plugin, see if there is any match in the onnx model
        for plugin in plugin_set:
            G_LOGGER.info(f"checking {plugin} in model")
            plugin_yaml = {}

            plugin_pattern_loc = os.path.join(plugin_dir, plugin, self.GRAPH_PATTERN_FILE_NAME)
            # create a new graph in every iteration, in case the pattern matching modifies the graph
            graph = gs.import_onnx(self.arg_groups[OnnxLoadArgs].load_onnx()) if self.arg_groups else gs.import_onnx(onnx.load(model_file))

            #get inputs, outputs, attributes from plugin
            G_LOGGER.ultra_verbose(f"calling get_matching_subgraphs from {plugin_pattern_loc}")
            ioattrs = common_backend.invoke_from_script(plugin_pattern_loc, "get_matching_subgraphs", graph)

            if ioattrs:
                G_LOGGER.ultra_verbose("match found")
                plugin_yaml["name"] = common_backend.invoke_from_script(plugin_pattern_loc, "get_plugin_metadata")['name']
                plugin_yaml["op"] = common_backend.invoke_from_script(plugin_pattern_loc, "get_plugin_metadata")['op']
                plugin_yaml["instances"] = ioattrs
                out_yaml.append(plugin_yaml)
                plugin_frequency[plugin] += len(ioattrs)

        G_LOGGER.info("the following plugins matched:")
        G_LOGGER.info(plugin_frequency)
        if list_plugins:
            return

        config_yaml = output_file or os.path.abspath(os.path.join(os.path.dirname(model_file),"config.yaml"))

        with open(config_yaml, "w") as stream:
            yaml.dump_all(out_yaml, stream, default_flow_style=False, sort_keys=False)
        G_LOGGER.info(f"Matching subgraphs saved to {config_yaml}")
