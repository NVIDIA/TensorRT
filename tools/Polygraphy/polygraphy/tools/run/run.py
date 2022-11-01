#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import argparse
import copy
from textwrap import dedent

from polygraphy import constants, mod
from polygraphy.exception import PolygraphyException
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import (
    ComparatorCompareArgs,
    ComparatorPostprocessArgs,
    ComparatorRunArgs,
    CompareFuncIndicesArgs,
    CompareFuncSimpleArgs,
    DataLoaderArgs,
    LoggerArgs,
    ModelArgs,
    OnnxFromTfArgs,
    OnnxInferShapesArgs,
    OnnxLoadArgs,
    OnnxrtRunnerArgs,
    OnnxrtSessionArgs,
    OnnxSaveArgs,
    PluginRefRunnerArgs,
    RunnerSelectArgs,
    TfConfigArgs,
    TfLoadArgs,
    TfRunnerArgs,
    TfTrtArgs,
    TrtConfigArgs,
    TrtLegacyRunnerArgs,
    TrtLoadEngineArgs,
    TrtLoadNetworkArgs,
    TrtLoadPluginsArgs,
    TrtRunnerArgs,
    TrtSaveEngineArgs,
)
from polygraphy.tools.base import Tool
from polygraphy.tools.script import Script, safe

try:
    # No need to lazy import since this is part of the standard library
    from importlib import metadata
except:
    # importlib.metadata may not exist in older versions of Python.
    metadata = mod.lazy_import("importlib_metadata")


PLUGIN_ENTRY_POINT = "polygraphy.run.plugins"


def generate_summary(model_file, runners, load_results):
    def join_list(lst):
        new_list = copy.copy(list(lst))
        if len(new_list) > 1:
            new_list[-1] = f"and {new_list[-1]}"
        return ", ".join(new_list) if len(new_list) > 2 else " ".join(new_list)

    summary = ""

    if runners:
        summary += "This script "
        if len(runners) > 1:
            summary += "compares "
        else:
            summary += "runs "
        if model_file:
            summary += f"{model_file} "

        summary += "between " if len(runners) > 1 else "using "
        summary += join_list(runners) + "."

    if load_results:
        summary += f"\nIt will check against outputs stored in {join_list(load_results)}\n"

    return summary


class Run(Tool):
    """
    Run inference and compare results across backends.

    The typical usage of `run` is:

        polygraphy run [model_file] [runners...] [runner_options...]

    `run` will then run inference on the specified model with all the specified runners
    and compare inference outputs between them.

    TIP: You can use `--gen-script` to generate a Python script that does exactly what the `run`
    command would otherwise do.
    """

    def __init__(self):
        super().__init__("run")

    def get_subscriptions_impl(self):
        deps = [
            RunnerSelectArgs(),
            ModelArgs(guess_model_type_from_runners=True),
            TfTrtArgs(),
            TfLoadArgs(allow_tftrt=True),
            TfConfigArgs(),
            TfRunnerArgs(),
            OnnxFromTfArgs(),
            OnnxSaveArgs(output_opt="save-onnx", output_short_opt=False),
            OnnxInferShapesArgs(),
            OnnxLoadArgs(allow_saving=True, allow_from_tf=True),
            OnnxrtSessionArgs(),
            OnnxrtRunnerArgs(),
            PluginRefRunnerArgs(),
            # We run calibration/inference with the same data, so it doesn't really matter if it's random.
            TrtConfigArgs(allow_random_data_calib_warning=False),
            TrtLoadPluginsArgs(),
            TrtLoadNetworkArgs(),
            TrtSaveEngineArgs(output_opt="save-engine", output_short_opt=False),
            TrtLoadEngineArgs(allow_saving=True),
            TrtRunnerArgs(),
            TrtLegacyRunnerArgs(),
            DataLoaderArgs(),
            ComparatorRunArgs(),
            ComparatorPostprocessArgs(),
            ComparatorCompareArgs(),
            CompareFuncSimpleArgs(),
            CompareFuncIndicesArgs(),
        ]

        # Initialize plugins
        self.loaded_plugins = []
        try:
            entry_points = metadata.entry_points()
        except PolygraphyException as err:
            G_LOGGER.warning(
                f"Could not load extension modules since `importlib.metadata` and `importlib_metadata` are missing."
            )
        else:
            if isinstance(entry_points, dict):
                # For compatibility with older versions of importlib_metadata
                plugins = entry_points.get(PLUGIN_ENTRY_POINT, [])
            else:
                entry_points = entry_points.select(group=PLUGIN_ENTRY_POINT)
                plugins = [entry_points[name] for name in entry_points.names]

            for plugin in plugins:
                try:
                    get_arg_groups_func = plugin.load()
                    plugin_arg_groups = get_arg_groups_func()
                except Exception as err:
                    G_LOGGER.warning(f"Failed to load plugin: {plugin.name}.\nNote: Error was:\n{err}")
                else:
                    deps.extend(plugin_arg_groups)
                    self.loaded_plugins.append(plugin.name)
        return deps

    def add_parser_args_impl(self, parser):
        parser.add_argument(
            "--gen",
            "--gen-script",
            help="Path to save a generated Python script, that will do exactly "
            "what `run` would. When this option is enabled, `run` will save the script and exit. "
            "Use a value of `-` to print the script to the standard output instead of saving it to a file",
            type=argparse.FileType("w"),
            dest="gen_script",
        )

    def show_start_end_logging_impl(self, args):
        # No need to print start/end messages when we're just creating a script
        return not args.gen_script

    def run_impl(self, args):
        G_LOGGER.verbose(f"Loaded extension modules: {self.loaded_plugins}")

        if self.arg_groups[ModelArgs].path is None and self.arg_groups[RunnerSelectArgs].runners:
            G_LOGGER.critical(
                "One or more runners was specified, but no model file was provided. Make sure you've specified the model path, "
                "and also that it's not being consumed as an argument for another parameter"
            )

        script = Script(
            summary=generate_summary(
                self.arg_groups[ModelArgs].path,
                list(self.arg_groups[RunnerSelectArgs].runners.values()),
                self.arg_groups[ComparatorCompareArgs].load_outputs_paths,
            )
        )

        self.arg_groups[LoggerArgs].add_to_script(script)

        self.arg_groups[RunnerSelectArgs].add_to_script(script)

        RESULTS_VAR_NAME = self.arg_groups[ComparatorRunArgs].add_to_script(script)
        SUCCESS_VAR_NAME = self.arg_groups[ComparatorCompareArgs].add_to_script(script, results_name=RESULTS_VAR_NAME)

        script.add_import(imports=["PolygraphyException"], frm="polygraphy.exception")
        exit_status = safe(
            dedent(
                f"""
                # Report Results
                if not {{success}}:
                {constants.TAB}raise PolygraphyException('FAILED')"""
            ),
            success=SUCCESS_VAR_NAME,
        )
        script.append_suffix(exit_status)

        if args.gen_script:
            script.save(args.gen_script)
        else:
            exec(str(script))
