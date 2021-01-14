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
import copy
import os

import polygraphy
from polygraphy.common import constants
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import (ComparatorCompareArgs, ComparatorRunArgs,
                                   DataLoaderArgs, LoggerArgs, ModelArgs,
                                   OnnxLoaderArgs, OnnxrtRunnerArgs,
                                   OnnxtfRunnerArgs, Tf2OnnxLoaderArgs,
                                   TfConfigArgs, TfLoaderArgs, TfRunnerArgs,
                                   TrtLegacyArgs, TrtLoaderArgs, TrtRunnerArgs)
from polygraphy.tools.base import Tool
from polygraphy.tools.util import misc as tools_util
from polygraphy.tools.util.script import Inline, Script
from polygraphy.util import misc


# FIXME: This should be moved into tools/args/
def add_runner_args(parser):
    class StoreRunnerOrdered(argparse.Action):
         def __call__(self, parser, namespace, values, option_string=None):
            if not hasattr(namespace, "runners"):
                namespace.runners = []
            namespace.runners.append(option_string.lstrip("-").replace("-", "_"))

    runner_args = parser.add_argument_group("Runners", "Options for selecting runners. Zero or more runners may be specified")

    def add_runner(option, help):
        runner_args.add_argument(option, help=help, action=StoreRunnerOrdered, dest="runners", default=[], nargs=0)

    add_runner("--trt", help="Run inference using TensorRT")
    add_runner("--trt-legacy", help="Run inference using Legacy TensorRT Runner. Only supports networks using implicit batch mode")
    add_runner("--tf", help="Run inference using TensorFlow")
    add_runner("--onnxrt", help="Run inference using ONNX Runtime")
    add_runner("--onnxtf", help="Run inference using the ONNX-TensorFlow Backend")


# Generate a summary line to add as a comment to the script
def generate_summary(model_file, runners, load_results):
    def join_list(lst):
        new_list = copy.copy(lst)
        if len(new_list) > 1:
            new_list[-1] = "and {:}".format(new_list[-1])
        return ", ".join(new_list)

    summary = ""

    if runners:
        summary += "This script "
        if len(runners) > 1:
            summary += "compares "
        else:
            summary += "runs "
        if model_file:
            summary += "{:} ".format(model_file)

        runner_names = {
            "trt": "TensorRT",
            "trt_legacy": "TensorRT Legacy",
            "tf": "TensorFlow",
            "onnxrt": "ONNX Runtime",
            "onnxtf": "ONNX-TensorFlow Backend",
            "cntk": "CNTK"
        }
        runners = [runner_names[runner] for runner in runners]
        summary += "between " if len(runners) > 1 else "using "
        summary += join_list(runners)

    if load_results:
        summary += "\nIt will check against outputs stored in {:}\n".format(join_list(load_results))

    return summary


################################# TOOL #################################


class Run(Tool):
    """
    Run inference and compare results across backends.
    """
    def __init__(self):
        super().__init__("run")
        self.subscribe_args(ModelArgs())
        self.subscribe_args(TfLoaderArgs())
        self.subscribe_args(TfConfigArgs())
        self.subscribe_args(TfRunnerArgs())
        self.subscribe_args(Tf2OnnxLoaderArgs())
        self.subscribe_args(OnnxLoaderArgs())
        self.subscribe_args(OnnxrtRunnerArgs())
        self.subscribe_args(OnnxtfRunnerArgs())
        self.subscribe_args(TrtLoaderArgs(network_api=True))
        self.subscribe_args(TrtRunnerArgs())
        self.subscribe_args(TrtLegacyArgs())
        self.subscribe_args(DataLoaderArgs())
        self.subscribe_args(ComparatorRunArgs())
        self.subscribe_args(ComparatorCompareArgs())


    def add_parser_args(self, parser):
        parser.add_argument("--gen", "--gen-script", help="Path to save a generated Python script, that will do exactly "
                            "what `run` would. When this option is enabled, `run` will just save the script and exit. "
                            "Use `-` to print the script to the standard output",
                            type=argparse.FileType("w"), dest="gen_script")
        add_runner_args(parser)


    def run(self, args):
        if self.makers[TrtLoaderArgs].network_api and not tools_util.get(args, "gen_script"):
            G_LOGGER.critical("Cannot use the --network-api option if --gen/--gen-script is not being used.")
        elif self.makers[TrtLoaderArgs].network_api and "trt" not in args.runners:
            args.runners.append("trt")

        if self.makers[ModelArgs].model_file is None and args.runners and self.makers[TrtLoaderArgs].network_api is None:
            G_LOGGER.critical("One or more runners was specified, but no model file was provided. Make sure you've specified the model path, "
                            "and also that it's not being consumed as an argument for another parameter")


        misc.log_module_info(polygraphy)

        script = self.build_script(args)

        if args.gen_script:
            with args.gen_script:
                args.gen_script.write(script)

                path = args.gen_script.name
                # Somehow, piping fools isatty, e.g. `polygraphy run --gen-script - | cat`
                if not args.gen_script.isatty() and path not in ["<stdout>", "<stderr>"]:
                    G_LOGGER.info("Writing script to: {:}".format(path))
                    # Make file executable
                    os.chmod(path, os.stat(path).st_mode | 0o111)
        else:
            exec(script)

        return 0


    # Generates a script based on command-line arguments
    def build_script(self, args):
        script = Script(summary=generate_summary(self.makers[ModelArgs].model_file, args.runners, args.load_results))

        self.makers[LoggerArgs].add_to_script(script)

        data_loader_name = self.makers[DataLoaderArgs].add_to_script(script)

        for runner_arg in args.runners:
            add_runner_func = {
                "tf": self.makers[TfRunnerArgs].add_to_script,
                "onnxrt": self.makers[OnnxrtRunnerArgs].add_to_script,
                "onnxtf": self.makers[OnnxtfRunnerArgs].add_to_script,
                "trt": lambda script: self.makers[TrtRunnerArgs].add_to_script(script, data_loader_name),
                "trt_legacy": self.makers[TrtLegacyArgs].add_to_script,
            }[runner_arg]
            add_runner_func(script)

        RESULTS_VAR_NAME = self.makers[ComparatorRunArgs].add_to_script(script, data_loader_name=data_loader_name)
        SUCCESS_VAR_NAME = self.makers[ComparatorCompareArgs].add_to_script(script, results_name=RESULTS_VAR_NAME)

        cmd_run = Inline("' '.join(sys.argv)")
        script.append_suffix(Script.format_str('# Report Results\ncmd_run={cmd}\nif {success}:\n{tab}G_LOGGER.finish("PASSED | Command: {{}}".format(cmd_run))\nelse:\n{tab}G_LOGGER.error("FAILED | Command: {{}}".format(cmd_run))', cmd=cmd_run, success=SUCCESS_VAR_NAME, tab=Inline(constants.TAB)))
        script.append_suffix("sys.exit(0 if {success} else 1)".format(success=SUCCESS_VAR_NAME))

        return str(script)
