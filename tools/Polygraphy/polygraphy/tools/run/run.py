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

from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import (
    ComparatorCompareArgs,
    ComparatorRunArgs,
    DataLoaderArgs,
    LoggerArgs,
    ModelArgs,
    OnnxLoaderArgs,
    OnnxrtRunnerArgs,
    OnnxSaveArgs,
    OnnxShapeInferenceArgs,
    PluginRefArgs,
    Tf2OnnxLoaderArgs,
    TfConfigArgs,
    TfLoaderArgs,
    TfRunnerArgs,
    TrtConfigArgs,
    TrtEngineLoaderArgs,
    TrtEngineSaveArgs,
    TrtLegacyArgs,
    TrtNetworkLoaderArgs,
    TrtPluginLoaderArgs,
    TrtRunnerArgs,
)
from polygraphy.tools.base import Tool
from polygraphy.tools.script import Script, inline, safe


# FIXME: This should be moved into tools/args/
def add_runner_args(parser):
    class StoreRunnerOrdered(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if not hasattr(namespace, "runners"):
                namespace.runners = []
            namespace.runners.append(option_string.lstrip("-").replace("-", "_"))

    runner_args = parser.add_argument_group(
        "Runners", "Options for selecting runners. Zero or more runners may be specified"
    )

    def add_runner(option, help):
        runner_args.add_argument(option, help=help, action=StoreRunnerOrdered, dest="runners", default=[], nargs=0)

    add_runner("--trt", help="Run inference using TensorRT")
    add_runner(
        "--trt-legacy",
        help="Run inference using Legacy TensorRT Runner. Only supports networks using implicit batch mode",
    )
    add_runner("--tf", help="Run inference using TensorFlow")
    add_runner("--onnxrt", help="Run inference using ONNX Runtime")
    add_runner(
        "--pluginref",
        help="Run inference for models containing single TensorRT plugins using a CPU reference implementation",
    )


# Generate a summary line to add as a comment to the script
def generate_summary(model_file, runners, load_results):
    def join_list(lst):
        new_list = copy.copy(lst)
        if len(new_list) > 1:
            new_list[-1] = "and {:}".format(new_list[-1])
        return ", ".join(new_list) if len(new_list) > 2 else " ".join(new_list)

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
            "pluginref": "CPU plugin references",
        }
        runners = [runner_names[runner] for runner in runners]
        summary += "between " if len(runners) > 1 else "using "
        summary += join_list(runners) + "."

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
        self.subscribe_args(TfLoaderArgs(tftrt=True))
        self.subscribe_args(TfConfigArgs())
        self.subscribe_args(TfRunnerArgs())
        self.subscribe_args(Tf2OnnxLoaderArgs())
        self.subscribe_args(OnnxSaveArgs(output="save-onnx", short_opt=None))
        self.subscribe_args(OnnxShapeInferenceArgs())
        self.subscribe_args(OnnxLoaderArgs(save=True))
        self.subscribe_args(OnnxrtRunnerArgs())
        self.subscribe_args(PluginRefArgs())
        self.subscribe_args(
            TrtConfigArgs(random_data_calib_warning=False)
        )  # We run calibration with the inference-time data
        self.subscribe_args(TrtPluginLoaderArgs())
        self.subscribe_args(TrtNetworkLoaderArgs())
        self.subscribe_args(TrtEngineSaveArgs(output="save-engine", short_opt=None))
        self.subscribe_args(TrtEngineLoaderArgs(save=True))
        self.subscribe_args(TrtRunnerArgs())
        self.subscribe_args(TrtLegacyArgs())
        self.subscribe_args(DataLoaderArgs())
        self.subscribe_args(ComparatorRunArgs())
        self.subscribe_args(ComparatorCompareArgs())

    def add_parser_args(self, parser):
        parser.add_argument(
            "--gen",
            "--gen-script",
            help="Path to save a generated Python script, that will do exactly "
            "what `run` would. When this option is enabled, `run` will just save the script and exit. "
            "Use `-` to print the script to the standard output",
            type=argparse.FileType("w"),
            dest="gen_script",
        )
        add_runner_args(parser)

    def run(self, args):
        if self.arg_groups[ModelArgs].model_file is None and args.runners:
            G_LOGGER.critical(
                "One or more runners was specified, but no model file was provided. Make sure you've specified the model path, "
                "and also that it's not being consumed as an argument for another parameter"
            )

        script = self.build_script(args)

        if args.gen_script:
            script.save(args.gen_script)
        else:
            exec(str(script))

    # Generates a script based on command-line arguments
    def build_script(self, args):
        script = Script(
            summary=generate_summary(self.arg_groups[ModelArgs].model_file, args.runners, args.load_results)
        )

        self.arg_groups[LoggerArgs].add_to_script(script)

        if not args.runners:
            G_LOGGER.warning("No runners have been selected. Inference will not be run!")

        for runner_arg in args.runners:
            add_runner_func = {
                "tf": self.arg_groups[TfRunnerArgs].add_to_script,
                "onnxrt": self.arg_groups[OnnxrtRunnerArgs].add_to_script,
                "trt": self.arg_groups[TrtRunnerArgs].add_to_script,
                "trt_legacy": self.arg_groups[TrtLegacyArgs].add_to_script,
                "pluginref": self.arg_groups[PluginRefArgs].add_to_script,
            }[runner_arg]
            add_runner_func(script)

        RESULTS_VAR_NAME = self.arg_groups[ComparatorRunArgs].add_to_script(script)
        SUCCESS_VAR_NAME = self.arg_groups[ComparatorCompareArgs].add_to_script(script, results_name=RESULTS_VAR_NAME)

        script.add_import(imports=["sys"])

        cmd_run = inline(safe("' '.join(sys.argv)"))
        exit_status = safe(
            "# Report Results\n"
            "cmd_run = {cmd}\n"
            "if not {success}:\n"
            '\tG_LOGGER.critical("FAILED | Command: {{}}".format(cmd_run))\n'
            'G_LOGGER.finish("PASSED | Command: {{}}".format(cmd_run))\n',
            cmd=cmd_run,
            success=SUCCESS_VAR_NAME,
        )
        script.append_suffix(exit_status)

        return script
