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
Demonstrates TensorRT capabilities with networks trained by NeMo.
Requires Python 3.6+
"""

import argparse
import os
import sys
from typing import List, Tuple

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

sys.path.append('../') # Include one-level up directory so to reuse HuggingFace utils.
from HuggingFace.run import (
    Action,
    NetworkScriptAction,
    WRAPPER_LIST_ACTION,
)
from HuggingFace.NNDF.logger import G_LOGGER
from HuggingFace.NNDF.general_utils import register_network_folders
from HuggingFace.NNDF.cuda_bootstrapper import bootstrap_ld_library_path

WRAPPER_RUN_ACTION = "run"
WRAPPER_ACCURACY_ACTION = "accuracy"
WRAPPER_BENCHMARK_ACTION = "benchmark"
WRAPPER_ACTIONS = [WRAPPER_LIST_ACTION, WRAPPER_RUN_ACTION, WRAPPER_ACCURACY_ACTION, WRAPPER_BENCHMARK_ACTION]

class ListAction(Action):
    def __init__(self, networks: List[str], parser: argparse.ArgumentParser):
        super().__init__(networks, parser)
        self.networks = networks

    def execute(self, args: argparse.Namespace):
        print("Networks that are supported by NeMo Demo:")
        [print(n) for n in self.networks]
        return 0

class RunAction(NetworkScriptAction):
    def execute(self, args: argparse.Namespace):
        module = self.load_script(args.script, args)
        module.RUN_CMD._parser = self.parser

        old_path = os.getcwd()
        # Execute script in each relevant folder
        try:
            os.chdir(args.network)
            _ = module.RUN_CMD()
        finally:
            os.chdir(old_path)

        return 0

    def add_args(self, parser: argparse.ArgumentParser):
        super().add_args(parser)
        run_group = parser.add_argument_group("run args")
        run_group.add_argument("script", choices=self.PER_NETWORK_SCRIPTS)

class BenchmarkAction(NetworkScriptAction):
    def execute(self, args: argparse.Namespace):
        module = self.load_script(args.script, args)
        module.RUN_CMD._parser = self.parser

        old_path = os.getcwd()
        # Execute script in each relevant folder
        try:
            os.chdir(args.network)
            _ = module.RUN_CMD()
        finally:
            os.chdir(old_path)

        return 0

    def add_args(self, parser: argparse.ArgumentParser):
        super().add_args(parser)
        benchmarking_group = parser.add_argument_group("benchmark args")
        benchmarking_group.add_argument("script", choices=self.PER_NETWORK_SCRIPTS)
        benchmarking_group.add_argument(
            "--input-seq-len",
            type=int,
            help="Specify fixed input sequence length for perf benchmarking. Required for benchmark except when both input_profile_max and output_profile_max are provided for trt",
        )
        benchmarking_group.add_argument(
            "--output-seq-len",
            type=int,
            help="Specify fixed output sequence length for perf benchmarking. Required for benchmark except when both input_profile_max and output_profile_max are provided for trt",
        )

class AccuracyAction(NetworkScriptAction):
    def execute(self, args: argparse.Namespace):
        module = self.load_script(args.script, args)
        module.RUN_CMD._parser = self.parser

        old_path = os.getcwd()
        # Execute script in each relevant folder
        try:
            os.chdir(args.network)
            _ = module.RUN_CMD()
        finally:
            os.chdir(old_path)

        return 0

    def add_args(self, parser: argparse.ArgumentParser):
        super().add_args(parser)
        accuracy_group = parser.add_argument_group("accuracy args")
        accuracy_group.add_argument("script", choices=self.PER_NETWORK_SCRIPTS)
        accuracy_group.add_argument(
            "--task",
            type=str,
            default="lambada",
            choices=["lambada"],
            help="Specify which task to be used for accuracy check.",
        )

def get_action(
    action_name: str, networks: List[str], parser: argparse.ArgumentParser
) -> Action:
    return {
        WRAPPER_LIST_ACTION: ListAction,
        WRAPPER_RUN_ACTION: RunAction,
        WRAPPER_BENCHMARK_ACTION: BenchmarkAction,
        WRAPPER_ACCURACY_ACTION: AccuracyAction,
    }[action_name](networks, parser)

def verify_python_version():
    if sys.version_info.major < 3 or sys.version_info.minor <= 6:
        raise RuntimeError("NeMo OSS Demo does not support Python <= 3.6 due to end-of-life.")
    if sys.version_info.major < 3 or sys.version_info.minor < 8 or (sys.version_info.minor == 8 and sys.version_info.micro < 10):
        G_LOGGER.warn("NeMo OSS Demo is not tested for Python < 3.8.10")

def get_default_parser(
    description: str = "", add_default_help=False
) -> Tuple[argparse.ArgumentParser, bool]:
    """
    Returns argparser for use by main(). Allows the ability to toggle default help message with a custom help flag
    so that argparser does not throw SystemExit when --help is passed in. Useful for custom --help functionality.

    Returns:
        (argparse.ArgumentParser): argparser used by main()
    """
    # This variable is set so that usage errors don't show up in wrapper
    parser = argparse.ArgumentParser(
        conflict_handler="resolve",
        description=description,
        add_help=add_default_help,
        prog="run.py",
    )

    required_group = parser.add_argument_group("required wrapper arguments")
    required_group.add_argument("action", choices=WRAPPER_ACTIONS)
    return parser

def main() -> None:
    """
    Parses network folders and responsible for passing --help flags to subcommands if --network is provided.
    """
    # Verify python version support
    verify_python_version()

    # Get all available network scripts
    networks = register_network_folders(os.getcwd())

    # Add network folder for entry point
    description = "Runs TensorRT networks that are based-off of NeMo variants."
    parser = get_default_parser(description)

    # Get the general network wrapper help
    known_args, _ = parser.parse_known_args()

    # Delegate parser to action specifics
    action = get_action(known_args.action, networks, parser)
    known_args, _ = parser.parse_known_args()

    # If bootstrap occurs, then the spawned process completes the rest of demo.
    # We can exit safely. We spawn after parsing basic args to reduce loading churn on rudimentary help commands.
    if bootstrap_ld_library_path():
        sys.exit(0)

    return action.execute(known_args)

if __name__ == "__main__":
    main()
