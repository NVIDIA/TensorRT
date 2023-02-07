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

"""
Demonstrates TensorRT capabilities with networks located in HuggingFace repository.
Requires Python 3.5+
"""

import os
import sys
import pickle
import argparse
import importlib

from abc import abstractmethod
from typing import List

# tabulate
from tabulate import tabulate

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# Wrapper actions supported
WRAPPER_RUN_ACTION = "run"
WRAPPER_LIST_ACTION = "list"
WRAPPER_COMPARE_ACTION = "compare"
WRAPPER_BENCHMARK_ACTION = "benchmark"
WRAPPER_ACTIONS = [WRAPPER_RUN_ACTION, WRAPPER_LIST_ACTION, WRAPPER_COMPARE_ACTION, WRAPPER_BENCHMARK_ACTION]

# NNDF
from NNDF.general_utils import process_per_result_entries, process_results, register_network_folders, RANDOM_SEED
from NNDF.logger import G_LOGGER

# huggingface
from transformers import set_seed

# Force seed to 42 for reproducibility.
set_seed(RANDOM_SEED)

class Action:
    def __init__(self, networks: List[str], parser: argparse.ArgumentParser):
        self.networks = networks
        self.parser = parser
        self.add_args(self.parser)

    @abstractmethod
    def execute(self, args: argparse.Namespace):
        pass

    @abstractmethod
    def add_args(self, parser: argparse.ArgumentParser):
        pass


class NetworkScriptAction(Action):

    # Reserved files names for each network folder
    FRAMEWORKS_SCRIPT_NAME = "frameworks"
    TRT_SCRIPT_NAME = "trt"
    ONNX_SCRIPT_NAME = "onnxrt"
    PER_NETWORK_SCRIPTS = [FRAMEWORKS_SCRIPT_NAME, TRT_SCRIPT_NAME, ONNX_SCRIPT_NAME]

    def add_args(self, parser):
        network_group = parser.add_argument_group("specify network")
        network_group.add_argument(
            "network", help="Network to run.", choices=self.networks
        )

    def load_script(self, script_name: str, args: argparse.Namespace):
        """Helper for loading a specific script for given network."""
        assert (
            script_name in self.PER_NETWORK_SCRIPTS
        ), "Script must be a reserved name."

        # Load the specific commandline script
        return importlib.import_module("{}.{}".format(args.network, script_name))


class RunAction(NetworkScriptAction):
    def execute(self, args: argparse.Namespace):
        module = self.load_script(args.script, args)
        module.RUN_CMD._parser = self.parser

        old_path = os.getcwd()
        # Execute script in each relevant folder
        try:
            os.chdir(args.network)
            results = module.RUN_CMD()
        finally:
            os.chdir(old_path)

        # Output to terminal
        print(results)

        # Dump results as a pickle file if applicable.
        # Useful for testing or post-processing.
        if args.save_output_fpath:
            with open(args.save_output_fpath, "wb") as f:
                pickle.dump(results, f)

        return 0

    def add_args(self, parser: argparse.ArgumentParser):
        super().add_args(parser)
        run_group = parser.add_argument_group("run args")
        run_group.add_argument("script", choices=self.PER_NETWORK_SCRIPTS)
        run_group.add_argument("--save-output-fpath", "-o", default=None, help="Outputs a pickled NetworkResult object. See networks.py for definition.")


class BenchmarkAction(NetworkScriptAction):
    def execute(self, args: argparse.Namespace):
        module = self.load_script(args.script, args)
        module.RUN_CMD._parser = self.parser

        old_path = os.getcwd()
        # Execute script in each relevant folder
        try:
            os.chdir(args.network)
            results = module.RUN_CMD.run_benchmark()
        finally:
            os.chdir(old_path)

        # Output to terminal
        print(results)

        return 0

    def add_args(self, parser: argparse.ArgumentParser):
        super().add_args(parser)
        run_group = parser.add_argument_group("benchmark args")
        run_group.add_argument("script", choices=self.PER_NETWORK_SCRIPTS)


class CompareAction(NetworkScriptAction):
    GENERAL_HEADERS = ["script", "accuracy"]

    def execute(self, args: argparse.Namespace):
        compare_group = []
        if args.compare is None:
            compare_group = self.PER_NETWORK_SCRIPTS
        else:
            compare_group = args.compare

        if len(compare_group) <= 1:
            G_LOGGER.error(
                "Comparison command must have atleast two groups to compare to."
            )
            exit()

        results = []
        # Get the parser for inference script which is a superset
        module = None
        try:
            module = self.load_script(self.TRT_SCRIPT_NAME, args)
        except ModuleNotFoundError as e:
            print("Unable to do comparison. TRT script not yet supported.")
            exit(1)

        nconfig = module.RUN_CMD.config
        nconfig.MetadataClass.add_inference_args(self.parser)
        self.parser.parse_known_args()

        results = []
        # It is possible certain scripts are not implemented
        # Allow the results to generate even if script does not exist.
        modified_compare_group = []
        for g in compare_group:
            cwd = os.getcwd()
            try:
                print()
                print("Collecting Data for {}".format(g))
                os.chdir(args.network)
                module = self.load_script(g, args)
                module.RUN_CMD._parser = self.parser
                results.append(module.RUN_CMD())
                modified_compare_group.append(g)
            except ModuleNotFoundError as e:
                print("{} is not valid, the demo does not support this script yet. Ignoring.".format(g))

            finally:
                os.chdir(cwd)

        headers, rows = process_per_result_entries(modified_compare_group, results)
        # Rows are grouped by input, flatten to show as one large table
        flattened_rows = [r for input_row in rows.values() for r in input_row]
        print()
        print(tabulate(flattened_rows, headers=headers))

        headers, rows = process_results(modified_compare_group, results, nconfig)
        print()
        print(tabulate(rows, headers=headers))

        return 0

    def add_args(self, parser: argparse.ArgumentParser):
        super().add_args(parser)
        compare_group = parser.add_argument_group("compare args")
        compare_group.add_argument(
            "--compare",
            "-c",
            nargs="+",
            default=None,
            choices=self.PER_NETWORK_SCRIPTS,
            help="Specific frameworks to compare. If none is specified, all are compared.",
        )


class ListAction(Action):
    def __init__(self, networks: List[str], parser: argparse.ArgumentParser):
        super().__init__(networks, parser)
        self.networks = networks

    def execute(self, args: argparse.Namespace):
        print("Networks that are supported by HuggingFace Demo:")
        [print(n) for n in self.networks]
        return 0


def get_action(
    action_name: str, networks: List[str], parser: argparse.ArgumentParser
) -> Action:
    return {
        WRAPPER_COMPARE_ACTION: CompareAction,
        WRAPPER_LIST_ACTION: ListAction,
        WRAPPER_RUN_ACTION: RunAction,
        WRAPPER_BENCHMARK_ACTION: BenchmarkAction,
    }[action_name](networks, parser)


def get_default_parser(
    networks: List[str], description: str = "", add_default_help=False
) -> argparse.ArgumentParser:
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

    if not add_default_help:
        parser.add_argument(
            "--help",
            "-h",
            help="Shows help message. If --network is supplied, returns help for specific script.",
            action="store_true",
        )
    return parser


def verify_python_version():
    if sys.version_info.major < 3 or sys.version_info.minor <= 6:
        raise RuntimeError("HuggingFace OSS Demo does not support Python <= 3.6 due to end-of-life.")


def main() -> None:
    """
    Parses network folders and responsible for passing --help flags to subcommands if --network is provided.
    """
    # Verify python version support
    verify_python_version()

    # Get all available network scripts
    networks = register_network_folders(os.getcwd())

    # Add network folder for entry point
    description = "Runs TensorRT networks that are based-off of HuggingFace variants."
    parser = get_default_parser(networks, description, add_default_help=False)

    # Get the general network wrapper help
    known_args, _ = parser.parse_known_args()

    # Delegate parser to action specifics
    action = get_action(known_args.action, networks, parser)
    known_args, _ = parser.parse_known_args()
    return action.execute(known_args)


if __name__ == "__main__":
    main()
