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
import os
import sys

from polygraphy import mod, util
from polygraphy.common import TensorMetadata
from polygraphy.comparator import AccuracyResults, RunResults, StreamingDataLoader
from polygraphy.comparator import util as comp_util
from polygraphy.json import load_json, serde
from polygraphy.logger import G_LOGGER
from polygraphy.tools.base import Tool

np = mod.lazy_import("numpy")


class Data(Tool):
    """
    Display information about inference inputs and outputs saved from Polygraphy's Comparator.run()
    (for example, outputs saved by `--save-outputs` or inputs saved by `--save-inputs` from `polygraphy run`).
    """

    def __init__(self):
        super().__init__("data")

    def add_parser_args(self, parser):
        parser.add_argument(
            "path",
            help="Path to input or output data saved from Polygraphy. This may be a single file, "
            "or a directory of per-iteration JSON files (as written by an extensionless "
            "--save-inputs/--save-outputs).",
        )
        parser.add_argument(
            "-a",
            "--all",
            help="Show information on all iterations present in the data instead of just the first",
            action="store_true",
        )
        parser.add_argument(
            "-s",
            "--show-values",
            help="Show values of the tensors instead of just metadata",
            action="store_true",
        )
        parser.add_argument(
            "--histogram",
            help="Show a histogram of the value distribution",
            action="store_true",
        )
        parser.add_argument(
            "-n",
            "--num-items",
            help="The number of values to show at the beginning and end of each dimension when printing arrays. "
            "Use a value of -1 to show all elements in the array. "
            "Defaults to 3.",
            type=int,
            default=3,
        )
        parser.add_argument(
            "--line-width",
            help="The number of characters per line when displaying arrays. "
            "Use a value of -1 to insert line breaks only at dimension end points. "
            "Defaults to 75.",
            type=int,
            default=75,
        )

    def run_impl(self, args):
        np.set_printoptions(
            edgeitems=sys.maxsize if args.num_items == -1 else args.num_items,
            linewidth=sys.maxsize if args.line_width == -1 else args.line_width,
        )

        def meta_from_iter_result(iter_result):
            meta = TensorMetadata()
            for name, arr in iter_result.items():
                meta.add(name, dtype=arr.dtype, shape=arr.shape)
            return meta

        def str_from_iters(iters):
            out_str = ""
            for index, iter_result in enumerate(iters):
                iter_meta = meta_from_iter_result(iter_result)
                indent = 1
                if len(iters) > 1 and args.all:
                    out_str += util.indent_block(
                        f"\n-- Iteration: {index}\n", indent - 1
                    )
                    indent = 2

                for name, arr in iter_result.items():
                    out_str += util.indent_block(
                        f"\n{name} {iter_meta[name]} | Stats: {comp_util.str_output_stats(arr)}",
                        indent - 1,
                    )
                    if args.histogram:
                        out_str += f"\n{util.indent_block(comp_util.str_histogram(arr), indent)}"
                    if args.show_values:
                        out_str += f"\n{util.indent_block(str(arr), indent)}"

                if indent == 2:
                    out_str += "\n"
                if not args.all:
                    break
            return out_str

        def display_results(results):
            results_str = ""
            results_str += f"==== Run Results ({len(results)} runners) ====\n\n"

            max_runner_width = max([len(runner_name) for runner_name in results.keys()], default=0)
            for runner_name, iters in results.items():
                results_str += f"---- {runner_name:<{max_runner_width}} ({len(iters)} iterations) ----\n"
                results_str += str_from_iters(iters) + "\n"

            results_str = util.indent_block(results_str, level=0).strip()
            G_LOGGER.info(results_str)

        def display_inputs(input_data):
            inputs_str = ""
            inputs_str += f"==== Data ({len(input_data)} iterations) ====\n"
            inputs_str += str_from_iters(input_data) + "\n"
            inputs_str = util.indent_block(inputs_str, level=0).strip()
            G_LOGGER.info(inputs_str)

        def display_accuracy_results(accuracy_results):
            # Saved accuracy comparison results: per comparison function, per runner pair, the stored
            # per-output metrics and pass/fail verdict. Thresholds are stored, so this needs no
            # comparison function.
            out_str = (
                f"==== Accuracy Results ({len(accuracy_results)} comparison(s)) ====\n"
            )
            for result in accuracy_results:
                out_str += f"\n---- Aggregation: {result.aggregation} ----\n"
                for runner_pair, iterations in result.items():
                    matched, _, total = result.stats(runner_pair)
                    out_str += util.indent_block(
                        f"\n{runner_pair[0]} vs. {runner_pair[1]} | "
                        f"{matched}/{total} iteration(s) matched\n",
                        0,
                    )
                    for index, iteration in enumerate(iterations):
                        out_str += util.indent_block(f"\n-- Iteration: {index}", 1)
                        for output_name, output_result in iteration.items():
                            verdict = "PASSED" if bool(output_result) else "FAILED"
                            line = f"\n{output_name} | {verdict}"
                            if not isinstance(output_result, bool):
                                metrics = ", ".join(
                                    f"{field}={value:.5g}"
                                    for field, value in output_result.metric_values().items()
                                    if value is not None
                                )
                                if metrics:
                                    line += f" | {metrics}"
                            out_str += util.indent_block(line, 2)
                        if not args.all:
                            break
                    out_str += "\n"
            G_LOGGER.info(util.indent_block(out_str, level=0).strip())

        # Note: It's important we have encode/decode JSON methods registered for the types we care
        # about, e.g. RunResults. Importing the class should generally guarantee this.
        #
        # A directory holds one JSON file per iteration (as written by an extensionless
        # --save-inputs/--save-outputs); stream it back with the same readers `run` uses. Peek the
        # first file to tell saved runner results (a RunResults) from saved input feed_dicts -- the
        # same distinction made for a single file below.
        if os.path.isdir(args.path):
            iter_files = serde.list_iteration_files(args.path)
            if not iter_files:
                G_LOGGER.critical(
                    f"No per-iteration '*.json' files found in directory: {args.path}"
                )
            if isinstance(load_json(iter_files[0]), RunResults):
                display_results(RunResults.concat(RunResults.load_streaming(args.path)))
            else:
                display_inputs(list(StreamingDataLoader(args.path)))
            return

        data = load_json(args.path)
        if isinstance(data, RunResults):
            display_results(data)
        elif isinstance(data, AccuracyResults):
            display_accuracy_results(data)
        else:
            if not util.is_sequence(data):
                data = [data]
            display_inputs(data)
