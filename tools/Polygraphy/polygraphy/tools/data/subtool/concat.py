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
from polygraphy import util
from polygraphy.comparator import RunResults
from polygraphy.json import load_json, save_json
from polygraphy.logger import G_LOGGER
from polygraphy.tools.base import Tool


class Concat(Tool):
    """
    Concatenates iterations from one or more input/output files into a single file.
    """

    def __init__(self):
        super().__init__("concat")

    def add_parser_args(self, parser):
        parser.add_argument(
            "paths",
            help="Path(s) to file(s) containing input or output data from Polygraphy. "
            "All files must be the same type (all inputs or all outputs).",
            nargs="+",
        )
        parser.add_argument(
            "-o", "--output", help="Path to the file to generate", required=True
        )

    def run_impl(self, args):
        inputs = []
        output_runs = []
        reference_keys = {}
        use_run_results = False

        def maybe_warn_mismatch(iter_inputs, path, iteration_index, stream_key):
            try:
                keys = list(iter_inputs.keys())
            except Exception:
                G_LOGGER.warning(
                    f"Iteration {iteration_index} from {path} is not a feed_dict; skipping name comparison."
                )
                return

            if stream_key not in reference_keys:
                reference_keys[stream_key] = keys
                return

            if keys != reference_keys[stream_key]:
                G_LOGGER.warning(
                    f"Iteration {iteration_index} from {path} contains different tensor names than earlier iterations.\n"
                    f"Note: Expected names: {reference_keys[stream_key]}, but got: {keys}."
                )

        for path in args.paths:
            data = load_json(path)
            if isinstance(data, RunResults):
                use_run_results = True
                if inputs:
                    G_LOGGER.critical(
                        "Cannot concatenate input data with output data. Please provide files of the same type."
                    )

                for runner_name, iters in data.items():
                    stream_key = f"runner:{runner_name}"
                    for iteration_index, iteration_outputs in enumerate(iters):
                        maybe_warn_mismatch(
                            iteration_outputs, path, iteration_index, stream_key
                        )

                # RunResults.concat does the by-name iteration concatenation across runs.
                output_runs.append(data)
            else:
                if use_run_results:
                    G_LOGGER.critical(
                        "Cannot concatenate input data with output data. Please provide files of the same type."
                    )
                if not util.is_sequence(data):
                    data = [data]

                for iteration_index, iteration_inputs in enumerate(data):
                    maybe_warn_mismatch(
                        iteration_inputs, path, iteration_index, "inputs"
                    )
                    inputs.append(iteration_inputs)

        if use_run_results:
            save_json(
                RunResults.concat(output_runs),
                args.output,
                description="output file containing concatenated iteration(s)",
            )
        else:
            save_json(
                inputs,
                args.output,
                description=f"input file containing {len(inputs)} iteration(s)",
            )
