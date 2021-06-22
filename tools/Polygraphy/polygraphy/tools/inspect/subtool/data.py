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
from polygraphy import util
from polygraphy.common import TensorMetadata
from polygraphy.comparator import RunResults
from polygraphy.comparator import util as comp_util
from polygraphy.json import load_json
from polygraphy.logger import G_LOGGER
from polygraphy.tools.base import Tool


class Data(Tool):
    """
    Display information about inference inputs and outputs saved from Polygraphy's Comparator.run()
    (for example, outputs saved by `--save-outputs` or inputs saved by `--save-inputs` from `polygraphy run`).
    """
    def __init__(self):
        super().__init__("data")


    def add_parser_args(self, parser):
        parser.add_argument("path", help="Path to a file containing input or output data from Polygraphy")
        parser.add_argument("-a", "--all", help="Show information on all iterations present in the data instead of just the first",
                            action="store_true")
        parser.add_argument("-s", "--show-values", help="Show values of the tensors instead of just metadata", action="store_true")
        parser.add_argument("--histogram", help="Show a histogram of the value distribution", action="store_true")


    def run(self, args):
        # Note: It's important we have encode/decode JSON methods registered
        # for the types we care about, e.g. RunResults. Importing the class should generally guarantee this.
        data = load_json(args.path)

        def meta_from_iter_result(iter_result):
            meta = TensorMetadata()
            for name, arr in iter_result.items():
                meta.add(name, dtype=arr.dtype, shape=arr.shape)
            return meta


        def str_from_iters(iters):
            out_str = ""
            for index, iter_result in enumerate(iters):
                if args.show_values:
                    for name, arr in iter_result.items():
                        out_str += "{:} [dtype={:}, shape={:}]\n{:}\n".format(name, arr.dtype, arr.shape, util.indent_block(str(arr)))
                else:
                    iter_meta = meta_from_iter_result(iter_result)
                    if len(iters) > 1 and args.all:
                        out_str += util.indent_block("Iteration: {:} | ".format(index))
                    out_str += "{:}\n".format(iter_meta)

                stat_str = "\n-- Statistics --"
                for name, arr in iter_result.items():
                    stat_str += "\n{:} | Stats\n".format(name)
                    stat_str += util.indent_block(comp_util.str_output_stats(arr)) + "\n"
                    if args.histogram:
                        stat_str += util.indent_block(comp_util.str_histogram(arr)) + "\n"

                out_str += stat_str

                if not args.all:
                    break
            return out_str


        def display_results(results):
            results_str = ""
            results_str += "==== Run Results ({:} runners) ====\n\n".format(len(results))

            for runner_name, iters in results.items():
                results_str += "---- {:35} ({:} iterations) ----\n".format(runner_name, len(iters))
                results_str += str_from_iters(iters) + "\n"

            results_str = util.indent_block(results_str, level=0).strip()
            G_LOGGER.info(results_str)


        def display_inputs(input_data):
            inputs_str = ""
            inputs_str += "==== Data ({:} iterations) ====\n\n".format(len(input_data))
            inputs_str += str_from_iters(input_data) + "\n"
            inputs_str = util.indent_block(inputs_str, level=0).strip()
            G_LOGGER.info(inputs_str)


        if isinstance(data, RunResults):
            display_results(data)
        else:
            if not util.is_sequence(data):
                data = [data]
            display_inputs(data)
