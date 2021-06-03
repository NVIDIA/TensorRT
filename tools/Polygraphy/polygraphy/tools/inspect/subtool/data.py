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
from polygraphy.common import TensorMetadata
from polygraphy.comparator.struct import RunResults
from polygraphy.logger import G_LOGGER
from polygraphy.tools.base import Tool
from polygraphy.util import misc


class Data(Tool):
    """
    Display information about inference inputs and outputs saved from Polygraphy's Comparator.run()
    (for example, outputs saved by `--save-results` or inputs saved by `--save-inputs` from `polygraphy run`).
    """
    def __init__(self):
        super().__init__("data")


    def add_parser_args(self, parser):
        parser.add_argument("path", help="Path to a file containing input or output data from Polygraphy")
        parser.add_argument("-a", "--all", help="Show information on all iterations present in the data instead of just the first",
                            action="store_true")
        parser.add_argument("-s", "--show-values", help="Show values of output tensors instead of just metadata", action="store_true")


    def run(self, args):
        data = misc.pickle_load(args.path)

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
                        out_str += "{:} [dtype={:}, shape={:}]\n{:}\n\n".format(name, arr.dtype, arr.shape, misc.indent_block(str(arr)))
                else:
                    iter_meta = meta_from_iter_result(iter_result)
                    if len(iters) > 1 and args.all:
                        out_str += misc.indent_block("Iteration: {:} | ".format(index))
                    out_str += "{:}\n".format(iter_meta)

                if not args.all:
                    break
            return out_str


        def display_results():
            results_str = ""
            results_str += "==== Run Results ({:} runners) ====\n\n".format(len(data))

            for runner_name, iters in data.items():
                results_str += "---- Runner: {:} ({:} iterations) ----\n".format(runner_name, len(iters))
                results_str += str_from_iters(iters) + "\n"

            results_str = misc.indent_block(results_str, level=0).strip()
            G_LOGGER.info(results_str)


        def display_inputs():
            inputs_str = ""
            inputs_str += "==== Input Data ({:} iterations) ====\n\n".format(len(data))
            inputs_str += str_from_iters(data) + "\n"
            inputs_str = misc.indent_block(inputs_str, level=0).strip()
            G_LOGGER.info(inputs_str)


        if isinstance(data, RunResults):
            display_results()
        else:
            display_inputs()
