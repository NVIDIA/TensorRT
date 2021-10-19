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

from polygraphy.logger import G_LOGGER
from polygraphy.tools.debug.subtool.base import BaseCheckerSubtool


class Build(BaseCheckerSubtool):
    r"""
    Repeatedly build an engine to isolate faulty tactics.

    `debug build` follows the same general process as other `debug` subtools.
    Specifically, it does the following during each iteration:

    1. Builds a TensorRT engine and saves it in the current directory as `polygraphy_debug.engine` by default.
    2. Evaluates it using the provided `--check` command.
    3. Sorts files specified by `--artifacts` into `good` and `bad` directories based on (2).
        This is useful for sorting tactic replays, which can then be further analyzed with `debug diff-tactics`.

    The typical usage of `debug build` is:

        polygraphy debug build <model> [trt_build_options...] [--save-tactics <replay_file>] \
            [--artifacts <replay_file>] --until <num_iterations> \
            --check <check_command>

    `polygraphy run` is usually a good choice for the `--check` command.
    """

    def __init__(self):
        super().__init__("build")

    def add_parser_args(self, parser):
        parser.add_argument(
            "--until",
            required=True,
            help="Controls when to stop running. "
            "Choices are: ['good', 'bad', int]. 'good' will keep running until the first 'good' run. "
            "'bad' will run until the first 'bad' run. An integer can be specified to run a set number of iterations. ",
        )

    def setup(self, args, network):
        try:
            self.until = int(args.until) - 1
        except:
            self.until = args.until
            if self.until not in ["good", "bad"]:
                G_LOGGER.critical("--until value must be an integer, 'good', or 'bad', but was: {:}".format(args.until))

    def stop(self, index, success):
        if self.until == "good":
            return success
        elif self.until == "bad":
            return not success

        return index >= self.until
