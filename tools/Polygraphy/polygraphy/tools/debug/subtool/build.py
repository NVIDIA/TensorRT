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

from polygraphy.tools.debug.subtool.base import BaseCheckerSubtool


class Build(BaseCheckerSubtool):
    r"""
    Repeatedly build an engine to isolate faulty tactics.

    `debug build` follows the same general process as other `debug` subtools (refer to the help output
    of the `debug` tool for more background information and details).

    Specifically, it does the following during each iteration:

    1. Builds a TensorRT engine and saves it in the current directory as `polygraphy_debug.engine` by default.
    2. Evaluates it using the `--check` command if it was provided, or in interactive mode otherwise.
    3. Sorts files specified by `--artifacts` into `good` and `bad` directories based on (2).
        This is useful for sorting tactic replays, which can then be further analyzed with `inspect diff-tactics`.

    The typical usage of `debug build` is:

        polygraphy debug build <model> [trt_build_options...] [--save-tactics <replay_file>] \
            [--artifacts <replay_file>] --until <num_iterations> \
            [--check <check_command>]

    `polygraphy run` is usually a good choice for the `--check` command.
    """

    def __init__(self):
        # Debug replays don't make any sense here because this tool is meant to track down non-deterministic behavior.
        super().__init__("build", allow_until_opt=True, allow_debug_replay=False)
