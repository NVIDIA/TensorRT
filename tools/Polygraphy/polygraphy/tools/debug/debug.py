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
from polygraphy.tools.base import Tool
from polygraphy.tools.debug.subtool import Build, Precision, Reduce, Repeat

# For backwards compatibility
from polygraphy.tools.inspect.subtool import DiffTactics


class Debug(Tool):
    r"""
    [EXPERIMENTAL] Debug a wide variety of model issues.

    The `debug` subtools work on the same general principles:

    1. Iteratively perform some task that generates some output
    2. Evaluate the generated output to determine if it should be considered `good` or `bad`
    3. Sort any tracked artifacts into `good` and `bad` directories based on (2)
    4. Make changes if required and then repeat the process

    The "some output" referred to in (1) is usually a model file and is written to the current
    directory by default during each iteration.

    In order to distinguish between `good` and `bad`, the subtool uses one of two methods:
        a. The `--check` command, if one is provided. It can be virtually any command, which makes `debug` extremely flexible.
        b. Prompting you. If no `--check` command is provided, the subtool will prompt you in an interactive fashion
            to report whether the iteration passed or failed.

    Per-iteration artifacts to track can be specified with `--artifacts`. When the iteration fails,
    they are moved into the `bad` directory and otherwise into the `good` directory.
    Artifacts can be any file or directory. This can be used, for example, to sort logs or
    TensorRT tactic replay files, or even the per-iteration output (usually a TensorRT engine or ONNX model).

    By default, if the status code of the `--check` command is non-zero, the iteration is considered a failure.
    You can optionally use additional command-line options to control what counts as a failure in a more fine-grained way.
    For example:
        * `--fail-regex` allows you to count faliures only when the output of `--check` (on `stdout` or `stderr`)
            matches one or more regular expression(s) and ignore any other errors.
        * `--fail-returncode` lets you specify a status code to count as a failure, excluding all other non-zeros status
            codes.

    Most subtools also provide a replay mechanism where a 'replay file' containing information about the
    status of each iteration is saved after each iteration. This can then be loaded during subsequent debugging commands
    in order to quickly resume debugging from the same point.

    The general usage of most `debug` subtools is:

        polygraphy debug <subtool> <model> [--artifacts files_to_sort_each_iteration...] \
            [--check <checker_command>]
    """

    def __init__(self):
        super().__init__("debug")

    def get_subtools_impl(self):
        return "Debug Subtools", [
            Build(),
            Precision(),
            DiffTactics(_issue_deprecation_warning=True),
            Reduce(),
            Repeat(),
        ]
