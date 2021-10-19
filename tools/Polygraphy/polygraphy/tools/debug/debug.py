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
from polygraphy.tools.base import Tool
from polygraphy.tools.debug.subtool import Build, DiffTactics, Precision, Reduce, Repeat


class Debug(Tool):
    r"""
    [EXPERIMENTAL] Debug a wide variety of model issues.

    Most of the `debug` subtools work on the same general principles:

    1. Iteratively perform some task that generates some output
    2. Evaluate the generated output to determine if it should be considered `good` or `bad`
    3. Sort any tracked artifacts into `good` and `bad` directories based on (2)

    The "some output" referred to in (1) is usually a model file and is written to the current
    directory (by default) during each iteration.

    In order to distinguish between `good` and `bad`, the subtool uses the `--check` command provided by
    the user (that's you!). It can be virtually any command, which makes `debug` extremely flexible.

    Artifacts to track can be specified with `--artifacts`. When the `--check` command exits with a failure,
    they are moved into the `bad` directory and otherwise into the `good` directory.

    By default, if the status code of the `--check` command is non-zero, the iteration is considered a failure.
    You can optionally use additional command-line options to control what counts as a failure in a more fine-grained way.
    For example:
        * `--fail-regex` allows you to count faliures only when the output of `--check` (on `stdout` or `stderr`)
            matches one or more regular expression(s) and ignore any other errors.
        * `--fail-returncode` lets you specify a status code to count as a failure, excluding all other non-zeros status
            codes.

    The general usage of most `debug` subtools is:

        polygraphy debug <subtool> <model> [--artifacts files_to_sort_each_iteration...] \
            --check <checker_command>
    """

    def __init__(self):
        super().__init__("debug")

    def add_parser_args(self, parser):
        subparsers = parser.add_subparsers(title="Debug Subtools", dest="subtool")
        subparsers.required = True

        SUBTOOLS = [
            Build(),
            Precision(),
            DiffTactics(),
            Reduce(),
            Repeat(),
        ]

        for subtool in SUBTOOLS:
            subtool.setup_parser(subparsers)
