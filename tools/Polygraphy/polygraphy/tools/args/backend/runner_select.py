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
import argparse

from polygraphy import mod
from polygraphy.common.interface import TypedList
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs, BaseRunnerArgs


def make_action_cls(runner_opt):
    class StoreRunnerOrdered(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if not hasattr(namespace, "runners"):
                namespace.runners = []
            namespace.runners.append(runner_opt)

    return StoreRunnerOrdered


class RunnerOptList(TypedList(lambda: tuple)):
    def keys(self):
        for opt, _ in self.lst:
            yield opt

    def values(self):
        for _, name in self.lst:
            yield name

    def items(self):
        yield from self.lst


@mod.export()
class RunnerSelectArgs(BaseArgs):
    """
    Runners: selecting runners to use for inference.
    """

    def add_parser_args_impl(self):
        self._opt_to_group_map = {}
        for arg_group in self.arg_groups.values():
            if not isinstance(arg_group, BaseRunnerArgs):
                continue

            name, opt = arg_group.get_name_opt()
            extra_help = arg_group.get_extra_help_text()
            # Use opt as the key since it's guaranteed to be unique.
            self._opt_to_group_map[opt] = arg_group
            self.group.add_argument(
                f"--{opt}",
                help=f"Run inference using {name}. {extra_help}",
                action=make_action_cls(opt),
                dest="runners",
                default=[],
                nargs=0,
            )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            runners (List[Tuple[str, str]]):
                    A list of tuples mapping runner option strings to human readable names for all selected runners,
                    in the order they were selected in.
                    For example:
                    ::

                        [("trt", "TensorRT"), ("onnxrt", "ONNX-Runtime")]
        """
        runner_opts = args_util.get(args, "runners")

        self.runners = RunnerOptList()
        for opt in runner_opts:
            self.runners.append((opt, self._opt_to_group_map[opt].get_name_opt()[0]))

    def add_to_script_impl(self, script):
        """
        Adds all selected runners to the script.

        Returns:
            str: The name of the list of runners in the script.
        """
        if not self.runners:
            G_LOGGER.warning("No runners have been selected. Inference will not be run!")

        for opt in self.runners.keys():
            self._opt_to_group_map[opt].add_to_script(script)
        return script.get_runners()
