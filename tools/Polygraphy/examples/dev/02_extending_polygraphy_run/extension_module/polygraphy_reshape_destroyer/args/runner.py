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
This file defines the `IdentityOnlyRunnerArgs` argument group, which manages
command-line options that control the `IdentityOnlyRunner` runner.

The argument group implements the standard `BaseRunnerArgs` interface, which inherits from `BaseArgs`.
"""

from polygraphy import mod
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseRunnerArgs
from polygraphy.tools.script import make_invocable
from polygraphy_reshape_destroyer.args.loader import ReplaceReshapeArgs


# NOTE: Much like loader argument groups, runner argument groups may depend on other argument groups.
@mod.export()
class IdentityOnlyRunnerArgs(BaseRunnerArgs):
    """
    Identity-Only Runner Inference: running inference with the identity-only runner.

    Depends on:

        - ReplaceReshapeArgs
    """

    def get_name_opt_impl(self):
        # Unlike regular `BaseArgs` argument groups, runner argument groups are also expected
        # to provide a human readable name for the runner as well as a name for
        # the option that will toggle the runner, not including leading dashes.
        #
        # We'll use "res-des" for the option, which will allow us to use the runner by setting `--res-des`.
        return "Identity-Only Runner", "res-des"

    def add_parser_args_impl(self):
        # Once again, to prevent collisions with other Polygraphy options, we prefix our option with `res-des`.
        self.group.add_argument(
            "--res-des-speed",
            help="Speed to run inference",
            choices=["slow", "medium", "fast"],
            # Since our runner uses `util.default`, we can use `None` as a universal default.
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            speed (str): Speed with which to run inference.
        """
        self.speed = args_util.get(args, "res_des_speed")

    def add_to_script_impl(self, script):
        # We'll rely on our ReplaceReshapeArgs argument group to create the ONNX-GraphSurgeon graph for us:
        loader_name = self.arg_groups[ReplaceReshapeArgs].add_to_script(script)

        # Next, we'll add an import for our runner.
        script.add_import(imports=["IdentityOnlyRunner"], frm="polygraphy_reshape_destroyer.backend")
        # Lastly, we can add our runner using the `Script.add_runner()` API.
        # Like in the loader implementation, additional arguments can be provided directly to `make_invocable`.
        script.add_runner(make_invocable("IdentityOnlyRunner", loader_name, speed=self.speed))

        # NOTE: Unlike the `add_to_script_impl` method of regular `BaseArgs`, that of `BaseRunnerArgs`
        #       is not expected to return anything.
