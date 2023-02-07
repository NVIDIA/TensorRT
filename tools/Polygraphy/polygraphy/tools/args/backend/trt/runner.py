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
from polygraphy import mod
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseRunnerArgs
from polygraphy.tools.args.model import ModelArgs
from polygraphy.tools.args.backend.trt.loader import TrtLoadEngineArgs
from polygraphy.tools.script import make_invocable


@mod.export()
class TrtRunnerArgs(BaseRunnerArgs):
    """
    TensorRT Inference: running inference with TensorRT.

    Depends on:

        - TrtLoadEngineArgs
    """

    def get_name_opt_impl(self):
        return "TensorRT", "trt"

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--optimization-profile",
            help="The index of optimization profile to use for inference",
            type=int,
            default=None,
            dest="optimization_profile",
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            optimization_profile (int): The index of the optimization profile to initialize the runner with.
        """
        self.optimization_profile = args_util.get(args, "optimization_profile")

    def add_to_script_impl(self, script):
        script.add_import(imports=["TrtRunner"], frm="polygraphy.backend.trt")
        loader_name = self.arg_groups[TrtLoadEngineArgs].add_to_script(script)
        script.add_runner(make_invocable("TrtRunner", loader_name, optimization_profile=self.optimization_profile))
