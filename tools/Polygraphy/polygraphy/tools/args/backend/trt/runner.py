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
from polygraphy import mod
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.backend.trt.loader import TrtLoadEngineArgs
from polygraphy.tools.args.base import BaseRunnerArgs
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
        self.group.add_argument(
            "--allocation-strategy",
            help="The way activation memory is allocated. "
            "static: Pre-allocate based on the max possible size across all profiles. "
            "profile: Allocate what's needed for the profile to use."
            "runtime: Allocate what's needed for the current input shapes.",
            type=str,
            default=None,
            dest="allocation_strategy",
            choices=["static", "profile", "runtime"],
        )
        self.group.add_argument(
            "--weight-streaming-budget",
            help="The amount of GPU memory in bytes that TensorRT can use for weights at runtime. The engine must be built with weight streaming enabled. It can take on the following values: "
            "None or -2: Disables weight streaming at runtime. "
            "-1: TensorRT will decide the streaming budget automatically. "
            "0 to 100%%: The percentage of weights that TRT keeps on the GPU. 0%% will stream the maximum number of weights."
            ">=0B: The exact amount of streamable weights that reside on the GPU (unit suffixes are supported).",
            type=str,
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            optimization_profile (int): The index of the optimization profile to initialize the runner with.
            allocation_strategy (str): The way activation memory is allocated.
            weight_streaming_budget (int): The size of the weights on the GPU in bytes.
            weight_streaming_percent (float): The percentage of weights on the GPU.
        """
        self.optimization_profile = args_util.get(args, "optimization_profile")
        self.allocation_strategy = args_util.get(args, "allocation_strategy")
        self.weight_streaming_budget = None
        self.weight_streaming_percent = None

        ws_arg = args_util.get(args, "weight_streaming_budget")
        if ws_arg and ws_arg.endswith("%"):
            percent = float(ws_arg[:-1])
            assert (
                0 <= percent <= 100
            ), "Invalid percentage for --weight-streaming-budget!"
            self.weight_streaming_percent = percent
        elif ws_arg:
            budget = args_util.parse_num_bytes(ws_arg)
            assert (
                budget == -2 or budget == -1 or budget >= 0
            ), "Invalid amount for --weight-streaming-budget!"
            self.weight_streaming_budget = budget

    def add_to_script_impl(self, script):
        script.add_import(imports=["TrtRunner"], frm="polygraphy.backend.trt")
        loader_name = self.arg_groups[TrtLoadEngineArgs].add_to_script(script)
        script.add_runner(
            make_invocable(
                "TrtRunner",
                loader_name,
                optimization_profile=self.optimization_profile,
                allocation_strategy=self.allocation_strategy,
                weight_streaming_budget=self.weight_streaming_budget,
                weight_streaming_percent=self.weight_streaming_percent,
            )
        )
