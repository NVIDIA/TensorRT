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
from polygraphy.tools.args.backend.tf.config import TfConfigArgs
from polygraphy.tools.args.backend.tf.loader import TfLoadArgs
from polygraphy.tools.script import make_invocable


@mod.export()
class TfRunnerArgs(BaseRunnerArgs):
    """
    TensorFlow Inference: running inference with TensorFlow.

    Depends on:

        - TfConfigArgs
        - TfLoadArgs
    """

    def get_name_opt_impl(self):
        return "TensorFlow", "tf"

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--save-timeline",
            help="[EXPERIMENTAL] Directory to save timeline JSON files for profiling inference (view at chrome://tracing)",
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            timeline_path (str): Path at which to save timeline files for profiling.
        """
        self.timeline_path = args_util.get(args, "save_timeline")

    def add_to_script_impl(self, script):
        script.add_import(imports=["TfRunner"], frm="polygraphy.backend.tf")

        graph_name = self.arg_groups[TfLoadArgs].add_to_script(script)
        config_name = self.arg_groups[TfConfigArgs].add_to_script(script)

        script.add_import(imports=["SessionFromGraph"], frm="polygraphy.backend.tf")
        loader_name = script.add_loader(
            make_invocable("SessionFromGraph", graph_name, config=config_name), "build_tf_session"
        )

        script.add_runner(make_invocable("TfRunner", loader_name, timeline_path=self.timeline_path))
