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

from polygraphy import mod
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.script import make_invocable_if_nondefault


@mod.export()
class TfConfigArgs(BaseArgs):
    def add_to_parser(self, parser):
        tf_args = parser.add_argument_group(
            "TensorFlow Session Configuration", "Options for the TensorFlow Session Configuration"
        )
        tf_args.add_argument(
            "--gpu-memory-fraction",
            help="Maximum percentage of GPU memory TensorFlow can allocate per process",
            type=float,
            default=None,
        )
        tf_args.add_argument(
            "--allow-growth", help="Allow GPU memory allocated by TensorFlow to grow", action="store_true", default=None
        )
        tf_args.add_argument(
            "--xla", help="[EXPERIMENTAL] Attempt to run graph with xla", action="store_true", default=None
        )

    def parse(self, args):
        self.gpu_memory_fraction = args_util.get(args, "gpu_memory_fraction")
        self.allow_growth = args_util.get(args, "allow_growth")
        self.xla = args_util.get(args, "xla")

    def add_to_script(self, script):
        config_loader_str = make_invocable_if_nondefault(
            "CreateConfig",
            gpu_memory_fraction=self.gpu_memory_fraction,
            allow_growth=self.allow_growth,
            use_xla=self.xla,
        )
        if config_loader_str is not None:
            script.add_import(imports=["CreateConfig"], frm="polygraphy.backend.tf")
            config_loader_name = script.add_loader(config_loader_str, "create_tf_config")
        else:
            config_loader_name = None
        return config_loader_name
