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
from polygraphy.tools.script import make_invocable


@mod.export()
class TfRunnerArgs(BaseArgs):
    def add_to_parser(self, parser):
        tf_args = parser.add_argument_group("TensorFlow Runner", "Options for TensorFlow Inference")
        tf_args.add_argument(
            "--save-timeline",
            help="[EXPERIMENTAL] Directory to save timeline JSON files for profiling inference (view at chrome://tracing)",
            default=None,
        )

    def register(self, maker):
        from polygraphy.tools.args.tf.config import TfConfigArgs
        from polygraphy.tools.args.tf.loader import TfLoaderArgs

        if isinstance(maker, TfLoaderArgs):
            self.tf_loader_args = maker
        if isinstance(maker, TfConfigArgs):
            self.tf_config_args = maker

    def check_registered(self):
        assert self.tf_loader_args is not None, "TfLoaderArgs is required!"
        assert self.tf_config_args is not None, "TfConfigArgs is required!"

    def parse(self, args):
        self.timeline_path = args_util.get(args, "save_timeline")

    def add_to_script(self, script):
        script.add_import(imports=["TfRunner"], frm="polygraphy.backend.tf")

        graph_name = self.tf_loader_args.add_to_script(script)
        config_name = self.tf_config_args.add_to_script(script)

        script.add_import(imports=["SessionFromGraph"], frm="polygraphy.backend.tf")
        loader_name = script.add_loader(
            make_invocable("SessionFromGraph", graph_name, config=config_name), "build_tf_session"
        )

        script.add_runner(make_invocable("TfRunner", loader_name, timeline_path=self.timeline_path))
