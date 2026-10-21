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
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs


class VisualArgs(BaseArgs):
    """
    Visualization: visualizing models graphically as an interactive DAG.
    """

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--visual",
            help="Launch an interactive GUI to visualize the model graph as a DAG. "
            "Nodes can be clicked to inspect their details.",
            action="store_true",
            default=False,
            dest="visual",
        )
        self.group.add_argument(
            "--visual-port",
            help="Port for the local HTTP server started by --visual. "
            "Use a fixed port when running inside a container so you can forward it "
            "(e.g. docker run -p 8000:8000 ...). Defaults to 8000.",
            type=int,
            default=8000,
            dest="visual_port",
        )
        self.group.add_argument(
            "--save-visual",
            help="Save the --visual HTML to a file instead of opening the browser. "
            "Useful for sharing or automated testing.",
            type=str,
            default=None,
            dest="save_visual",
        )

    def parse_impl(self, args):
        """
        Attributes:
            visual (bool): Whether to launch the interactive graph viewer.
            port (int): Port for the viewer's local HTTP server.
            save_path (str): If set, write the HTML here instead of serving it.
        """
        self.visual = args_util.get(args, "visual")
        self.port = args_util.get(args, "visual_port")
        self.save_path = args_util.get(args, "save_visual")
