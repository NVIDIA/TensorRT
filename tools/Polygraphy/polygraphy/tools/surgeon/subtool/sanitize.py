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
from polygraphy.tools.surgeon.subtool.base import BaseSurgeonSubtool


class Sanitize(BaseSurgeonSubtool):
    """
    [EXPERIMENTAL] Remove unused nodes and fold constants where possible.
    """
    def __init__(self):
        super().__init__("sanitize")


    def add_parser_args(self, parser):
        parser.add_argument("--fold-constants", help="Fold constants in the graph by computing subgraphs whose values "
                            "are not dependent on runtime inputs.", action="store_true", default=None)
        super().add_parser_args(parser, gs=True, output=True)


    def run(self, args):
        _, graph = super().import_graph(args)
        if args.fold_constants:
            graph.fold_constants()
        super().export_graph(graph, args)
