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
from polygraphy.tools.base import Tool
from polygraphy.tools.inspect.subtool import Data, Model, Tactics, Capability


class Inspect(Tool):
    """
    View information about various types of files.
    """

    def __init__(self):
        super().__init__("inspect")

    def add_parser_args(self, parser):
        subparsers = parser.add_subparsers(title="Inspection Subtools", dest="subtool")
        subparsers.required = True

        SUBTOOLS = [
            Model(),
            Data(),
            Tactics(),
            Capability()
        ]

        for subtool in SUBTOOLS:
            subtool.setup_parser(subparsers)
