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
from polygraphy.tools.debug.subtool.iterative_debug_args import ArtifactSortArgs, CheckCmdArgs, IterativeDebugArgs


class Repeat(Tool):
    """
    [EXPERIMENTAL] Run an arbitrary command repeatedly, sorting generated artifacts
    into `good` and `bad` directories.
    """

    def __init__(self):
        super().__init__("repeat")

    def get_subscriptions_impl(self):
        return [CheckCmdArgs(), ArtifactSortArgs(), IterativeDebugArgs(allow_iter_art_opt=False, allow_until_opt=True)]

    def show_start_end_logging_impl(self, args):
        return True

    def run_impl(self, args):
        self.arg_groups[IterativeDebugArgs].iterate()
