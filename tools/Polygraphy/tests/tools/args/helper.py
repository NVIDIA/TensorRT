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

from polygraphy import util
import argparse


class ArgGroupTestHelper:
    def __init__(self, arg_group, deps=None):
        self.deps = util.default(deps, [])

        self.arg_group = arg_group
        self.parser = argparse.ArgumentParser()

        arg_groups = {type(self.arg_group): self.arg_group}
        arg_groups.update({type(dep): dep for dep in self.deps})

        for dep in self.deps:
            dep.register(arg_groups)
        self.arg_group.register(arg_groups)

        for dep in self.deps:
            dep.add_parser_args(self.parser)
        self.arg_group.add_parser_args(self.parser)

    def parse_args(self, cli_args):
        args = self.parser.parse_args(cli_args)
        for dep in self.deps:
            dep.parse(args)
        self.arg_group.parse(args)
        return args

    def __getattr__(self, name):
        if name in ["arg_group", "parser"]:
            return super().__getattr__(name)
        return getattr(self.arg_group, name)
