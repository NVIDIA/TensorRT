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

import os
import subprocess as sp
from polygraphy import util
import argparse
import time

TEST_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

# Paramaters to run test on when testing trtexec args mapping
TRTEXEC_PATH_PARAMS = ["trtexec"]
TRTEXEC_EXPORT_TIMES_PARAMS = ["times.json"]
TRTEXEC_EXPORT_OUTPUT_PARAMS = ["output.json"]
TRTEXEC_EXPORT_PROFILE_PARAMS = ["profile.json"]
TRTEXEC_EXPORT_LAYER_INFO_PARAMS = ["layer_info.json"]

def poly_run(args):
    """
    Helper function to run the `polygraphy run` command with nececssary args
    """
    cmd_args = ["polygraphy", "run"] + args
    output = sp.run(cmd_args, stdout=sp.PIPE).stdout.decode('utf-8')
    return 'PASSED' in output

def is_file_non_empty(path):
    return os.stat(path).st_size != 0

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

def time_func(func, warm_up=10, iters=50):
    for _ in range(warm_up):
        func()

    total = 0
    for _ in range(iters):
        start = time.time()
        func()
        end = time.time()
        total += end - start
    return total / float(iters)

