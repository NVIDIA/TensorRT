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
import time

import tensorrt as trt
from polygraphy.backend.trt import get_trt_logger

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir))

# Use bin/polygraphy for any invocations of Polygraphy that don't use the script_runner fixture.
POLYGRAPHY_CMD = [os.path.join(ROOT_DIR, "bin", "polygraphy")]

# CLI tools and all their subtools
ALL_TOOLS = {
    "run": [],
    "convert": [],
    "inspect": ["data", "model", "tactics", "capability", "diff-tactics"],
    "surgeon": ["extract", "insert", "sanitize"],
    "template": ["trt-network", "trt-config", "onnx-gs"],
    "debug": ["build", "precision", "reduce", "repeat"],
    "data": ["to-input"],
}


def get_file_size(path):
    return os.stat(path).st_size


def is_file_empty(path):
    return get_file_size(path) == 0


def is_file_non_empty(path):
    return not is_file_empty(path)


def time_func(func, warm_up=25, iters=100):
    for _ in range(warm_up):
        func()

    start = time.time()
    for _ in range(iters):
        func()
    end = time.time()
    return (end - start) / float(iters)


HAS_DLA = None


def has_dla():
    global HAS_DLA
    if HAS_DLA is None:
        builder = trt.Builder(get_trt_logger())

        try:
            HAS_DLA = builder.num_DLA_cores > 0
        except AttributeError:
            HAS_DLA = False

    return HAS_DLA
