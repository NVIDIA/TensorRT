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
import os
import time


def get_file_size(path):
    return os.stat(path).st_size


def is_file_empty(path):
    return get_file_size(path) == 0


def is_file_non_empty(path):
    return not is_file_empty(path)


def time_func(func, warm_up=10, iters=100):
    for _ in range(warm_up):
        func()

    total = 0
    for _ in range(iters):
        start = time.time()
        func()
        end = time.time()
        total += end - start
    return total / float(iters)
