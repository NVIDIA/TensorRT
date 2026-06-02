#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ctypes
import glob
import os

CURDIR = os.path.realpath(os.path.dirname(__file__))


def try_load(library):
    try:
        ctypes.CDLL(library)
    except OSError:
        pass


def try_load_libs_from_dir(path):
    for lib in glob.iglob(os.path.join(path, "*.so*")):
        try_load(lib)
    for lib in glob.iglob(os.path.join(path, "*.dll*")):
        try_load(lib)


# Try loading all packaged libraries. This is a nop if there are no libraries packaged.
try_load_libs_from_dir(CURDIR)
