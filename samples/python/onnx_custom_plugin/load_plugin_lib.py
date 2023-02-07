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
import ctypes

WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(os.path.realpath(__file__))
IS_WINDOWS = os.name == "nt"
if IS_WINDOWS:
    HARDMAX_PLUGIN_LIBRARY_NAME = "customHardmaxPlugin.dll"
    HARDMAX_PLUGIN_LIBRARY = [
        os.path.join(WORKING_DIR, "build", "Debug", HARDMAX_PLUGIN_LIBRARY_NAME),
        os.path.join(WORKING_DIR, "build", "Release", HARDMAX_PLUGIN_LIBRARY_NAME),
    ]
else:
    HARDMAX_PLUGIN_LIBRARY_NAME = "libcustomHardmaxPlugin.so"
    HARDMAX_PLUGIN_LIBRARY = [os.path.join(WORKING_DIR, "build", HARDMAX_PLUGIN_LIBRARY_NAME)]

def load_plugin_lib():
    for plugin_lib in HARDMAX_PLUGIN_LIBRARY:
        if os.path.isfile(plugin_lib):
            try:
                # Python specifies that winmode is 0 by default, but some implementations
                # incorrectly default to None instead. See:
                # https://docs.python.org/3.8/library/ctypes.html
                # https://github.com/python/cpython/blob/3.10/Lib/ctypes/__init__.py#L343
                ctypes.CDLL(plugin_lib, winmode=0)
            except TypeError:
                # winmode only introduced in python 3.8
                ctypes.CDLL(plugin_lib)
            return

    raise IOError(
        "\n{}\n{}\n{}\n".format(
            "Failed to load library ({}).".format(HARDMAX_PLUGIN_LIBRARY_NAME),
            "Please build the Hardmax sample plugin.",
            "For more information, see the included README.md",
        )
    )
