#!/usr/bin/env python3
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


"""
This is a command-line interface to several common utilities.

Note: this script requires graphviz which can be installed manually:
    $ sudo apt-get --yes install graphviz
    $ python3 -m pip install graphviz
"""

import sys
import os
import argparse
import trex
import logging
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), "utils"))
import utils.draw_engine as draw_engine
import utils.summarize_engine as summarize_engine
have_process_engine = True
try:
    import utils.process_engine as process_engine
except ModuleNotFoundError:
    # process_engine can only be used in environments that have tensorrt python bindings installed.
    logging.warning("Failed to import process_engine. Disabling sub-command `process`.")
    logging.warning("This happens when the TensorRT Python library is not installed or fails to initialize. For example, when running TREx on a computer without a GPU).")
    have_process_engine = False


def main():
    try:
        sys.argv[1]
    except IndexError:
        sys.argv.append("-h")
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='TREx sub-commands')
    draw_engine.make_subcmd_parser(subparsers)
    summarize_engine.make_subcmd_parser(subparsers)
    if have_process_engine:
        process_engine.make_subcmd_parser(subparsers)
    parser.add_argument('-v', '--version', action='version',
        version=f'%(prog)s {trex.__version__}',
        help="Show program's version number and exit.")

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        logging.error("Failed to parse trex arguments")


if __name__ == "__main__":
    main()


