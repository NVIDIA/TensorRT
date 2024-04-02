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
This script generates an SVG diagram of the input engine graph SVG file.

Note: this script requires graphviz which can be installed manually:
    $ sudo apt-get --yes install graphviz
    $ python3 -m pip install graphviz
"""


import warnings
import argparse
import shutil
import trex.graphing
import trex.engine_plan


def draw_engine(engine_json_fname: str, profiling_json_fname: str=None, **kwargs):
    graphviz_is_installed = shutil.which("dot") is not None
    if not graphviz_is_installed:
        print("graphviz is required but it is not installed.\n")
        print("To install on Ubuntu:")
        print("sudo apt --yes install graphviz")
        exit()

    try:
        if kwargs["display_constants"] and not kwargs["display_regions"]:
            warnings.warn("Ignoring argument --display_constants because it requires --display_regions.")
    except KeyError:
        pass

    plan = trex.engine_plan.EnginePlan(engine_json_fname, profiling_file=profiling_json_fname)
    layer_node_formatter = trex.graphing.layer_type_formatter
    graph = trex.graphing.to_dot(plan, layer_node_formatter, **kwargs)
    trex.graphing.render_dot(graph, engine_json_fname, "svg")


def make_subcmd_parser(subparsers):
    draw = lambda args: draw_engine(
        engine_json_fname=args.input,
        profiling_json_fname=args.profiling_json,
        display_regions=args.display_regions,
        display_layer_names=not args.no_layer_names,
        display_constants=args.display_constant,
    )
    draw_parser = subparsers.add_parser("draw", help="Draw a TensorRT engine.")
    draw_parser.set_defaults(func=draw)
    _make_parser(draw_parser)


def _make_parser(parser):
    parser.add_argument("input", help="name of engine JSON file to draw.")
    parser.add_argument("--profiling_json", "-pj",
        default=None, help="name of engine JSON file to draw")
    parser.add_argument("--display_regions", "-dr",
        action='store_true', help="render memory regions as graph nodes.")
    parser.add_argument("--display_constant", "-dc",
        action='store_true', help="render constant input tensors.")
    parser.add_argument("--no_layer_names", "-no_ln",
        action='store_true', help="render constants.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args(_make_parser(parser))

    draw_engine(
        engine_json_fname=args.input,
        profiling_json_fname=args.profiling_json,
        display_regions=True,
        expand_layer_details=False,
        display_latency=True,
    )
