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


r"""
This script generates an SVG diagram of the input engine graph SVG file.


$ trex draw mymodel.graph.json -pj=mymodel.profile.json -dm -dr -dc --query="transformer_blocks.(12|37).*"
$ trex draw mymodel.graph.json -pj=/mymodel.profile.json -dm --query="single_transformer_blocks\.8[a-zA-Z]*

Note: this script requires graphviz which can be installed manually:
    $ sudo apt-get --yes install graphviz
    $ python3 -m pip install graphviz
"""


import os
import re
import warnings
import argparse
import shutil
import subprocess
from typing import Tuple
import networkx as nx
import trex.graphing
import trex.engine_plan


def convert_dot_to_svg(dot_file, svg_file):
    try:
        subprocess.run(['dot', '-Tsvg', dot_file, '-o', svg_file], check=True)
        print(f"Successfully converted {dot_file} to {svg_file}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def create_directed_subgraph(G: nx.DiGraph, center_node: str, radius: int):
    """
    Create a directed subgraph containing all nodes within a given radius
    from the center node, preserving edge directionality.

    Parameters:
        G (nx.DiGraph): The original directed graph.
        center_node (node): The node to center the subgraph around.
        radius (int): The maximum distance from the center node.

    Returns:
        nx.DiGraph: A subgraph containing nodes within the radius and their edges.
    """
    try:
        # Perform BFS to find nodes within the radius
        fwd_beam = nx.single_source_shortest_path_length(G, center_node, cutoff=radius).keys()
        bwd_beam = nx.single_source_shortest_path_length(G.reverse(copy=False), center_node, cutoff=radius).keys()
        nodes_within_radius = list(fwd_beam) + list(bwd_beam)

        # Create a subgraph induced on these nodes
        subgraph = G.subgraph(nodes_within_radius)
        return subgraph
    except nx.exception.NodeNotFound:
        return None


def query_layers(plan, G: nx.DiGraph, query: str):
    matched_nodes = []
    for name, data in G.nodes(data=True):
        found = re.search(query, name) or re.search(query, data["tooltip"])
        if found:
            radius = 1
            fwd_beam = nx.single_source_shortest_path_length(G, name, cutoff=radius).keys()
            bwd_beam = nx.single_source_shortest_path_length(G.reverse(copy=False), name, cutoff=radius).keys()
            nodes_within_radius = list(fwd_beam) + list(bwd_beam)
            matched_nodes += nodes_within_radius

    # Create a subgraph induced on these nodes
    print(f"Found {len(matched_nodes)} matching nodes")
    subgraph = G.subgraph(matched_nodes)
    return subgraph


def nx_from_dot(graph, engine_json_fname):
    output_fname = os.path.abspath(f"{engine_json_fname}.dot")
    with open(output_fname, 'w') as f:
        f.write(str(graph).replace(':', ';'))  # Replace colons for safety
    print("Computing subgraph...")
    G = nx.DiGraph(nx.nx_pydot.read_dot(output_fname))
    return G, output_fname


def nx_to_svg(G, output_fname):
    print("Rendering subgraph...")
    nx.nx_pydot.write_dot(G, output_fname + ".radius.dot")
    convert_dot_to_svg(output_fname + ".radius.dot", output_fname + ".radius.dot.svg")


def draw_subgraph_beam(plan, graph, engine_json_fname, center_op, radius):
    G, output_fname = nx_from_dot(graph, engine_json_fname)
    if G is None:
        print("Internal error: could not generate nx.DiGraph")
        return
    neighbors = create_directed_subgraph(G, center_op, radius=radius)
    if not neighbors:
        print(f"Could not find {center_op}")
        return
    nx_to_svg(neighbors, output_fname)


def draw_subgraph_query(plan, graph, engine_json_fname, query):
    G, output_fname = nx_from_dot(graph, engine_json_fname)
    if G is None:
        print("Internal error: could not generate nx.DiGraph")
        return
    neighbors = query_layers(plan, G, query)
    if not neighbors:
        print(f"Could not find any matches for query: {query}")
        return
    nx_to_svg(neighbors, output_fname)


def draw_engine(engine_json_fname: str,
    profiling_json_fname: str=None,
    output_format: str=None,
    beam_center: str=None,
    beam_radius: int=0,
    query: str=None,
    **kwargs
):
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

    try:
        if kwargs["display_region_names"] and not kwargs["display_regions"]:
            kwargs["display_regions"] = True
    except KeyError:
        pass
    plan = trex.engine_plan.EnginePlan(engine_json_fname, profiling_file=profiling_json_fname)
    layer_node_formatter = trex.graphing.layer_type_formatter
    graph = trex.graphing.to_dot(plan, layer_node_formatter, **kwargs)
    output_format = output_format or "svg"

    def write_dot():
        output_fname = os.path.abspath(f"{engine_json_fname}.dot")
        with open(output_fname, 'w') as f:
            f.write(str(graph))

    if output_format == "dot":
        write_dot()
    elif beam_center and beam_radius >0:
        write_dot()
        draw_subgraph_beam(plan, graph, engine_json_fname, beam_center, beam_radius)
    elif query:
        write_dot()
        draw_subgraph_query(plan, graph, engine_json_fname, query)
    else:
        trex.graphing.render_dot(graph, engine_json_fname, output_format)


def make_subcmd_parser(subparsers):
    draw = lambda args: draw_engine(
        engine_json_fname=args.input,
        profiling_json_fname=args.profiling_json,
        output_format=args.output_format,
        beam_center=args.beam[0],
        beam_radius=args.beam[1],
        query=args.query,
        # Arguments to DotGraph
        display_forking_regions=True,
        display_region_reuse=True,
        display_edge_details=False,
        remove_disconnected_layers=True,
        highlight_layers=[args.beam[0]],
        display_regions=args.display_regions,
        display_layer_names=not args.no_layer_names,
        display_constants=args.display_constant,
        display_metadata=args.display_metadata,
        display_region_names=args.display_region_names,
    )
    draw_parser = subparsers.add_parser("draw", help="Draw a TensorRT engine.")
    draw_parser.set_defaults(func=draw)
    _make_parser(draw_parser)


def parse_beam(value) -> Tuple[str, int]:
    try:
        beam_center, beam_radius = value.split(',')
        return beam_center, int(beam_radius)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Beam must be in the format 'some_string,integer'")


def _make_parser(parser):
    parser.add_argument("input", help="name of engine JSON file to draw.")
    parser.add_argument("--profiling_json", "-pj",
        default=None, help="name of engine JSON file to draw")
    parser.add_argument("--output_format", "-of",
        default="svg", choices=["svg", "png", "dot"], help="output formats")
    parser.add_argument("--display_regions", "-dr",
        action='store_true', help="render memory regions as graph nodes.")
    parser.add_argument("--display_constant", "-dc",
        action='store_true', help="render constant input tensors.")
    parser.add_argument("--no_layer_names", "-no_ln",
        action='store_true', help="don't render layer names.")
    parser.add_argument("--display_metadata", "-dm",
        action='store_true', help="display metadata.")
    parser.add_argument("--display_region_names", "-drn",
        action='store_true', help="display region names.")
    parser.add_argument("--beam", "-b", default=("",0),
        type=parse_beam,
        help="draw the induced subgraph of neighbors centered at node n within a given radius.")
    parser.add_argument("--query", "-q",
        help="draw the induced subgraph of all nodes whose name or metadata matches the regular expression.")


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
