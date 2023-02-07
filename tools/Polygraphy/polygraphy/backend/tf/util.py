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
from collections import defaultdict

from polygraphy import mod, util
from polygraphy.common import TensorMetadata
from polygraphy.logger import G_LOGGER

tf = mod.lazy_import("tensorflow<2.0")


def load_graph(path):
    """
    Loads a TensorFlow frozen model.

    Args:
        path (Union[str, tf.Graph, tf.GraphDef]):
                A path to the frozen model, or a frozen TensorFlow graph or graphdef.

    Returns:
        tf.Graph: The TensorFlow graph
    """
    if isinstance(path, tf.Graph):
        return path

    if isinstance(path, str):
        graphdef = tf.compat.v1.GraphDef()

        import google

        try:
            graphdef.ParseFromString(util.load_file(path, description="GraphDef"))
        except google.protobuf.message.DecodeError:
            G_LOGGER.backtrace()
            G_LOGGER.critical(f"Could not import TensorFlow GraphDef from: {path}. Is this a valid TensorFlow model?")
    elif isinstance(path, tf.compat.v1.GraphDef):
        graphdef = path

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graphdef, name="")
        return graph


def find_nodes_by_ops(graphdef, ops):
    ops = set(ops)
    return [node for node in graphdef.node if any([op in node.op for op in ops])]


def map_node_outputs(graphdef):
    def sanitize_input_name(input_name):
        # Strip port information and control symbol
        split_input = input_name.split(":")
        if len(split_input) > 1:
            split_input.pop(-1)
        return ":".join(split_input).replace("^", "")

    node_outputs = defaultdict(list)
    for node in graphdef.node:
        for input_name in node.input:
            node_outputs[sanitize_input_name(input_name)].append(node)
    return node_outputs


def get_tensor_metadata(tensors):
    metadata = TensorMetadata()
    for tensor in tensors:
        try:
            shape = [elem.value if hasattr(elem, "value") else elem for elem in tensor.shape]
        except ValueError:
            # Happens when rank is unknown
            shape = None
        metadata.add(tensor.name, dtype=tensor.dtype.as_numpy_dtype, shape=shape)
    return metadata


def get_input_metadata(graph):
    input_tensors = []
    input_nodes = find_nodes_by_ops(graph.as_graph_def(), ["Placeholder", "FIFOQueue"])
    G_LOGGER.verbose(f"Found input tensors: {[f'{n.name}: {n.op}' for n in input_nodes]}")
    for node in input_nodes:
        input_tensors.append(graph.get_tensor_by_name(node.name + ":0"))

    G_LOGGER.verbose(f"Retrieved TensorFlow input_tensors: {input_tensors}")
    return get_tensor_metadata(input_tensors)


def get_output_metadata(graph, layerwise=False):
    graphdef = graph.as_graph_def()

    node_output_map = map_node_outputs(graphdef)

    def is_output_node(node):
        # Make sure that we're not using hanging nodes as outputs - must have at least one input.
        if len(node_output_map[node.name]) != 0 or len(node.input) == 0:
            return False

        # Tensors with no shape cannot be outputs and TensorFlow doesn't like certain ops as outputs.
        EXCLUDE_OPS = [
            "Switch",
            "FusedBatchNorm",
            "Assert",
            "NextIteration",
            "Enter",
            "LoopCond",
            "Exit",
            "Print",
            "Assign",
            "NoOp",
            "ReadVariableOp",
            "VarIsInitializedOp",
            "Const",
        ]

        # Additionally, we sometimes need to exclude entire namespaces e.g. while loops.
        EXCLUDE_NAMESPACES = ["while", "Assert"]

        if any([ex_op in node.op for ex_op in EXCLUDE_OPS]) or any([ns in node.name for ns in EXCLUDE_NAMESPACES]):
            G_LOGGER.extra_verbose(
                f"Excluding {node.name}, op {node.op} is not a valid output op or is part of an excluded namespace (Note: excluded namespaces: {EXCLUDE_NAMESPACES})"
            )
            return False

        return True

    # For layerwise mode, every layer becomes an output.
    if layerwise:
        output_nodes = list(graphdef.node)
        G_LOGGER.verbose(f"Running in layerwise mode. Marking {len(output_nodes)} layers as potential outputs")
    else:
        output_nodes = [node for node in graphdef.node if is_output_node(node)]
    G_LOGGER.extra_verbose(f"Found likely output nodes: {output_nodes}")

    output_tensors = []
    for node in output_nodes:

        tensor_name = node.name + ":0"
        try:
            tensor = graph.get_tensor_by_name(tensor_name)
            output_tensors.append(tensor)
        except KeyError:
            G_LOGGER.warning(f"Could not import: {tensor_name}. Skipping.")
    if len(output_tensors) != len(output_nodes):
        G_LOGGER.warning(
            f"Excluded {len(output_nodes) - len(output_tensors)} ops that don't seem like outputs. Use -vv/--super-verbose, or set logging verbosity to EXTRA_VERBOSE to view them."
        )

    G_LOGGER.extra_verbose(f"Found output op types in graph: {set(tensor.op.type for tensor in output_tensors)}")
    G_LOGGER.verbose(f"Retrieved TensorFlow output_tensors: {output_tensors}")
    return get_tensor_metadata(output_tensors)


def get_graph_output_names(graph):
    return list(get_output_metadata(graph).keys())


def str_from_graph(graph, show_layers=None, show_attrs=None, show_weights=None):
    show_layers = util.default(show_layers, False)
    show_attrs = util.default(show_attrs, False)
    show_weights = util.default(show_weights, False)

    graph_str = ""
    input_metadata = get_input_metadata(graph)
    output_metadata = get_output_metadata(graph)

    graph_str += f"---- {len(input_metadata)} Graph Inputs ----\n{input_metadata}\n\n"
    graph_str += f"---- {len(output_metadata)} Graph Outputs ----\n{output_metadata}\n\n"
    graph_str += f"---- {len(graph.as_graph_def().node)} Nodes ----\n"
    if show_layers:
        G_LOGGER.warning(
            "Displaying layer information is unsupported for TensorFlow graphs. "
            "Please use --show layers attrs weights if you would like to see the raw nodes"
        )
        if show_attrs or show_weights:
            for node in graph.as_graph_def().node:
                graph_str += str(node) + "\n"
        graph_str += "\n"
    return util.indent_block(graph_str, level=0)
