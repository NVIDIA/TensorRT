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
import copy
from collections import OrderedDict

from polygraphy import mod, util, constants
from polygraphy.common import TensorMetadata
from polygraphy.logger import G_LOGGER, LogMode

gs = mod.lazy_import("onnx_graphsurgeon")
numpy_helper = mod.lazy_import("onnx.numpy_helper")
onnx = mod.lazy_import("onnx")


def get_num_nodes(model):
    def _get_num_graph_nodes(graph):
        num_nodes = len(graph.node)
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    num_nodes += _get_num_graph_nodes(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attr.graphs:
                        num_nodes += _get_num_graph_nodes(subgraph)
        return num_nodes

    return _get_num_graph_nodes(model.graph)


def all_tensor_names(model, include_inputs=None):
    include_inputs = util.default(include_inputs, False)

    all_outputs = [output for node in model.graph.node if node.op_type != "Constant" for output in node.output]
    if include_inputs:
        all_outputs += [inp.name for inp in model.graph.input]
    all_outputs = util.unique_list(all_outputs)
    return all_outputs


def _check_has_tensors(model, outputs):
    all_outputs = all_tensor_names(model, include_inputs=True)
    util.check_sequence_contains(all_outputs, outputs, name="the model", items_name="outputs", check_extra=False)


def mark_outputs(model, outputs):
    # Clear the old outputs
    while model.graph.output:
        model.graph.output.pop()

    outputs = util.unique_list(outputs)
    _check_has_tensors(model, outputs)

    value_info_map = {t.name: t for t in model.graph.value_info}
    out_tensors = []
    for output in outputs:
        value_info = value_info_map.get(output, onnx.helper.make_empty_tensor_value_info(output))
        out_tensors.append(value_info)

    G_LOGGER.ultra_verbose(f"Marked output tensors in ONNX model: {out_tensors}")
    model.graph.output.extend(out_tensors)
    return model


def mark_layerwise(model):
    # Add all non-constant node outputs as graph outputs
    model = mark_outputs(model, all_tensor_names(model))
    return model


def unmark_outputs(model, outputs):
    outputs = util.unique_list(outputs)
    _check_has_tensors(model, outputs)

    cur_outputs = []
    while model.graph.output:
        cur_outputs.append(model.graph.output.pop())
    cur_outputs = list(reversed(cur_outputs))  # Preserve ordering

    for out in cur_outputs:
        if out.name not in outputs:
            model.graph.output.extend([out])

    return model


def get_shape(tensor):
    shape = []
    if isinstance(tensor, onnx.TensorProto):
        shape = tensor.dims
    else:
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            elif dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append(-1)
    return shape


def get_dtype(tensor):
    if isinstance(tensor, onnx.TensorProto):
        onnx_type = tensor.data_type
    else:
        onnx_type = tensor.type.tensor_type.elem_type
    if onnx_type in onnx.mapping.TENSOR_TYPE_TO_NP_TYPE:
        return onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_type]
    return None


def get_values(tensor):
    try:
        return numpy_helper.to_array(tensor)
    except Exception as err:
        G_LOGGER.error(f"Failed to load weights.\nNote: Error was: {err}", mode=LogMode.ONCE)
    return "<error: failed to load weights>"


def get_tensor_metadata(tensors):
    metadata = TensorMetadata()
    for tensor in tensors:
        metadata.add(name=tensor.name, dtype=get_dtype(tensor), shape=get_shape(tensor))
    return metadata


def get_input_metadata(graph):
    # Some "inputs" are actually weights with initalizers, so we need to eliminate those.
    initializer_names = {tensor.name for tensor in graph.initializer}
    input_tensors = [tensor for tensor in graph.input if tensor.name not in initializer_names]
    return get_tensor_metadata(input_tensors)


def get_output_metadata(graph):
    return get_tensor_metadata(graph.output)


def str_from_onnx(model, show_layers=None, show_attrs=None, show_weights=None):
    """
    Converts an ONNX Graph to a human-readable representation

    Args:
        graph (onnx.GraphProto): The onnx graph.
        show_layers (bool): Whether to display per-layer information.
        show_attrs (bool): Whether to display per-layer attributes.
        show_weights (bool): Whether to display the value of weights.

    Returns:
        str
    """
    show_layers = util.default(show_layers, False)
    show_attrs = util.default(show_attrs, False)
    show_weights = util.default(show_weights, False)

    def get_opset():
        default_opset = "Unknown"
        other_opsets = {}
        for info in model.opset_import:
            if not info.domain:
                default_opset = info.version
            else:
                other_opsets[info.domain] = info.version
        return default_opset, other_opsets

    default_opset, other_opsets = get_opset()
    onnx_str = ""
    onnx_str += f"Name: {model.graph.name} | ONNX Opset: {default_opset}"
    if other_opsets:
        onnx_str += f" | Other Opsets: {other_opsets}"
    onnx_str += "\n\n"

    onnx_str += str_from_onnx_graph(
        model.graph, tensors={}, show_layers=show_layers, show_attrs=show_attrs, show_weights=show_weights
    )
    return onnx_str


def str_from_onnx_graph(graph, tensors, show_layers, show_attrs, show_weights, indent_level=0):

    input_metadata = get_input_metadata(graph)
    output_metadata = get_output_metadata(graph)
    initializer_metadata = get_tensor_metadata(graph.initializer)

    # Subgraph inputs should remain separate from each other, hence copy the tensors map
    tensors = copy.copy(tensors)
    tensors.update(get_tensor_metadata(graph.value_info))
    tensors.update(initializer_metadata)
    tensors.update(input_metadata)
    tensors.update(output_metadata)

    graph_type = "Graph" if indent_level == 0 else "Subgraph"

    onnx_str = ""
    if show_attrs and graph.doc_string:
        onnx_str += f"---- Docstring ----\n{graph.doc_string}\n\n"

    onnx_str += f"---- {len(input_metadata)} {graph_type} Input(s) ----\n{input_metadata}\n\n"
    onnx_str += f"---- {len(output_metadata)} {graph_type} Output(s) ----\n{output_metadata}\n\n"

    onnx_str += f"---- {len(initializer_metadata)} Initializer(s) ----\n"
    if show_weights:
        for init in graph.initializer:
            onnx_str += f"Initializer | {init.name} [dtype={get_dtype(init)}, shape={get_shape(init)}] | Values:\n{util.indent_block(str(get_values(init)))}\n\n"
        if not graph.initializer:
            onnx_str += "{}\n\n"
    elif show_layers:
        onnx_str += str(initializer_metadata)
        onnx_str += "\n\n"
    else:
        onnx_str += "\n"

    def get_names_and_meta(names):
        names_lst = []
        metadata = TensorMetadata()
        for name in names:
            dtype, shape = tensors.get(name, (None, None))
            if name in initializer_metadata:
                name = f"Initializer | {name}"
            names_lst.append(name)
            metadata.add(name=name, dtype=dtype, shape=shape)
        return names_lst, metadata

    # Maps values from the AttributeType enum to their string representations, e.g., {1: "FLOAT"}
    ATTR_TYPE_MAPPING = dict(zip(onnx.AttributeProto.AttributeType.values(), onnx.AttributeProto.AttributeType.keys()))

    # Maps an ONNX attribute to the corresponding Python property
    ONNX_PYTHON_ATTR_MAPPING = {
        "FLOAT": "f",
        "INT": "i",
        "STRING": "s",
        "TENSOR": "t",
        "GRAPH": "g",
        "FLOATS": "floats",
        "INTS": "ints",
        "STRINGS": "strings",
    }

    def attrs_to_dict(attrs):
        attr_dict = OrderedDict()
        for attr in attrs:

            def process_attr(attr_str: str):
                processed = getattr(attr, ONNX_PYTHON_ATTR_MAPPING[attr_str])
                if attr_str == "STRING":
                    processed = processed.decode()
                elif attr_str == "TENSOR":
                    tensor_str = f"Tensor: [dtype={get_dtype(processed)}, shape={get_shape(processed)}]"
                    if show_weights:
                        tensor_str += " | Values:\n" + util.indent_block(str(get_values(processed)))
                    processed = tensor_str
                elif attr_str == "GRAPH":
                    processed = "\n" + str_from_onnx_graph(
                        processed,
                        tensors,
                        indent_level=indent_level + 2,
                        show_layers=show_layers,
                        show_attrs=show_attrs,
                        show_weights=show_weights,
                    )
                elif attr_str == "FLOATS" or attr_str == "INTS":
                    # Proto hacky list to normal Python list
                    processed = [p for p in processed]
                elif attr_str == "STRINGS":
                    processed = [p.decode() for p in processed]
                return processed

            if attr.type in ATTR_TYPE_MAPPING:
                attr_str = ATTR_TYPE_MAPPING[attr.type]
                if attr_str in ONNX_PYTHON_ATTR_MAPPING:
                    attr_dict[attr.name] = process_attr(attr_str)
                else:
                    G_LOGGER.warning(f"Attribute of type {attr_str} is currently unsupported. Skipping attribute.")
            else:
                G_LOGGER.warning(
                    f"Attribute type: {attr.type} was not recognized. Was the graph generated with a newer IR version than the installed `onnx` package? Skipping attribute."
                )
        return attr_dict

    onnx_str += f"---- {len(graph.node)} Node(s) ----\n"
    if show_layers:
        for index, node in enumerate(graph.node):
            input_names, input_meta = get_names_and_meta(node.input)
            output_names, output_meta = get_names_and_meta(node.output)

            onnx_str += util.str_from_layer(
                "Node", index, node.name, node.op_type, input_names, input_meta, output_names, output_meta
            )

            if show_attrs:
                attrs = attrs_to_dict(node.attribute)
                if attrs:
                    onnx_str += util.indent_block("---- Attributes ----") + "\n"
                for key, val in attrs.items():
                    attr_str = ""
                    if node.name:
                        attr_str += f"{node.name}."
                    onnx_str += util.indent_block(f"{attr_str}{key} = {val}") + "\n"
            onnx_str += "\n"

    return util.indent_block(onnx_str, indent_level)


##
## ONNX-GraphSurgeon utilities
##


def meta_from_gs_tensors(tensors):
    """Get TensorMetadata from a list of ONNX-GraphSurgeon tensors"""
    meta = TensorMetadata()
    for tensor in tensors:
        meta.add(tensor.name, tensor.dtype, tensor.shape)
    return meta


def set_shapes_from_layerwise_meta(graph, layerwise_meta):
    """
    Args:
        graph (gs.Graph): An ONNX graphsurgeon graph.
        layerwise_meta (TensorMetadata): Metadata for tensors in the graph.
    """
    for tensor in graph.tensors().values():
        if isinstance(tensor, gs.Variable) and tensor.name in layerwise_meta:
            tensor.shape = layerwise_meta[tensor.name].shape
            tensor.dtype = layerwise_meta[tensor.name].dtype


def lower_constant_nodes(graph):
    """Converts the outputs of Constant nodes into constant tensors, removing the nodes"""
    remove_nodes = set()
    with graph.node_ids():
        for node in graph.nodes:
            if node.op == "Constant" and "value" in node.attrs:
                node.outputs[0].to_constant(node.attrs["value"].values)
                remove_nodes.add(node.id)
        # Iterate from the end so we don't shift the list under us.
        for node_id in sorted(remove_nodes, reverse=True):
            del graph.nodes[node_id]
    return graph
