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
import copy
import fnmatch
from collections import OrderedDict

from polygraphy import mod, util
from polygraphy.common import TensorMetadata
from polygraphy.datatype import DataType
from polygraphy.logger import G_LOGGER, LogMode

gs = mod.lazy_import("onnx_graphsurgeon")
onnx = mod.lazy_import("onnx")
onnx_numpy_helper = mod.lazy_import("onnx.numpy_helper")


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

    all_outputs = [
        output
        for node in model.graph.node
        if node.op_type != "Constant"
        for output in node.output
        # Skip empty-name outputs, which ONNX uses to denote unused optional
        # outputs of multi-output ops (e.g. LSTM). They are not real tensors and
        # marking them would create untyped graph outputs that break inference.
        if output
    ]
    if include_inputs:
        all_outputs += [inp.name for inp in model.graph.input]
    all_outputs = util.unique_list(all_outputs)
    return all_outputs


def _check_has_tensors(model, outputs):
    all_outputs = all_tensor_names(model, include_inputs=True)
    util.check_sequence_contains(
        all_outputs, outputs, name="the model", items_name="outputs", check_extra=False
    )


def _expand_wildcard_patterns(names, candidates):
    """Expand fnmatch wildcard patterns in `names` against `candidates`."""
    result = []
    for name in names:
        if any(c in name for c in ("*", "?", "[", "]")):
            matches = fnmatch.filter(candidates, name)
            if not matches:
                G_LOGGER.warning(f"No tensors matched wildcard pattern: '{name}'")
            result.extend(matches)
        else:
            result.append(name)
    return result


def mark_outputs(model, outputs):
    # Clear the old outputs
    while model.graph.output:
        model.graph.output.pop()

    outputs = _expand_wildcard_patterns(outputs, all_tensor_names(model))
    outputs = util.unique_list(outputs)
    if not outputs:
        G_LOGGER.critical(
            "No outputs were selected. Please check your wildcard patterns."
        )
    _check_has_tensors(model, outputs)

    value_info_map = {t.name: t for t in model.graph.value_info}
    out_tensors = []
    for output in outputs:
        value_info = value_info_map.get(
            output, onnx.helper.make_empty_tensor_value_info(output)
        )
        out_tensors.append(value_info)

    G_LOGGER.ultra_verbose(f"Marked output tensors in ONNX model: {out_tensors}")
    model.graph.output.extend(out_tensors)
    return model


def mark_layerwise(model):
    # Add all non-constant node outputs as graph outputs
    model = mark_outputs(model, all_tensor_names(model))
    return model


def mark_by_op_type(model, op_types):
    """
    Mark outputs of all nodes whose op_type matches any of the given types (case-insensitive).

    If a requested op type is not found in the model, logs a non-critical error and shows
    similar op types using difflib.

    Args:
        model (onnx.ModelProto): The ONNX model to modify.
        op_types (Sequence[str]): Op type names to match.

    Returns:
        onnx.ModelProto: The model with matched node outputs marked.
    """
    import difflib

    all_op_types = {node.op_type for node in model.graph.node}
    lower_to_canonical = {t.lower(): t for t in all_op_types}

    outputs = []
    for requested in op_types:
        canonical = lower_to_canonical.get(requested.lower())
        if canonical is None:
            similar = difflib.get_close_matches(
                requested, all_op_types, n=5, cutoff=0.6
            )
            hint = (
                f" Did you mean one of: {similar}?"
                if similar
                else f" Available op types: {sorted(all_op_types)}"
            )
            G_LOGGER.error(
                f"No ONNX nodes of op type '{requested}' were found in the model.{hint}"
            )
        else:
            for node in model.graph.node:
                if node.op_type == canonical:
                    for out in node.output:
                        if out:
                            outputs.append(out)

    if outputs:
        model = mark_outputs(model, outputs)
    return model


def unmark_outputs(model, outputs):
    outputs = _expand_wildcard_patterns(outputs, all_tensor_names(model))
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
    return DataType.from_dtype(onnx_type, source_module="onnx")


def get_values(tensor):
    try:
        return onnx_numpy_helper.to_array(tensor)
    except Exception as err:
        G_LOGGER.error(
            f"Failed to load weights.\nNote: Error was: {err}", mode=LogMode.ONCE
        )
    return "<error: failed to load weights>"


def get_tensor_metadata(tensors):
    metadata = TensorMetadata()
    for tensor in tensors:
        metadata.add(name=tensor.name, dtype=get_dtype(tensor), shape=get_shape(tensor))
    return metadata


def get_input_metadata(graph):
    # Some "inputs" are actually weights with initalizers, so we need to eliminate those.
    initializer_names = {tensor.name for tensor in graph.initializer}
    input_tensors = [
        tensor for tensor in graph.input if tensor.name not in initializer_names
    ]
    return get_tensor_metadata(input_tensors)


def get_output_metadata(graph):
    return get_tensor_metadata(graph.output)


def str_from_onnx(model, show_layers=None, show_attrs=None, show_weights=None):
    """
    Converts an ONNX model to a human-readable string representation.

    Args:
        model (onnx.ModelProto): The ONNX model.
        show_layers (bool): Whether to display per-layer information.
        show_attrs (bool): Whether to display per-layer attributes.
        show_weights (bool): Whether to display the value of weights.

    Returns:
        str
    """
    from polygraphy.tools.inspect.subtool.model.extractors import graph_data_from_onnx
    from polygraphy.tools.inspect.subtool.model.text import str_from_graph_data

    show_layers = util.default(show_layers, False)
    show_attrs = util.default(show_attrs, False)
    show_weights = util.default(show_weights, False)

    graph_data = graph_data_from_onnx(model, show_weights=show_weights)
    return str_from_graph_data(
        graph_data,
        show_layers=show_layers,
        show_attrs=show_attrs,
        show_weights=show_weights,
    )


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
            tensor.dtype = DataType.to_dtype(
                DataType.from_dtype(layerwise_meta[tensor.name].dtype), "onnx"
            )


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


def get_unbounded_dds_tensors(graph):
    graph.toposort()
    # A dict of operators that might produce a output tensor with unbounded DDS, when the value of the input tensor
    # at the corresponding index is a runtime value. For example, "Range" => "1" means that if the input 1 of the Range
    # operator is a runtime value, e.g. not a const tensor or an initializer, then the Range output tensor size is unbounded.
    dispatcher_dict = {
        "Range": [1],  # the limit input of the Range operator
        "Pad": [1],  # the pads input of the Pad operator
        "Resize": [3],  # the sizes input of the Resize operator
        "Tile": [1],  # the repeats input of the Tile operator
        "Expand": [1],  # the shape input of the Expand operator
    }

    # Check if the given operator produces a output tensor with unbounded DDS.
    def check_op(node, const_tensor_set):
        # Check if the operator is inside the dispatcher dict.
        if node.op in dispatcher_dict:
            input_idx_list = dispatcher_dict[node.op]
            for input_idx in input_idx_list:
                if input_idx < len(node.inputs):
                    input_tensor = node.inputs[input_idx]
                    # Check if the corresponding input tensor is a runtime value and its producer is not Min operator.
                    # If a tensor is produced by a Min operator, its upper bound has already been set.
                    if (
                        input_tensor.name not in const_tensor_set
                        and len(input_tensor.inputs) >= 1
                        and input_tensor.inputs[0].op != "Min"
                    ):
                        return input_tensor
        return None

    # Find all constant tensors.
    def get_const_tensors(graph):
        return {
            tensor.name
            for tensor in graph.tensors().values()
            if isinstance(tensor, gs.Constant)
        }

    # Find all dynamic shape symbols, customers will set upper bounds for these symbols when building the model in TensorRT.
    def get_dynamic_shapes(graph):
        dynamic_shape_set = set()
        for tensor in graph.inputs:
            for shape in tensor.shape:
                if isinstance(shape, str):
                    dynamic_shape_set.add(shape)
        return dynamic_shape_set

    # Find all tensors with unbounded DDS.
    def get_target_tensors(graph):
        # Find dynamic shapes, these shapes should have upper bounds in TensorRT.
        dynamic_shape_set = get_dynamic_shapes(graph)

        # Find const tensors. For those operators in the dispatch dict, constant inputs will not introduce outputs with unbounded DDS.
        const_tensor_set = get_const_tensors(graph)

        # Our target is to find those input tensors that cause its consumer nodes generated unbounded outputs.
        # If a tensor has named dimensions that appeared before in its symbolic shape, it means that the shape is *not* data dependent,
        # and so will have an upper bound.
        target_tensor_names = set()
        target_tensor_list = []
        for node in graph.nodes:
            check_node = False
            # Check if the node's output contains a new introduced dynamic shape.
            for tensor in node.outputs:
                # Always check nodes if tensor.shape is None.
                # This happens when the symbolic inference does not work correctly due to some restrictions.
                if tensor.shape is None:
                    check_node = True
                else:
                    for shape in tensor.shape:
                        # If a shape is a dynamic shape, then it is a str.
                        # Only check the node that first introduced the dynamic shape.
                        if isinstance(shape, str) and shape not in dynamic_shape_set:
                            dynamic_shape_set.add(shape)
                            check_node = True
            # Check if the node will generate an unbounded output size.
            if check_node:
                target_tensor = check_op(node, const_tensor_set)
                # Avoid duplication.
                if (
                    target_tensor is not None
                    and target_tensor.name not in target_tensor_names
                ):
                    target_tensor_names.add(target_tensor.name)
                    target_tensor_list.append(target_tensor)
        return target_tensor_list

    return get_target_tensors(graph)
