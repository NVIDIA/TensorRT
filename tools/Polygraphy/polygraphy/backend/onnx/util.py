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
import copy
from collections import OrderedDict

from polygraphy import mod, util
from polygraphy.common import TensorMetadata
from polygraphy.logger import G_LOGGER

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


def all_tensor_names(model):
    all_outputs = [output for node in model.graph.node if node.op_type != "Constant" for output in node.output]
    all_outputs = util.unique_list(all_outputs)
    return all_outputs


def check_outputs_not_found(not_found, all_outputs):
    if not_found:
        G_LOGGER.critical(
            "The following outputs were not found: {:}.\nNote: Available tensors:\n\t{:}".format(
                not_found, "\n\t".join(all_outputs)
            )
        )


def mark_outputs(model, outputs):
    # Clear the old outputs
    while model.graph.output:
        model.graph.output.pop()

    all_outputs = all_tensor_names(model)
    all_outputs_set = set(all_outputs)
    value_info_map = {t.name: t for t in model.graph.value_info}

    out_tensors = []
    not_found = set()
    for output in outputs:
        if output in all_outputs_set:
            value_info = value_info_map.get(output, onnx.helper.make_empty_tensor_value_info(output))
            out_tensors.append(value_info)
        else:
            not_found.add(output)

    check_outputs_not_found(not_found, all_outputs)
    model.graph.output.extend(out_tensors)
    return model


def mark_layerwise(model):
    # Add all non-constant node outputs as graph outputs
    model = mark_outputs(model, all_tensor_names(model))
    return model


def unmark_outputs(model, outputs):
    outputs = set(outputs)

    cur_outputs = []
    while model.graph.output:
        cur_outputs.append(model.graph.output.pop())
    cur_outputs = list(reversed(cur_outputs))  # Preserve ordering

    unmarked_outputs = set()
    for out in cur_outputs:
        if out.name not in outputs:
            model.graph.output.extend([out])
        else:
            unmarked_outputs.add(out.name)

    not_found = outputs - unmarked_outputs
    check_outputs_not_found(not_found, [t.name for t in model.graph.output])
    return model


def get_shape(tensor):
    shape = []
    if isinstance(tensor, onnx.TensorProto):
        shape = tensor.dims
    else:
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
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
    return numpy_helper.to_array(tensor)


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


def str_from_onnx(model, mode="full"):
    """
    Converts an ONNX Graph to a human-readable representation

    Args:
        graph (onnx.GraphProto): The onnx graph.
        mode (str): Controls what is displayed. Choices: ["none", "basic", "attrs", "full"]

    Returns:
        str
    """

    def get_opset():
        try:
            return model.opset_import[0].version
        except:
            G_LOGGER.warning("Model does not contain opset information!")
            return None

    onnx_str = ""
    onnx_str += "Name: {:} | Opset: {:}\n".format(model.graph.name, get_opset())
    onnx_str += "\n"

    onnx_str += str_from_onnx_graph(model.graph, mode=mode, tensors={})
    return onnx_str


def str_from_onnx_graph(graph, mode, tensors, indent_level=0):
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
    onnx_str += "---- {:} {:} Input(s) ----\n{:}\n\n".format(len(input_metadata), graph_type, input_metadata)
    onnx_str += "---- {:} {:} Output(s) ----\n{:}\n\n".format(len(output_metadata), graph_type, output_metadata)

    onnx_str += "---- {:} Initializer(s) ----\n".format(len(initializer_metadata))
    if mode == "full":
        for init in graph.initializer:
            onnx_str += "Initializer | {:} [dtype={:}, shape={:}] | Values:\n{:}\n\n".format(
                init.name, get_dtype(init), get_shape(init), util.indent_block(str(get_values(init)))
            )
        if not graph.initializer:
            onnx_str += "{}\n\n"
    elif mode != "none":
        onnx_str += str(initializer_metadata)
        onnx_str += "\n\n"
    else:
        onnx_str += "\n"

    def metadata_from_names(names):
        metadata = TensorMetadata()
        for name in names:
            dtype, shape = tensors.get(name, (None, None))
            if name in initializer_metadata:
                name = "Initializer | {:}".format(name)
            metadata.add(name=name, dtype=dtype, shape=shape)
        return metadata

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
                    tensor_str = "Tensor: [dtype={:}, shape={:}]".format(get_dtype(processed), get_shape(processed))
                    if mode == "full":
                        tensor_str += " | Values:\n" + util.indent_block(str(get_values(processed)))
                    processed = tensor_str
                elif attr_str == "GRAPH":
                    processed = "\n" + str_from_onnx_graph(processed, mode, tensors, indent_level=indent_level + 2)
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
                    G_LOGGER.warning(
                        "Attribute of type {:} is currently unsupported. Skipping attribute.".format(attr_str)
                    )
            else:
                G_LOGGER.warning(
                    "Attribute type: {:} was not recognized. Was the graph generated with a newer IR "
                    "version than the installed `onnx` package? Skipping attribute.".format(attr.type)
                )
        return attr_dict

    onnx_str += "---- {:} Node(s) ----\n".format(len(graph.node))
    if mode != "none":
        for index, node in enumerate(graph.node):
            input_info = metadata_from_names(node.input)
            output_info = metadata_from_names(node.output)

            onnx_str += util.str_from_layer("Node", index, node.name, node.op_type, input_info, output_info)

            if mode in ["attrs", "full"]:
                attrs = attrs_to_dict(node.attribute)
                if attrs:
                    onnx_str += util.indent_block("---- Attributes ----") + "\n"
                for key, val in attrs.items():
                    attr_str = ""
                    if node.name:
                        attr_str += "{:}.".format(node.name)
                    onnx_str += util.indent_block("{:}{:} = {:}".format(attr_str, key, val)) + "\n"
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
