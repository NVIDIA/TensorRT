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

from typing import List, Sequence, Union

import numpy as np
import onnx
import onnx.numpy_helper
from onnx_graphsurgeon.exporters.base_exporter import BaseExporter
from onnx_graphsurgeon.ir.function import Function
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.ir.tensor import (
    Constant,
    SparseValues,
    LazyValues,
    Tensor,
    Variable,
)
from onnx_graphsurgeon.logger import G_LOGGER
from onnx_graphsurgeon.util import misc


def dtype_to_onnx(dtype: Union[np.dtype, "onnx.TensorProto.DataType"]) -> int:
    if isinstance(dtype, int):
        return dtype
    return onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))


def check_duplicate_node_names(nodes: Sequence[Node], level=G_LOGGER.WARNING):
    # Check if node names are unique. If not, log based on severity.

    # Note:
    # Empty string or None attribute values are not considered duplicates.
    name_map = {}
    for node in nodes:
        if not node.name:
            continue
        if node.name in name_map:
            msg = "Found distinct Nodes that share the same name:\n[id: {:}]:\n {:}---\n[id: {:}]:\n {:}\n".format(
                id(name_map[node.name]),
                name_map[node.name],
                id(node),
                node,
            )
            G_LOGGER.log(msg, level)
        else:
            name_map[node.name] = node


def update_import_domains(graph):
    # Update the import_domains field to contain the graph's ONNX opset,
    # as well as other non-ONNX domains which are used by this graph's nodes.
    # Returns the updated value of the import_domains field.

    # Add domain of the standard ONNX opset.
    if graph.import_domains is None:
        graph.import_domains = [onnx.helper.make_opsetid("", graph.opset)]

    # Crawl over all nodes in this graph and its subgraphs, and add the nodes' domains.
    all_used_domains = {node.domain for node in graph.nodes}
    for subgraph in graph.subgraphs(recursive=True):
        all_used_domains |= {n.domain for n in subgraph.nodes}
    all_used_domains.discard(None)

    # Update self.import_domains with any missing domains.
    current_domains = {opsetid.domain for opsetid in graph.import_domains}
    DEFAULT_CUSTOM_OPSET_VERSION = 1
    for used_domain in all_used_domains:
        if used_domain not in current_domains:
            graph.import_domains.append(onnx.helper.make_opsetid(used_domain, DEFAULT_CUSTOM_OPSET_VERSION))
            current_domains.add(used_domain)
    return graph.import_domains


# Converts a fp32 gs.Constant to a bf16 onnx.TensorProto
def tensor_to_onnx_bf16(tensor: Constant):

    # Converts the fp32 numpy array to bf16 values and store in a uint16 numpy array
    def np_float32_to_bf16_as_uint16(arr):
        new_arr = np.empty(arr.size, dtype=np.uint16)
        flatten = arr.flatten()
        for i in range(arr.size):
            new_arr[i] = onnx.helper.float32_to_bfloat16(flatten[i])
        return new_arr.reshape(arr.shape)

    arr_bf16_as_uint16 = np_float32_to_bf16_as_uint16(tensor.values)

    onnx_tensor = onnx.TensorProto()
    onnx_tensor.data_type = onnx.TensorProto.BFLOAT16
    onnx_tensor.dims.extend(arr_bf16_as_uint16.shape)
    onnx_tensor.raw_data = arr_bf16_as_uint16.tobytes()

    return onnx_tensor


class OnnxExporter(BaseExporter):
    @staticmethod
    def export_tensor_proto(tensor: Constant) -> onnx.TensorProto:
        # Do *not* load LazyValues into an intermediate numpy array - instead, use
        # the original onnx.TensorProto directly.
        if isinstance(tensor._values, LazyValues):
            onnx_tensor = tensor._values.tensor
        else:
            if dtype_to_onnx(tensor.dtype) != dtype_to_onnx(tensor.export_dtype):
                assert tensor.dtype == np.float32, (
                    f"Cannot convert onnx dtype {dtype_to_onnx(tensor.dtype)} to {dtype_to_onnx(tensor.export_dtype)}."
                    "Only float32 to bfloat16 is supported"
                )
                assert tensor.export_dtype == onnx.TensorProto.BFLOAT16, (
                    f"Cannot convert onnx dtype {dtype_to_onnx(tensor.dtype)} to {dtype_to_onnx(tensor.export_dtype)}."
                    "Only float32 to bfloat16 is supported"
                )
                onnx_tensor = tensor_to_onnx_bf16(tensor)
            else:
                onnx_tensor = onnx.numpy_helper.from_array(tensor.values)

            if tensor.data_location is not None:
                onnx_tensor.data_location = tensor.data_location
        onnx_tensor.name = tensor.name
        return onnx_tensor

    @staticmethod
    def export_sparse_tensor_proto(tensor: Constant) -> onnx.SparseTensorProto:
        return tensor._values.tensor

    @staticmethod
    def export_value_info_proto(tensor: Tensor, do_type_check: bool) -> onnx.ValueInfoProto:
        if do_type_check and tensor.dtype is None:
            G_LOGGER.critical(
                "Graph input and output tensors must include dtype information. Please set the dtype attribute for: {:}".format(
                    tensor
                )
            )

        if tensor.dtype is not None:
            if isinstance(tensor, Constant) or tensor.type == "tensor_type":
                onnx_tensor = onnx.helper.make_tensor_value_info(tensor.name, dtype_to_onnx(tensor.dtype), tensor.shape)
            elif tensor.type == "sequence_type":
                onnx_tensor = onnx.helper.make_tensor_sequence_value_info(
                    tensor.name, dtype_to_onnx(tensor.dtype), tensor.shape
                )
            elif tensor.type == "sparse_tensor_type":
                onnx_tensor = onnx.helper.make_sparse_tensor_value_info(
                    tensor.name, dtype_to_onnx(tensor.dtype), tensor.shape
                )
        else:
            onnx_tensor = onnx.helper.make_empty_tensor_value_info(tensor.name)
        return onnx_tensor

    @staticmethod
    def export_attributes(attrs: dict) -> List[onnx.AttributeProto]:
        onnx_attrs: List[onnx.AttributeProto] = []
        for key, val in attrs.items():
            if isinstance(val, Tensor):
                val = OnnxExporter.export_tensor_proto(val)
            elif isinstance(val, Graph):
                # Subgraphs don't need to have types specified for their tensors.
                val = OnnxExporter.export_graph(val, do_type_check=False)
            elif isinstance(val, Node.AttributeRef):
                onnx_attr = onnx.AttributeProto()
                onnx_attr.name = key
                onnx_attr.type = misc.convert_to_onnx_attr_type(val.type)

                # Netron has a bug which makes it crash if a Tensor attribute has no tensor data.
                # So provide some meaningless tensor data for Netron to read.
                if val.type == Tensor:
                    tensor_proto = OnnxExporter.export_tensor_proto(Constant("", np.array([0], dtype=np.float32)))
                    onnx_attr.t.CopyFrom(tensor_proto)

                onnx_attr.ref_attr_name = val.name
                onnx_attrs.append(onnx_attr)
                continue
            elif isinstance(val, type):
                # May be a numpy type
                try:
                    val = dtype_to_onnx(val)
                except TypeError:
                    pass
            onnx_attrs.append(onnx.helper.make_attribute(key, val))
        return onnx_attrs

    @staticmethod
    def export_node(node: Node) -> onnx.NodeProto:
        # Cannot pass in attrs directly as make_node will change the order
        onnx_node = onnx.helper.make_node(
            node.op,
            inputs=[t.name for t in node.inputs],
            outputs=[t.name for t in node.outputs],
            name=node.name,
            domain=node.domain,
        )
        onnx_node.attribute.extend(OnnxExporter.export_attributes(node.attrs))
        return onnx_node

    @staticmethod
    def export_function(func: Function) -> onnx.FunctionProto:
        """
        Export an onnx-graphsurgeon Function to an ONNX FunctionProto.

        Args:
            func (Function): The function to export.
        """
        # Unlike onnx Graphs, onnx Functions don't have an 'initializer' field.
        # So we need to replace all Constant tensors with onnx Constant nodes which produce them.
        # We need to be careful to (a) preserve topological ordering and (b) not make the new nodes visible to the user.
        func_nodes = func.nodes.copy()
        new_const_nodes = []
        for tensor in func.tensors().values():
            if isinstance(tensor, Constant):
                # Copying the tensor prevents the new node from appearing in the Constant tensor's inputs.
                new_const_nodes.append(Node("Constant", attrs={"value": tensor}, outputs=[tensor.copy()]))
        # Const nodes have no inputs, so this maintains a topological ordering.
        func_nodes = new_const_nodes + func_nodes

        check_duplicate_node_names(func_nodes, level=G_LOGGER.WARNING)
        nodes = [OnnxExporter.export_node(node) for node in func_nodes]

        # Update the import_domains field to include all domains used by this function.
        opset_imports = update_import_domains(func)

        onnx_inputs = [inp.name for inp in func.inputs]
        onnx_outputs = [out.name for out in func.outputs]

        attributes = []
        attribute_protos = dict()
        for attr_name, default_val in func.attrs.items():
            if default_val is None:
                attributes.append(attr_name)
            else:
                attribute_protos[attr_name] = default_val
        attribute_protos = OnnxExporter.export_attributes(attribute_protos)

        return onnx.helper.make_function(
            func.domain or "",
            func.name,
            onnx_inputs,
            onnx_outputs,
            nodes,
            opset_imports,
            attributes=attributes,
            attribute_protos=attribute_protos,
            doc_string=func.doc_string,
        )

    @staticmethod
    def export_graph(graph: Graph, do_type_check=True) -> onnx.GraphProto:
        """
        Export an onnx-graphsurgeon Graph to an ONNX GraphProto.

        Args:
            graph (Graph): The graph to export.

            do_type_check (bool): Whether to check that input and output tensors have data types defined, and fail if not.
                                  Defaults to True.
        """
        check_duplicate_node_names(graph.nodes, level=G_LOGGER.WARNING)
        nodes = [OnnxExporter.export_node(node) for node in graph.nodes]
        inputs = [OnnxExporter.export_value_info_proto(inp, do_type_check) for inp in graph.inputs]
        outputs = [OnnxExporter.export_value_info_proto(out, do_type_check) for out in graph.outputs]
        tensor_map = graph.tensors()
        initializer = [
            OnnxExporter.export_tensor_proto(tensor)
            for tensor in tensor_map.values()
            if isinstance(tensor, Constant)
            and not isinstance(tensor._values, SparseValues)
        ]
        sparse_initializer = [
            OnnxExporter.export_sparse_tensor_proto(tensor)
            for tensor in tensor_map.values()
            if isinstance(tensor, Constant) and isinstance(tensor._values, SparseValues)
        ]

        # Remove inputs and outputs to export ValueInfoProtos
        for tensor in graph.inputs + graph.outputs:
            if tensor.name in tensor_map:
                del tensor_map[tensor.name]

        # Omit tensors from value_info if we don't know their shape/dtype
        def has_value_info(tensor):
            return isinstance(tensor, Variable) and (tensor.dtype is not None or tensor.shape is not None)

        value_info = [
            OnnxExporter.export_value_info_proto(tensor, do_type_check)
            for tensor in tensor_map.values()
            if has_value_info(tensor)
        ]

        return onnx.helper.make_graph(
            nodes=nodes,
            name=graph.name,
            inputs=inputs,
            outputs=outputs,
            initializer=initializer,
            sparse_initializer=sparse_initializer,
            doc_string=graph.doc_string,
            value_info=value_info,
        )


def export_onnx(graph: Graph, do_type_check=True, **kwargs) -> "onnx.ModelProto":
    """
    Exports an onnx-graphsurgeon Graph to an ONNX model.

    Args:
        graph (Graph): The graph to export

        do_type_check (bool): Whether to check that input and output tensors have data types defined, and fail if not.
                              Defaults to True.
        kwargs: Additional arguments to onnx.helper.make_model

    Returns:
        onnx.ModelProto: A corresponding ONNX model.
    """
    onnx_graph = OnnxExporter.export_graph(graph, do_type_check=do_type_check)
    onnx_functions = [OnnxExporter.export_function(func) for func in graph.functions]
    kwargs["functions"] = onnx_functions

    if "opset_imports" not in kwargs:
        kwargs["opset_imports"] = update_import_domains(graph)

    model = onnx.helper.make_model(onnx_graph, **kwargs)
    model.producer_name = graph.producer_name
    model.producer_version = graph.producer_version
    return model
