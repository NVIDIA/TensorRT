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

import numpy as np
import onnx
import onnx.numpy_helper
from onnx_graphsurgeon.exporters.base_exporter import BaseExporter
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.ir.tensor import Constant, LazyValues, Tensor, Variable
from onnx_graphsurgeon.logger.logger import G_LOGGER


def dtype_to_onnx(dtype: np.dtype) -> int:
    return onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]


class OnnxExporter(BaseExporter):
    @staticmethod
    def export_tensor_proto(tensor: Constant) -> onnx.TensorProto:
        # Do *not* load LazyValues into an intermediate numpy array - instead, use
        # the original onnx.TensorProto directly.
        if isinstance(tensor._values, LazyValues):
            onnx_tensor = tensor._values.tensor
        else:
            onnx_tensor = onnx.numpy_helper.from_array(tensor.values)
            if tensor.data_location is not None:
                onnx_tensor.data_location = tensor.data_location
        onnx_tensor.name = tensor.name
        return onnx_tensor

    @staticmethod
    def export_value_info_proto(tensor: Variable, do_type_check: bool) -> onnx.ValueInfoProto:
        if do_type_check and tensor.dtype is None:
            G_LOGGER.critical(
                "Graph input and output tensors must include dtype information. Please set the dtype attribute for: {:}".format(
                    tensor
                )
            )

        if tensor.dtype is not None:
            onnx_tensor = onnx.helper.make_tensor_value_info(tensor.name, dtype_to_onnx(tensor.dtype), tensor.shape)
        else:
            onnx_tensor = onnx.helper.make_empty_tensor_value_info(tensor.name)
        return onnx_tensor

    @staticmethod
    def export_node(node: Node, do_type_check: bool) -> onnx.NodeProto:
        # Cannot pass in attrs directly as make_node will change the order
        onnx_node = onnx.helper.make_node(
            node.op,
            inputs=[t.name for t in node.inputs],
            outputs=[t.name for t in node.outputs],
            name=node.name,
            domain=node.domain,
        )
        # Convert Tensors and Graphs to TensorProtos and GraphProtos respectively
        for key, val in node.attrs.items():
            if isinstance(val, Tensor):
                val = OnnxExporter.export_tensor_proto(val)
            elif isinstance(val, Graph):
                val = OnnxExporter.export_graph(val, do_type_check)
            elif isinstance(val, type):
                # May be a numpy type
                try:
                    val = dtype_to_onnx(val)
                except TypeError:
                    pass
            onnx_node.attribute.extend([onnx.helper.make_attribute(key, val)])
        return onnx_node

    @staticmethod
    def export_graph(graph: Graph, do_type_check=True) -> onnx.GraphProto:
        """
        Export an onnx-graphsurgeon Graph to an ONNX GraphProto.

        Args:
            graph (Graph): The graph to export.

            do_type_check (bool): Whether to check that input and output tensors have data types defined, and fail if not.
        """
        nodes = [OnnxExporter.export_node(node, do_type_check) for node in graph.nodes]
        inputs = [OnnxExporter.export_value_info_proto(inp, do_type_check) for inp in graph.inputs]
        outputs = [OnnxExporter.export_value_info_proto(out, do_type_check) for out in graph.outputs]
        tensor_map = graph.tensors()
        initializer = [
            OnnxExporter.export_tensor_proto(tensor) for tensor in tensor_map.values() if isinstance(tensor, Constant)
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
            doc_string=graph.doc_string,
            value_info=value_info,
        )


def export_onnx(graph: Graph, do_type_check=True, **kwargs) -> "onnx.ModelProto":
    """
    Exports an onnx-graphsurgeon Graph to an ONNX model.

    Args:
        graph (Graph): The graph to export

        do_type_check (bool): Whether to check that input and output tensors have data types defined, and fail if not.
        kwargs: Additional arguments to onnx.helper.make_model

    Returns:
        onnx.ModelProto: A corresponding ONNX model.
    """
    onnx_graph = OnnxExporter.export_graph(graph, do_type_check=do_type_check)

    if "opset_imports" not in kwargs:
        if graph.import_domains is None:
            kwargs["opset_imports"] = [onnx.helper.make_opsetid("", graph.opset)]
        else:
            kwargs["opset_imports"] = graph.import_domains

    model = onnx.helper.make_model(onnx_graph, **kwargs)
    model.producer_name = graph.producer_name
    model.producer_version = graph.producer_version
    return model
