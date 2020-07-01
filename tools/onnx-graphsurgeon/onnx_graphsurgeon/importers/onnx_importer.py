#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from onnx_graphsurgeon.ir.tensor import Tensor, Constant, Variable
from onnx_graphsurgeon.importers.base_importer import BaseImporter
from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.util import misc

from typing import List, Union
from collections import OrderedDict
import onnx.numpy_helper
import numpy as np
import onnx
import copy

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

def get_onnx_tensor_shape(onnx_tensor: Union[onnx.ValueInfoProto, onnx.TensorProto]) -> List[int]:
    shape = []
    if isinstance(onnx_tensor, onnx.TensorProto):
        shape = onnx_tensor.dims
    else:
        for dim in onnx_tensor.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
    return shape


def get_onnx_tensor_dtype(onnx_tensor: Union[onnx.ValueInfoProto, onnx.TensorProto]) -> np.dtype:
    if isinstance(onnx_tensor, onnx.TensorProto):
        onnx_type = onnx_tensor.data_type
    else:
        onnx_type = onnx_tensor.type.tensor_type.elem_type
    if onnx_type in onnx.mapping.TENSOR_TYPE_TO_NP_TYPE:
        return onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_type]
    return None


class OnnxImporter(BaseImporter):
    @staticmethod
    def get_opset(model: onnx.ModelProto):
        try:
            return model.opset_import[0].version
        except:
            G_LOGGER.warning("Model does not contain opset information! Using default opset.")
            return None


    @staticmethod
    def import_tensor(onnx_tensor: Union[onnx.ValueInfoProto, onnx.TensorProto]) -> Tensor:
        try:
            values = onnx.numpy_helper.to_array(onnx_tensor)
            return Constant(name=onnx_tensor.name, values=values)
        except ValueError:
            return Variable(name=onnx_tensor.name, dtype=get_onnx_tensor_dtype(onnx_tensor), shape=get_onnx_tensor_shape(onnx_tensor))


    @staticmethod
    def import_node(onnx_node: onnx.NodeProto, tensor_map: "OrderedDict[str, Tensor]") -> Node:
        def attrs_to_dict(attrs):
            attr_dict = OrderedDict()
            for attr in attrs:
                def process_attr(attr_str: str):
                    processed = getattr(attr, ONNX_PYTHON_ATTR_MAPPING[attr_str])
                    if attr_str == "STRING":
                        processed = processed.decode()
                    elif attr_str == "TENSOR":
                        processed = OnnxImporter.import_tensor(processed)
                    elif attr_str == "GRAPH":
                        processed = OnnxImporter.import_graph(processed, tensor_map)
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
                        G_LOGGER.warning("Attribute of type {:} is currently unsupported. Skipping attribute.".format(attr_str))
                else:
                    G_LOGGER.warning("Attribute type: {:} was not recognized. Was the graph generated with a newer IR version than the installed `onnx` package? Skipping attribute.".format(attr.type))
            return attr_dict

        # Optional inputs/outputs are represented by empty tensors. All other tensors should already have been populated during shape inference.
        def check_tensor(name: str):
            if name not in tensor_map:
                if name:
                    G_LOGGER.debug("Tensor: {:} was not generated during shape inference, or shape inference was not run on this model. Creating a new Tensor.".format(name))
                    tensor_map[name] = Variable(name)
                else:
                    # Empty tensors are not tracked by the graph, as these represent optional inputs/outputs that have been omitted.
                    G_LOGGER.verbose("Generating empty tensor")
                    return Variable.empty()
            return tensor_map[name]

        # Retrieve Tensors for node inputs/outputs. Only empty tensors should need to be newly added.
        def retrieve_node_inputs() -> List[Tensor]:
            inputs = [] # List[Tensor]
            for input_name in onnx_node.input:
                inputs.append(check_tensor(input_name))
            return inputs

        def retrieve_node_outputs() -> List[Tensor]:
            outputs = [] # List[Tensor]
            for output_name in onnx_node.output:
                outputs.append(check_tensor(output_name))
            return outputs

        return Node(op=onnx_node.op_type, name=onnx_node.name, attrs=attrs_to_dict(onnx_node.attribute), inputs=retrieve_node_inputs(), outputs=retrieve_node_outputs())


    @staticmethod
    def import_graph(onnx_graph: onnx.GraphProto, tensor_map: "OrderedDict[str, Tensor]"=None, opset=None) -> Graph:
        """
        Imports a Graph from an ONNX Graph.

        Args:
            onnx_graph (onnx.GraphProto): The ONNX graph to import.

        Optional Args:
            tensor_map (OrderedDict[str, Tensor]): A mapping of tensor names to Tensors. This is generally only useful for subgraph import.
            opset (int): The ONNX opset to use for this graph.
        """
        # Tensor map should not be modified - may be from outer graph
        tensor_map = copy.copy(misc.default_value(tensor_map, OrderedDict()))

        # Retrieves a Tensor from tensor_map if present, otherwise imports the tensor
        # If overwrite=True, this function will overwrite previously imported tensors
        # if the new tensor has more information available.
        def get_tensor(onnx_tensor: Union[onnx.ValueInfoProto, onnx.TensorProto], overwrite=False) -> Tensor:
            if onnx_tensor.name not in tensor_map:
                tensor_map[onnx_tensor.name] = OnnxImporter.import_tensor(onnx_tensor)
            elif overwrite:
                tensor = OnnxImporter.import_tensor(onnx_tensor)
                if isinstance(tensor_map[onnx_tensor.name], Variable):
                    tensor_map[onnx_tensor.name].dtype = tensor_map[onnx_tensor.name].dtype or tensor.dtype
                    tensor_map[onnx_tensor.name].shape = tensor_map[onnx_tensor.name].shape or tensor.shape
            return tensor_map[onnx_tensor.name]

        # Import initializers contents into Constants.
        G_LOGGER.debug("Importing initializers")
        for initializer in onnx_graph.initializer:
            get_tensor(initializer)

        # Import all tensors whose shapes are known. Tensors may be repeated, and some of these
        # duplicates may not include shape/dtype information, so overwrite is set to True
        # so that we can capture all the information available about the tensor
        G_LOGGER.debug("Importing tensors with known shapes")
        for tensor in onnx_graph.value_info:
            get_tensor(tensor, overwrite=True)

        # Import graph inputs and outputs. Initializers are not considered to be inputs.
        initializer_names = set([tensor.name for tensor in onnx_graph.initializer])
        G_LOGGER.debug("Importing graph inputs")
        graph_inputs = [] # List[Tensor]
        for inp in onnx_graph.input:
            if inp.name not in initializer_names:
                tensor = get_tensor(inp)
                graph_inputs.append(tensor)

        G_LOGGER.debug("Importing graph outputs")
        graph_outputs = [] # List[Tensor]
        for out in onnx_graph.output:
            tensor = get_tensor(out)
            graph_outputs.append(tensor)

        G_LOGGER.debug("Importing nodes")
        nodes = [] # List[Node]
        for onnx_node in onnx_graph.node:
            node = OnnxImporter.import_node(onnx_node, tensor_map)
            nodes.append(node)

        return Graph(nodes=nodes, inputs=graph_inputs, outputs=graph_outputs, name=onnx_graph.name, doc_string=onnx_graph.doc_string, opset=opset)
