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
from collections import OrderedDict
from typing import List, Union, Dict, Any

import numpy as np
import onnx
import onnx.numpy_helper
from onnx_graphsurgeon.importers.base_importer import BaseImporter
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
from onnx_graphsurgeon.logger import G_LOGGER, LogMode
from onnx_graphsurgeon.util import misc

# Maps values from the AttributeType enum to their string representations, e.g., {1: "FLOAT"}
ATTR_TYPE_MAPPING = {v: k for k, v in onnx.AttributeProto.AttributeType.items()}

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


def get_onnx_tensor_shape(
    onnx_tensor: Union[onnx.ValueInfoProto, onnx.TensorProto]
) -> List[int]:
    shape = None
    if isinstance(onnx_tensor, onnx.TensorProto) or isinstance(
        onnx_tensor, onnx.SparseTensorProto
    ):
        shape = onnx_tensor.dims
    else:
        if onnx_tensor.type.tensor_type.HasField("shape"):
            shape = []
            for dim in onnx_tensor.type.tensor_type.shape.dim:
                if dim.HasField("dim_param"):
                    shape.append(dim.dim_param)
                elif dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                else:
                    shape.append(None)
    return shape


def get_dtype_name(onnx_type):
    return {val: key for key, val in onnx.TensorProto.DataType.items()}[onnx_type]


def get_itemsize(dtype):
    np_dtype = get_numpy_type(dtype)
    if np_dtype is not None:
        return np.dtype(np_dtype).itemsize

    if dtype == onnx.TensorProto.BFLOAT16:
        return 2

    if dtype in [
        onnx.TensorProto.FLOAT8E4M3FN,
        onnx.TensorProto.FLOAT8E4M3FNUZ,
        onnx.TensorProto.FLOAT8E5M2,
        onnx.TensorProto.FLOAT8E5M2FNUZ,
    ]:
        return 1
    G_LOGGER.critical(f"Unsupported type: {dtype}")


def get_numpy_type(onnx_type):
    if not isinstance(onnx_type, int):
        # Already a NumPy type
        return onnx_type

    numpy_unsupported_types = [
        onnx.TensorProto.BFLOAT16,
        onnx.TensorProto.FLOAT8E4M3FN,
        onnx.TensorProto.FLOAT8E4M3FNUZ,
        onnx.TensorProto.FLOAT8E5M2,
        onnx.TensorProto.FLOAT8E5M2FNUZ,
    ]

    # TENSOR_TYPE_TO_NP_TYPE maps types unsupported by NumPy to random other types.
    # This obviously breaks things, so we need to treat this as a special case.
    if (
        onnx_type not in numpy_unsupported_types
        and onnx_type in onnx.helper.get_all_tensor_dtypes()
    ):
        return onnx.helper.tensor_dtype_to_np_dtype(onnx_type)
    return None


def get_onnx_tensor_dtype(
    onnx_tensor: Union[onnx.ValueInfoProto, onnx.TensorProto]
) -> Union[np.dtype, "onnx.TensorProto.DataType"]:
    if isinstance(onnx_tensor, onnx.TensorProto):
        onnx_dtype = onnx_tensor.data_type
    elif isinstance(onnx_tensor, onnx.SparseTensorProto):
        onnx_dtype = onnx_tensor.values.data_type
    else:
        if onnx_tensor.type.HasField("tensor_type"):
            onnx_dtype = onnx_tensor.type.tensor_type.elem_type
        elif onnx_tensor.type.HasField("sequence_type"):
            onnx_dtype = onnx_tensor.type.sequence_type.elem_type.tensor_type.elem_type
        elif onnx_tensor.type.HasField("map_type"):
            onnx_dtype = onnx_tensor.type.map_type.value_type
        elif onnx_tensor.type.HasField("optional_type"):
            onnx_dtype = onnx_tensor.type.optional_type.elem_type
        elif onnx_tensor.type.HasField("sparse_tensor_type"):
            onnx_dtype = onnx_tensor.type.sparse_tensor_type.elem_type
        else:
            onnx_dtype = onnx_tensor.type.opaque_type

    dtype = get_numpy_type(onnx_dtype)
    if dtype is not None:
        return dtype

    G_LOGGER.warning(
        f"Could not convert: {get_dtype_name(onnx_dtype)} to a corresponding NumPy type. "
        f"The original ONNX type will be preserved. ",
        mode=LogMode.ONCE,
    )
    return onnx_dtype


def get_onnx_tensor_type(
    onnx_tensor: Union[onnx.ValueInfoProto, onnx.TensorProto]
) -> str:
    if isinstance(onnx_tensor, onnx.TensorProto):
        onnx_type = "tensor_type"
    else:
        if onnx_tensor.type.HasField("tensor_type"):
            onnx_type = "tensor_type"
        elif onnx_tensor.type.HasField("sequence_type"):
            onnx_type = "sequence_type"
        elif onnx_tensor.type.HasField("map_type"):
            onnx_type = "map_type"
        elif onnx_tensor.type.HasField("optional_type"):
            onnx_type = "optional_type"
        elif onnx_tensor.type.HasField("opaque_type"):
            onnx_type = "opaque_type"
        elif onnx_tensor.type.HasField("sparse_tensor_type"):
            onnx_type = "sparse_tensor_type"
        else:
            onnx_type = None

    return onnx_type


def get_onnx_tensor_type(
    onnx_tensor: Union[onnx.ValueInfoProto, onnx.TensorProto]
) -> str:
    if isinstance(onnx_tensor, onnx.TensorProto):
        onnx_type = "tensor_type"
    else:
        if onnx_tensor.type.HasField("tensor_type"):
            onnx_type = "tensor_type"
        elif onnx_tensor.type.HasField("sequence_type"):
            onnx_type = "sequence_type"
        elif onnx_tensor.type.HasField("map_type"):
            onnx_type = "map_type"
        elif onnx_tensor.type.HasField("optional_type"):
            onnx_type = "optional_type"
        elif onnx_tensor.type.HasField("opaque_type"):
            onnx_type = "opaque_type"
        elif onnx_tensor.type.HasField("sparse_tensor_type"):
            onnx_type = "sparse_tensor_type"
        else:
            onnx_type = None

    return onnx_type


class OnnxImporter(BaseImporter):
    @staticmethod
    def get_opset(model_or_func: Union[onnx.ModelProto, onnx.FunctionProto]):
        class_name = (
            "Function" if isinstance(model_or_func, onnx.FunctionProto) else "Model"
        )
        try:
            for importer in OnnxImporter.get_import_domains(model_or_func):
                if importer.domain == "" or importer.domain == "ai.onnx":
                    return importer.version
            G_LOGGER.warning(
                f"{class_name} does not contain ONNX domain opset information! Using default opset."
            )
            return None
        except:
            G_LOGGER.warning(
                f"{class_name} does not contain opset information! Using default opset."
            )
            return None

    @staticmethod
    def get_import_domains(model_or_func: Union[onnx.ModelProto, onnx.FunctionProto]):
        return model_or_func.opset_import

    @staticmethod
    def import_tensor(
        onnx_tensor: Union[
            onnx.ValueInfoProto, onnx.TensorProto, onnx.SparseTensorProto
        ]
    ) -> Tensor:
        if isinstance(onnx_tensor, onnx.SparseTensorProto):
            return Constant(
                name=onnx_tensor.values.name,
                values=SparseValues(onnx_tensor),
                data_location=onnx_tensor.values.data_location,
            )
        elif isinstance(onnx_tensor, onnx.TensorProto):
            data_location = (
                int(onnx_tensor.data_location)
                if onnx_tensor.HasField("data_location")
                else None
            )
            return Constant(
                name=onnx_tensor.name,
                values=LazyValues(onnx_tensor),
                data_location=data_location,
            )
        else:
            # A ValueInfoProto inside a subgraph might not have shape & type specified.
            tensor = Variable(onnx_tensor.name)
            if onnx_tensor.type.ByteSize() > 0:
                tensor.dtype = get_onnx_tensor_dtype(onnx_tensor)
                tensor.shape = get_onnx_tensor_shape(onnx_tensor)
                tensor.type = get_onnx_tensor_type(onnx_tensor)
            return tensor

    @staticmethod
    def import_attributes(
        onnx_attributes: List[onnx.AttributeProto],
        tensor_map: "OrderedDict[str, Tensor]",
        subgraph_tensor_map: "OrderedDict[str, Tensor]",
        opset: int,
        import_domains: onnx.OperatorSetIdProto,
    ) -> "OrderedDict[str, Any]":
        attr_dict = OrderedDict()
        for attr in onnx_attributes:

            def process_attr(attr_str: str):
                if attr.ref_attr_name:
                    attr_type = misc.convert_from_onnx_attr_type(attr.type)
                    return Node.AttributeRef(attr.ref_attr_name, attr_type)
                processed = getattr(attr, ONNX_PYTHON_ATTR_MAPPING[attr_str])
                if attr_str == "STRING":
                    processed = processed.decode()
                elif attr_str == "TENSOR":
                    processed = OnnxImporter.import_tensor(processed)
                elif attr_str == "GRAPH":
                    processed = OnnxImporter.import_graph(
                        processed,
                        misc.combine_dicts(tensor_map, subgraph_tensor_map),
                        opset=opset,
                        import_domains=import_domains,
                    )
                elif attr_str == "FLOATS" or attr_str == "INTS":
                    processed = list(processed)
                elif attr_str == "STRINGS":
                    processed = [p.decode() for p in processed]
                return processed

            if attr.type in ATTR_TYPE_MAPPING:
                attr_str = ATTR_TYPE_MAPPING[attr.type]
                if attr_str in ONNX_PYTHON_ATTR_MAPPING:
                    attr_dict[attr.name] = process_attr(attr_str)
                else:
                    G_LOGGER.warning(
                        "Attribute of type {:} is currently unsupported. Skipping attribute.".format(
                            attr_str
                        )
                    )
            else:
                G_LOGGER.warning(
                    "Attribute type: {:} was not recognized. Was the graph generated with a newer IR version than the installed `onnx` package? Skipping attribute.".format(
                        attr.type
                    )
                )
        return attr_dict

    @staticmethod
    def import_node(
        onnx_node: onnx.NodeProto,
        tensor_map: "OrderedDict[str, Tensor]",
        subgraph_tensor_map: "OrderedDict[str, Tensor]",
        opset,
        import_domains: onnx.OperatorSetIdProto,
    ) -> Node:
        # Optional inputs/outputs are represented by empty tensors. All other tensors should already have been populated during shape inference.
        def get_tensor(name: str, check_outer_graph=True):
            # Prioritize the subgraph even if check_outer_graph is set
            if name in subgraph_tensor_map:
                return subgraph_tensor_map[name]

            if check_outer_graph and name in tensor_map:
                return tensor_map[name]

            if not name:
                # Empty tensors are not tracked by the graph, as these represent optional inputs/outputs that have been omitted.
                G_LOGGER.verbose("Generating empty tensor")
                return Variable.empty()

            G_LOGGER.verbose(
                "Tensor: {:} was not generated during shape inference, or shape inference was not run on this model. Creating a new Tensor.".format(
                    name
                )
            )
            subgraph_tensor_map[name] = Variable(name)
            return subgraph_tensor_map[name]

        # Retrieve Tensors for node inputs/outputs. Only empty tensors should need to be newly added.
        def retrieve_node_inputs() -> List[Tensor]:
            inputs = []  # List[Tensor]
            for input_name in onnx_node.input:
                inputs.append(get_tensor(input_name))
            return inputs

        def retrieve_node_outputs() -> List[Tensor]:
            outputs = []  # List[Tensor]
            for output_name in onnx_node.output:
                # Node outputs cannot come from the outer graph, they must be created within the inner graph.
                outputs.append(get_tensor(output_name, check_outer_graph=False))
            return outputs

        attributes = OnnxImporter.import_attributes(
            onnx_node.attribute, tensor_map, subgraph_tensor_map, opset, import_domains
        )

        return Node(
            op=onnx_node.op_type,
            name=onnx_node.name,
            attrs=attributes,
            inputs=retrieve_node_inputs(),
            outputs=retrieve_node_outputs(),
            domain=onnx_node.domain if onnx_node.HasField("domain") else None,
        )

    @staticmethod
    def import_function(
        onnx_function: onnx.FunctionProto,
        model_opset: int = None,
        model_import_domains: onnx.OperatorSetIdProto = None,
    ) -> Function:
        opset = OnnxImporter.get_opset(onnx_function) or model_opset
        import_domains = (
            OnnxImporter.get_import_domains(onnx_function) or model_import_domains
        )
        subgraph_tensor_map = OrderedDict()  # Tensors in this function

        def make_tensor(name: str) -> Tensor:
            if name not in subgraph_tensor_map:
                subgraph_tensor_map[name] = Variable(name)
            return subgraph_tensor_map[name]

        function_inputs = [make_tensor(inp) for inp in onnx_function.input]
        function_outputs = [make_tensor(out) for out in onnx_function.output]
        nodes = [
            OnnxImporter.import_node(
                onnx_node, dict(), subgraph_tensor_map, opset, import_domains
            )
            for onnx_node in onnx_function.node
        ]

        attributes = dict()
        if onnx_function.attribute:
            attributes = {attr_name: None for attr_name in onnx_function.attribute}
        if onnx_function.attribute_proto:
            attrs_with_default_value = OnnxImporter.import_attributes(
                onnx_function.attribute_proto,
                None,
                subgraph_tensor_map,
                opset,
                import_domains,
            )
            attributes.update(attrs_with_default_value)

        return Function(
            onnx_function.name,
            onnx_function.domain,
            nodes=nodes,
            inputs=function_inputs,
            outputs=function_outputs,
            doc_string=onnx_function.doc_string,
            opset=opset,
            import_domains=import_domains,
            attrs=attributes,
        )

    @staticmethod
    def import_graph(
        onnx_graph: onnx.GraphProto,
        tensor_map: "OrderedDict[str, Tensor]" = None,
        opset=None,
        import_domains: onnx.OperatorSetIdProto = None,
        producer_name: str = None,
        producer_version: str = None,
        functions: List[Function] = None,
    ) -> Graph:
        """
        Imports a Graph from an ONNX Graph.

        Args:
            onnx_graph (onnx.GraphProto): The ONNX graph to import.

            tensor_map (OrderedDict[str, Tensor]): A mapping of tensor names to Tensors. This is generally only useful for subgraph import.
            opset (int): The ONNX opset to use for this graph.
            producer_name (str): The name of the tool used to generate the model. Defaults to "".
            producer_version (str): The version of the generating tool. Defaults to "".
            functions (List[Function]): The list of custom functions which are available to use in the model.
        """
        functions = misc.default_value(functions, [])
        tensor_map = copy.copy(
            misc.default_value(tensor_map, OrderedDict())
        )  # Outer graph tensors, read-only
        subgraph_tensor_map = OrderedDict()  # Tensors in this subgraph

        # Retrieves a Tensor from subgraph_tensor_map or the outer graph (tensor_map) if present, otherwise imports the tensor
        # If overwrite=True, this function will overwrite previously imported tensors
        # if the new tensor has more information available.
        def get_tensor(
            onnx_tensor: Union[
                onnx.ValueInfoProto, onnx.TensorProto, onnx.SparseTensorProto
            ],
            overwrite=False,
            check_outer_graph=True,
        ) -> Tensor:
            if isinstance(onnx_tensor, onnx.SparseTensorProto):
                name = onnx_tensor.values.name
            else:
                name = onnx_tensor.name
            # Prioritize the subgraph even if check_outer_graph is set
            if name in subgraph_tensor_map:
                if overwrite:
                    tensor = OnnxImporter.import_tensor(onnx_tensor)
                    if isinstance(subgraph_tensor_map[name], Variable):
                        subgraph_tensor_map[name].dtype = (
                            subgraph_tensor_map[name].dtype or tensor.dtype
                        )
                        subgraph_tensor_map[name].shape = (
                            subgraph_tensor_map[name].shape or tensor.shape
                        )
                return subgraph_tensor_map[name]

            if check_outer_graph and name in tensor_map:
                return tensor_map[name]

            subgraph_tensor_map[name] = OnnxImporter.import_tensor(onnx_tensor)
            return subgraph_tensor_map[name]

        # Import initializers contents into Constants.
        G_LOGGER.verbose("Importing initializers")
        for initializer in onnx_graph.initializer:
            get_tensor(initializer)
        for initializer in onnx_graph.sparse_initializer:
            get_tensor(initializer)

        # Import all tensors whose shapes are known. Tensors may be repeated, and some of these
        # duplicates may not include shape/dtype information, so overwrite is set to True
        # so that we can capture all the information available about the tensor
        G_LOGGER.verbose("Importing tensors with known shapes")
        for tensor in onnx_graph.value_info:
            get_tensor(tensor, overwrite=True)

        # Import graph inputs and outputs. Initializers are not considered to be inputs.
        # Graph inputs and outputs can never come from the outer graph!
        initializer_names = set(
            [tensor.name for tensor in onnx_graph.initializer]
            + [tensor.values.name for tensor in onnx_graph.sparse_initializer]
        )
        G_LOGGER.verbose("Importing graph inputs")
        graph_inputs = []  # List[Tensor]
        for inp in onnx_graph.input:
            if inp.name not in initializer_names:
                tensor = get_tensor(inp, check_outer_graph=False)
                graph_inputs.append(tensor)

        G_LOGGER.verbose("Importing graph outputs")
        graph_outputs = []  # List[Tensor]
        for out in onnx_graph.output:
            tensor = get_tensor(out, check_outer_graph=False)
            graph_outputs.append(tensor)

        G_LOGGER.verbose("Importing nodes")
        nodes = []  # List[Node]
        for onnx_node in onnx_graph.node:
            node = OnnxImporter.import_node(
                onnx_node, tensor_map, subgraph_tensor_map, opset, import_domains
            )
            nodes.append(node)

        return Graph(
            nodes=nodes,
            inputs=graph_inputs,
            outputs=graph_outputs,
            name=onnx_graph.name,
            doc_string=onnx_graph.doc_string,
            producer_name=producer_name,
            producer_version=producer_version,
            opset=opset,
            import_domains=import_domains,
            functions=functions,
        )


def import_onnx(onnx_model: "onnx.ModelProto") -> Graph:
    """
    Import an onnx-graphsurgeon Graph from the provided ONNX model.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.

    Returns:
        Graph: A corresponding onnx-graphsurgeon Graph.
    """
    model_opset = OnnxImporter.get_opset(onnx_model)
    model_import_domains = OnnxImporter.get_import_domains(onnx_model)
    functions: List[Function] = [
        OnnxImporter.import_function(
            onnx_function,
            model_opset=model_opset,
            model_import_domains=model_import_domains,
        )
        for onnx_function in onnx_model.functions
    ]

    # Functions are identified by their name and domain.
    # Make sure that no two Functions share the same name and domain.
    function_unqiue_ids = set()
    for func in functions:
        unique_id = func.unique_id
        if unique_id in function_unqiue_ids:
            msg = "Model contains duplicate function definitions with "
            msg += f'name="{func.name}" and domain="{func.domain}"'
            G_LOGGER.warning(msg)

    return OnnxImporter.import_graph(
        onnx_model.graph,
        opset=model_opset,
        import_domains=model_import_domains,
        producer_name=onnx_model.producer_name,
        producer_version=onnx_model.producer_version,
        functions=functions,
    )
