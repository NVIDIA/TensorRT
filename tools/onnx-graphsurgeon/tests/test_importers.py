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

from collections import OrderedDict

import numpy as np
import onnx
import onnx.numpy_helper
import onnx.shape_inference
import pytest
from onnx_graphsurgeon.importers.onnx_importer import OnnxImporter
from onnx_graphsurgeon.ir.tensor import Constant, Tensor, Variable, SparseValues
from onnx_graphsurgeon.ir.function import Function
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.logger import G_LOGGER

from onnx_models import (
    dim_param_model,
    ext_weights,
    identity_model,
    initializer_is_output_model,
    lstm_model,
    nested_dup_names,
    scan_model,
    sparse_nnz_model,
    sparse_nnz_rank_model,
)

G_LOGGER.severity = G_LOGGER.ULTRA_VERBOSE


class TestOnnxImporter(object):
    @pytest.mark.parametrize(
        "onnx_type, expected_type",
        [
            (onnx.TensorProto.FLOAT, np.float32),
            (onnx.TensorProto.BFLOAT16, onnx.TensorProto.BFLOAT16),
            (onnx.TensorProto.FLOAT8E4M3FN, onnx.TensorProto.FLOAT8E4M3FN),
            (onnx.TensorProto.FLOAT8E4M3FNUZ, onnx.TensorProto.FLOAT8E4M3FNUZ),
            (onnx.TensorProto.FLOAT8E5M2, onnx.TensorProto.FLOAT8E5M2),
            (onnx.TensorProto.FLOAT8E5M2FNUZ, onnx.TensorProto.FLOAT8E5M2FNUZ),
        ],
    )
    def test_import_variable_tensor(self, onnx_type, expected_type):
        name = "test0"
        shape = (1, 2, 3, 4)
        onnx_tensor = onnx.helper.make_tensor_value_info(name, onnx_type, shape)
        tensor = OnnxImporter.import_tensor(onnx_tensor)
        assert type(tensor) == Variable
        assert tensor.name == name
        assert tensor.dtype == expected_type
        assert tuple(tensor.shape) == shape

    def test_import_constant_tensor(self):
        shape = (3, 3, 3)
        dtype = np.float32
        onnx_tensor = onnx.numpy_helper.from_array(np.ones(shape=shape, dtype=dtype))
        tensor = OnnxImporter.import_tensor(onnx_tensor)
        assert type(tensor) == Constant
        assert tensor.dtype == dtype
        assert tuple(tensor.shape) == shape

    def test_import_tensor_unknown_metadata(self):
        name = "test0"
        onnx_tensor = onnx.helper.make_empty_tensor_value_info(name)
        tensor = OnnxImporter.import_tensor(onnx_tensor)
        assert type(tensor) == Variable
        assert tensor.name == name

    # An empty string in `dim_param` should be treated like a dynamic dimension
    def test_import_empty_dim_param_tensor(self):
        shape = (1, 2, "non-empty", "")
        onnx_tensor = onnx.helper.make_tensor_value_info(
            "test0", onnx.TensorProto.FLOAT, shape
        )
        tensor = OnnxImporter.import_tensor(onnx_tensor)
        assert type(tensor) == Variable
        assert tuple(tensor.shape) == shape

    # Sometimes, tensor shape is not known, in which case we shouldn't import it
    def test_import_unknown_shape_tensor(self):
        shape = None
        onnx_tensor = onnx.helper.make_tensor_value_info(
            "test0", onnx.TensorProto.FLOAT, shape
        )
        tensor = OnnxImporter.import_tensor(onnx_tensor)
        assert type(tensor) == Variable
        assert tensor.shape is None

    # Scalars can be represented in ONNX with a dim that includes neither a dim_param nor dim_value
    def test_import_empty_dim_tensor(self):
        shape = (None,)
        onnx_tensor = onnx.helper.make_tensor_value_info(
            "test0", onnx.TensorProto.FLOAT, shape
        )
        onnx_tensor.type.tensor_type.shape.dim[0].ClearField("dim_value")
        onnx_tensor.type.tensor_type.shape.dim[0].ClearField("dim_param")

        tensor = OnnxImporter.import_tensor(onnx_tensor)
        assert type(tensor) == Variable
        assert tuple(tensor.shape) == shape

    # TODO: Test all attribute types - missing graph
    def test_import_node(self):
        op = "Test"
        inputs = ["x"]
        outputs = ["y"]
        float_attr = 4.0
        int_attr = 10
        str_attr = "constant"
        tensor_vals = np.ones(shape=(1, 2, 3, 4), dtype=np.float32)
        tensor_attr = onnx.numpy_helper.from_array(tensor_vals)
        floats_attr = [1.0, 2.0, 3.0, 4.0]
        ints_attr = [4, 3, 2, 1]
        strings_attr = ["constant", "and", "variable"]

        onnx_node = onnx.helper.make_node(
            op,
            inputs,
            outputs,
            float_attr=float_attr,
            int_attr=int_attr,
            str_attr=str_attr,
            tensor_attr=tensor_attr,
            floats_attr=floats_attr,
            ints_attr=ints_attr,
            strings_attr=strings_attr,
        )
        node = OnnxImporter.import_node(
            onnx_node, OrderedDict(), OrderedDict(), opset=11, import_domains=None
        )
        assert node.op == op
        assert node.attrs["float_attr"] == float_attr
        assert node.attrs["int_attr"] == int_attr
        assert node.attrs["str_attr"] == str_attr
        # Tensor should turn into a Constant
        assert np.all(node.attrs["tensor_attr"].values == tensor_vals)
        assert node.attrs["floats_attr"] == floats_attr
        assert node.attrs["ints_attr"] == ints_attr
        assert node.attrs["strings_attr"] == strings_attr

    def test_import_node_ref_attrs(self):
        op = "Test"
        inputs = ["x"]
        outputs = ["y"]
        attrs = {"attr1": 1, "attr2": 2.0}
        referencing_attr = "attr3"
        referenced_attr = "attr4"

        onnx_node = onnx.helper.make_node(op, inputs, outputs, **attrs)
        onnx_attr_ref = onnx.helper.make_attribute_ref(
            referencing_attr, onnx.AttributeProto.FLOAT
        )
        onnx_attr_ref.ref_attr_name = referenced_attr
        onnx_node.attribute.append(onnx_attr_ref)
        node = OnnxImporter.import_node(
            onnx_node, OrderedDict(), OrderedDict(), opset=11, import_domains=None
        )
        assert node.op == op
        assert node.attrs["attr1"] == 1
        assert node.attrs["attr2"] == 2.0
        assert node.attrs["attr3"] == Node.AttributeRef(referenced_attr, float)

    def test_import_function(self):
        name = "Test"
        domain = "com.test"
        inputs = ["X", "Y"]
        outputs = ["Z"]
        nodes = [
            onnx.helper.make_node("Add", ["X", "Y"], ["V"], attr1=1),
            onnx.helper.make_node("Add", ["X", "V"], ["W"], attr2=2),
            onnx.helper.make_node("Mul", ["V", "W"], ["Z"], attr3=3),
        ]
        opset = 18
        opset_imports = [onnx.helper.make_operatorsetid("ai.onnx", opset)]
        attributes = ["attr1", "attr2"]
        attribute_protos = [onnx.helper.make_attribute("attr3", 3)]
        doc_string = "docstring"
        onnx_function = onnx.helper.make_function(
            domain,
            name,
            inputs,
            outputs,
            nodes,
            opset_imports,
            attributes=attributes,
            attribute_protos=attribute_protos,
            doc_string=doc_string,
        )
        func = OnnxImporter.import_function(onnx_function)
        assert type(func) == Function
        assert func.name == name
        assert func.domain == domain
        assert func.doc_string == doc_string
        assert list(func.import_domains) == list(opset_imports)
        assert set(func.attrs.keys()) == set(attributes) | {
            a.name for a in attribute_protos
        }
        assert func.opset == opset
        assert all([isinstance(t, Tensor) for t in func.inputs + func.outputs])
        assert sorted(inputs) == sorted([t.name for t in func.inputs])
        assert sorted(outputs) == sorted([t.name for t in func.outputs])
        assert sorted([n.op_type for n in nodes]) == sorted([n.op for n in func.nodes])
        assert attribute_protos[0].i == func.attrs[attribute_protos[0].name]

    @pytest.mark.parametrize(
        "model",
        [
            identity_model(),
            lstm_model(),
            scan_model(),
            dim_param_model(),
            initializer_is_output_model(),
            nested_dup_names(),
            ext_weights(),
        ],
        ids=lambda model: str(model),
    )
    def test_import_graph(self, model):
        graph = OnnxImporter.import_graph(model.load().graph)
        model.assert_equal(graph)

    def test_import_graph_value_info(self):
        model = onnx.shape_inference.infer_shapes(identity_model().load())
        graph = OnnxImporter.import_graph(model.graph)
        tensors = graph.tensors()
        assert all(
            [
                type(tensor) == Variable and tensor.dtype is not None and tensor.shape
                for tensor in tensors.values()
            ]
        )

    def test_import_graph_tensor_map_preserved(self):
        model = identity_model()
        tensor_map = OrderedDict()
        graph = OnnxImporter.import_graph(model.load().graph, tensor_map=tensor_map)
        assert len(tensor_map) == 0
        model.assert_equal(graph)

    def test_import_graph_with_initializer(self):
        model = lstm_model()
        graph = OnnxImporter.import_graph(model.load().graph)
        model.assert_equal(graph)

    def test_import_graph_with_dim_param(self):
        model = dim_param_model()
        graph = OnnxImporter.import_graph(model.load().graph)
        model.assert_equal(graph)

    def test_import_graph_with_sparse_nnz_rank(self):
        model = sparse_nnz_rank_model()
        graph = OnnxImporter.import_graph(model.load().graph)
        tensors = graph.tensors()

        assert "w_sparse" in tensors
        sparse_tensor = tensors["w_sparse"]

        ref_value = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ).reshape(sparse_tensor._values.shape)
        assert (
            type(sparse_tensor) == Constant
            and type(sparse_tensor._values) == SparseValues
        )
        assert (tensors["w_sparse"]._values.load() == ref_value).all()

    def test_import_graph_with_sparse_nnz(self):
        model = sparse_nnz_model()
        graph = OnnxImporter.import_graph(model.load().graph)
        tensors = graph.tensors()

        assert "w_sparse" in tensors
        sparse_tensor = tensors["w_sparse"]

        ref_value = np.array(
            [
                0.0,
                1.0,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
                3.0,
                0.0,
                0.0,
                0.0,
                0.0,
                4.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                5.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ).reshape(sparse_tensor._values.shape)
        assert (
            type(sparse_tensor) == Constant
            and type(sparse_tensor._values) == SparseValues
        )
        assert (tensors["w_sparse"]._values.load() == ref_value).all()
