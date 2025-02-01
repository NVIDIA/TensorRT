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
from __future__ import annotations

from collections import OrderedDict
from typing import Sequence
from unittest import mock

import ml_dtypes
import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from numpy.typing import DTypeLike, NDArray
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

from onnx_graphsurgeon.exporters.onnx_exporter import (
    OnnxExporter,
    constant_to_onnx_tensor,
)
from onnx_graphsurgeon.importers.onnx_importer import OnnxImporter
from onnx_graphsurgeon.ir.function import Function
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.ir.tensor import Constant, LazyValues, Tensor, Variable


class TestOnnxExporter(object):

    def _bytes_to_np_array(self, b: bytes, shape: Sequence[int], dtype: DTypeLike) -> NDArray:
        """Construct a np.array from raw bytes.

        For ML-specific types like float8_e3m4, `onnx.numpy_helper.to_array` does not convert to arrays with correct
        numerical representations, but only puts the correct bytes into 8-bit containers like uint8. This makes
        numerical comparisons impossible and hence not useful for testing. Instead, this function always converts to
        arrays with correct numerical representations as long as the `dtype` argument is given `ml_dtypes.xxx`.
        """
        return np.frombuffer(b, dtype=dtype).reshape(shape)

    def test_should_export_not_load_lazy_values(self):
        """Test that when exporting a gs.Constant with LazyValues, the exporter does not load the lazy values."""
        # Precondition.
        name = "constant_tensor"
        shape = (3, 3, 3)
        dtype = np.float32
        onnx_tensor = onnx.numpy_helper.from_array(np.ones(shape=shape, dtype=dtype))
        tensor = Constant(name=name, values=LazyValues(onnx_tensor))

        # Under test.
        onnx_tensor = OnnxExporter.export_tensor_proto(tensor)

        # Postcondition.
        assert isinstance(tensor._values, LazyValues)  # Exporter should *not* load LazyValues into a numpy array.

    def test_export_constant_tensor_to_tensor_proto(self):
        name = "constant_tensor"
        shape = (3, 224, 224)
        values = np.random.random_sample(size=shape).astype(np.float32)

        tensor = Constant(name=name, values=values)
        onnx_tensor = OnnxExporter.export_tensor_proto(tensor)
        assert onnx_tensor.name == name
        assert np.all(self._bytes_to_np_array(onnx_tensor.raw_data, onnx_tensor.dims, np.float32) == values)
        assert onnx_tensor.data_type == onnx.TensorProto.FLOAT
        assert tuple(onnx_tensor.dims) == shape

    def test_should_export_constant_tensor_with_different_target_dtype(self) -> None:
        """Test that `export_tensor_proto` exports a TensorProto with correct data when the target dtype is different
        from source dtype.
        """
        # Precondition.
        name = "constant_tensor"
        shape = (3, 224, 224)
        values = np.random.random_sample(size=shape).astype(np.float32)

        tensor = Constant(name=name, values=values, export_dtype=onnx.TensorProto.FLOAT8E4M3FN)

        # Under test.
        onnx_tensor = OnnxExporter.export_tensor_proto(tensor)

        # Postcondition.
        assert onnx_tensor.name == name
        assert np.all(
            np.isclose(
                self._bytes_to_np_array(onnx_tensor.raw_data, onnx_tensor.dims, ml_dtypes.float8_e4m3fn),
                values,
                atol=0.1,
            )
        )
        assert onnx_tensor.data_type == onnx.TensorProto.FLOAT8E4M3FN
        assert tuple(onnx_tensor.dims) == shape

    def test_should_export_constant_tensor_with_ml_dtype(self) -> None:
        """Test that `export_tensor_proto` exports a TensorProto with correct data when the Constant has values in
        ml_dtypes.
        """
        # Precondition.
        name = "constant_tensor"
        shape = (3, 224, 224)
        values = np.random.random_sample(size=shape).astype(ml_dtypes.float8_e4m3fn)

        tensor = Constant(name=name, values=values)

        # Under test.
        onnx_tensor = OnnxExporter.export_tensor_proto(tensor)

        # Postcondition.
        assert onnx_tensor.name == name
        assert np.all(
            self._bytes_to_np_array(onnx_tensor.raw_data, onnx_tensor.dims, ml_dtypes.float8_e4m3fn) == tensor.values
        )
        assert onnx_tensor.data_type == onnx.TensorProto.FLOAT8E4M3FN
        assert tuple(onnx_tensor.dims) == shape

    def test_should_export_constant_tensor_with_ml_dtype_raise_error_when_onnx_dtype_not_supported(self) -> None:
        """Test that `export_tensor_proto` raises an error when the corresponding ONNX data type is not supported and
        prompts the user to upgrade the ONNX package.
        """
        # Precondition.
        name = "constant_tensor"
        shape = (3, 224, 224)
        values = np.random.random_sample(size=shape).astype(ml_dtypes.int4)
        tensor = Constant(name=name, values=values)

        # Mock an outdated ONNX package that does not support INT4.
        with mock.patch("onnx.TensorProto") as mock_TensorProto:
            del mock_TensorProto.INT4

            # Under test and postcondition.
            with pytest.raises(RuntimeError) as e:
                OnnxExporter.export_tensor_proto(tensor)
            assert (
                str(e.value) == "Current ONNX package does not support INT4. Please upgrade ONNX to the latest version."
            )

    def test_should_export_constant_tensor_pass_when_ml_dtypes_not_installed(self) -> None:
        """Test that `export_tensor_proto` passes when the ml_dtypes package was not installed."""
        # Precondition.
        tensor = Constant(name="constant_tensor", values=np.random.random_sample(size=(3, 224, 224)))

        with mock.patch.dict("sys.modules", {"ml_dtypes": None}):
            # Make sure that the ml_dtypes package is not installed.
            with pytest.raises(ImportError):
                import ml_dtypes  # noqa: F401

            # Under test and postcondition.
            OnnxExporter.export_tensor_proto(tensor)

    def test_should_export_constant_tensor_with_lazy_values(self) -> None:
        """Test that `export_tensor_proto` correctly exports a Constant object with LazyValues."""
        # Precondition.
        constant_name = "constant_name"
        lazy_value_name = "lazy_value_name"
        shape = (2,)
        values = np.random.random_sample(size=shape).astype(np.float32)

        input_onnx_tensor = onnx.TensorProto(
            dims=shape, data_type=onnx.TensorProto.FLOAT, raw_data=values.tobytes(), name=lazy_value_name
        )
        tensor = Constant(name=constant_name, values=LazyValues(tensor=input_onnx_tensor))

        # Under test.
        exported_onnx_tensor = OnnxExporter.export_tensor_proto(tensor)

        # Postcondition.
        # Importantly, the exported TensorProto's name should be `constant_name`, NOT `lazy_value_name`.
        assert exported_onnx_tensor.name == constant_name
        assert np.all(self._bytes_to_np_array(exported_onnx_tensor.raw_data, exported_onnx_tensor.dims, np.float32))
        assert exported_onnx_tensor.data_type == onnx.TensorProto.FLOAT
        assert tuple(exported_onnx_tensor.dims) == shape

    def test_export_constant_tensor_to_value_info_proto(self):
        name = "constant_tensor"
        shape = (3, 224, 224)
        values = np.random.random_sample(size=shape).astype(np.float32)

        tensor = Constant(name=name, values=values)
        onnx_tensor = OnnxExporter.export_value_info_proto(tensor, do_type_check=True)
        assert onnx_tensor.name == name
        assert onnx_tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT

        onnx_shape = []
        for dim in onnx_tensor.type.tensor_type.shape.dim:
            onnx_shape.append(dim.dim_value)
        assert tuple(onnx_shape) == shape

    @pytest.mark.parametrize(
        "export_dtype, container_dtype, threshold, onnx_to_numpy_converter",
        [
            (
                onnx.TensorProto.BFLOAT16,
                np.uint16,
                0.02,
                onnx.numpy_helper.bfloat16_to_float32,
            ),
            (
                onnx.TensorProto.FLOAT8E4M3FN,
                np.uint8,
                0.35,
                lambda x, dims: onnx.numpy_helper.float8e4m3_to_float32(x, dims, fn=True, uz=False),
            ),
        ],
    )
    def test_export_numpy_unsupported_dtypes_accuracy(
        self, export_dtype, container_dtype, threshold, onnx_to_numpy_converter
    ):
        name = "constant_tensor"
        shape = (3, 224, 224)
        values = np.random.random_sample(size=shape).astype(np.float32)

        tensor = Constant(name=name, values=values, export_dtype=export_dtype)
        onnx_tensor = constant_to_onnx_tensor(tensor)
        np_arr = np.frombuffer(onnx_tensor.raw_data, dtype=container_dtype)
        np_arr_fp32 = onnx_to_numpy_converter(np_arr, dims=values.shape)

        assert np.max(np.abs(np_arr_fp32 - values)) <= threshold

    @pytest.mark.parametrize(
        "dtype, expected_type",
        [
            (np.float32, onnx.TensorProto.FLOAT),
            (onnx.TensorProto.BFLOAT16, onnx.TensorProto.BFLOAT16),
            (onnx.TensorProto.FLOAT8E4M3FN, onnx.TensorProto.FLOAT8E4M3FN),
            (onnx.TensorProto.FLOAT8E4M3FNUZ, onnx.TensorProto.FLOAT8E4M3FNUZ),
            (onnx.TensorProto.FLOAT8E5M2, onnx.TensorProto.FLOAT8E5M2),
            (onnx.TensorProto.FLOAT8E5M2FNUZ, onnx.TensorProto.FLOAT8E5M2FNUZ),
        ],
    )
    def test_export_variable_tensor(self, dtype, expected_type):
        name = "variable_tensor"
        shape = (3, 224, 224)

        tensor = Variable(dtype=dtype, shape=shape, name=name)
        onnx_tensor = OnnxExporter.export_value_info_proto(tensor, do_type_check=True)
        assert onnx_tensor.name == name
        assert onnx_tensor.type.tensor_type.elem_type == expected_type

        onnx_shape = []
        for dim in onnx_tensor.type.tensor_type.shape.dim:
            onnx_shape.append(dim.dim_value)
        assert tuple(onnx_shape) == shape

    def test_export_variable_tensor_empty_dim_param(self):
        shape = ("", 224, 224)

        tensor = Variable(dtype=np.float32, shape=shape, name="variable_tensor")
        onnx_tensor = OnnxExporter.export_value_info_proto(tensor, do_type_check=True)

        onnx_shape = []
        for dim in onnx_tensor.type.tensor_type.shape.dim:
            onnx_shape.append(
                dim.dim_value if dim.HasField("dim_value") else dim.dim_param
            )
        assert tuple(onnx_shape) == shape

    # When a tensor shape is unknown, we should leave the shape field empty.
    def test_export_variable_tensor_empty_shape(self):
        shape = None

        tensor = Variable(dtype=np.float32, shape=shape, name="variable_tensor")
        onnx_tensor = OnnxExporter.export_value_info_proto(tensor, do_type_check=True)
        assert not onnx_tensor.type.tensor_type.HasField("shape")

    # When a tensor shape is unknown, we should leave the shape field empty.
    def test_export_variable_tensor_scalar_shape(self):
        shape = [None]

        tensor = Variable(dtype=np.float32, shape=shape, name="variable_tensor")
        onnx_tensor = OnnxExporter.export_value_info_proto(tensor, do_type_check=True)
        assert not onnx_tensor.type.tensor_type.shape.dim[0].HasField("dim_param")
        assert not onnx_tensor.type.tensor_type.shape.dim[0].HasField("dim_value")

    # TODO: Test subgraph export.
    def test_export_node(self):
        name = "TestNode"
        op = "Test"
        inputs = [Variable(name="input")]
        outputs = [Variable(name="output")]
        attrs = OrderedDict()
        attrs["float_attr"] = 4.0
        attrs["int_attr"] = 10
        attrs["str_attr"] = "constant"
        attrs["tensor_attr"] = Constant(
            "testTensor", np.ones(shape=(1, 2, 3, 4), dtype=np.float32)
        )
        attrs["floats_attr"] = [1.0, 2.0, 3.0, 4.0]
        attrs["ints_attr"] = [4, 3, 2, 1]
        attrs["strings_attr"] = ["constant", "and", "variable"]
        attrs["dtype_attr"] = np.float32
        node = Node(op=op, name=name, inputs=inputs, outputs=outputs, attrs=attrs)

        onnx_node = OnnxExporter.export_node(node)
        assert onnx_node.name == name
        assert onnx_node.op_type == op
        assert onnx_node.input == ["input"]
        assert onnx_node.output == ["output"]
        for onnx_attr, (name, attr) in zip(onnx_node.attribute, attrs.items()):
            assert onnx_attr.name == name
            if isinstance(attr, float):
                assert onnx_attr.f == attr
            elif isinstance(attr, int):
                assert onnx_attr.i == attr
            elif isinstance(attr, str):
                assert onnx_attr.s.decode() == attr
            elif isinstance(attr, Tensor):
                assert (
                    onnx_attr.t.SerializeToString()
                    == OnnxExporter.export_tensor_proto(attr).SerializeToString()
                )
            elif isinstance(attr, list):
                if isinstance(attr[0], float):
                    assert onnx_attr.floats == attr
                elif isinstance(attr[0], int):
                    assert onnx_attr.ints == attr
                elif isinstance(attr[0], str):
                    assert [s.decode() for s in onnx_attr.strings] == attr
                else:
                    raise AssertionError(
                        "Unrecognized list attribute: ({:}: {:}) of type: {:}".format(
                            name, attr, type(attr)
                        )
                    )
            elif isinstance(attr, type):
                assert onnx_attr.i == onnx.helper.np_dtype_to_tensor_dtype(
                    np.dtype(attr)
                )
            else:
                raise AssertionError(
                    "Unrecognized attribute: ({:}: {:}) of type: {:}".format(
                        name, attr, type(attr)
                    )
                )

    def test_export_node_ref_attrs(self):
        op = "Test"
        inputs = [Variable(name="input")]
        outputs = [Variable(name="output")]
        attrs = OrderedDict(
            {
                "attr1": 1,
                "attr2": 2.0,
                "attr3": Node.AttributeRef("attr4", int),
            }
        )
        node = Node(op=op, inputs=inputs, outputs=outputs, attrs=attrs)

        onnx_node = OnnxExporter.export_node(node)

        assert onnx_node.attribute[0].name == "attr1"
        assert onnx_node.attribute[0].i == attrs["attr1"]
        assert onnx_node.attribute[1].name == "attr2"
        assert onnx_node.attribute[1].f == attrs["attr2"]
        assert onnx_node.attribute[2].name == "attr3"
        assert onnx_node.attribute[2].ref_attr_name == "attr4"
        assert onnx_node.attribute[2].type == onnx.AttributeProto.INT

    def test_export_function(self):
        name = "Test"
        domain = "org.test"
        W = Variable("W", dtype=np.float32)
        X = Variable("X", dtype=np.float32)
        Y = Variable("Y", dtype=np.float32)
        Z = Variable("Z", dtype=np.float32)
        nodes = [
            Node("Add", inputs=[W, X], outputs=[Y]),
            Node("Mul", inputs=[X, Y], outputs=[Z]),
        ]
        inputs = [W, X]
        outputs = [Z]
        doc_string = "docstring"
        opset = 15
        attributes = {"attr1": None, "attr2": 2.0, "attr3": None}
        func = Function(
            name,
            domain=domain,
            nodes=nodes,
            inputs=inputs,
            outputs=outputs,
            doc_string=doc_string,
            opset=opset,
            attrs=attributes,
        )
        func.functions = [func]
        onnx_func = OnnxExporter.export_function(func)

        assert onnx_func.name == name
        assert onnx_func.domain == domain
        assert onnx_func.doc_string == doc_string
        assert sorted(onnx_func.attribute) == sorted(
            [name for name, val in attributes.items() if val is None]
        )
        assert len(onnx_func.attribute_proto) == 1
        assert onnx_func.attribute_proto[0].name == "attr2"
        assert onnx_func.attribute_proto[0].f == 2.0
        assert sorted(onnx_func.input) == sorted([t.name for t in inputs])
        assert sorted(onnx_func.output) == sorted([t.name for t in outputs])
        assert sorted([n.op_type for n in onnx_func.node]) == sorted(
            [n.op for n in nodes]
        )
        assert onnx_func.opset_import[0].version == opset

    # See test_importers for import correctness checks
    # This function first imports an ONNX graph, and then re-exports it with no changes.
    # The exported ONNX graph should exactly match the original.
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
            sparse_nnz_model(),
            sparse_nnz_rank_model(),
        ],
        ids=lambda model: str(model),
    )
    def test_export_graph(self, model):
        onnx_graph = model.load().graph
        graph = OnnxImporter.import_graph(onnx_graph)
        exported_onnx_graph = OnnxExporter.export_graph(graph)
        reimported_graph = OnnxImporter.import_graph(exported_onnx_graph)
        assert graph == reimported_graph
        assert graph.opset == reimported_graph.opset

        # ONNX exports the initializers in this model differently after importing - ONNX GS can't do much about this.
        if model.path != lstm_model().path:
            assert onnx_graph == exported_onnx_graph
