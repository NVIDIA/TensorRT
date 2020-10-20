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

from onnx_graphsurgeon.exporters.onnx_exporter import OnnxExporter
from onnx_graphsurgeon.importers.onnx_importer import OnnxImporter
from onnx_graphsurgeon.logger.logger import G_LOGGER

from onnx_models import identity_model, lstm_model, scan_model, dim_param_model, initializer_is_output_model

from onnx_graphsurgeon.ir.tensor import Tensor, Constant, Variable
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node

from collections import OrderedDict
import onnx.numpy_helper
import numpy as np
import pytest
import onnx

class TestOnnxExporter(object):
    def test_export_constant_tensor_to_tensor_proto(self):
        name = "constant_tensor"
        shape = (3, 224, 224)
        values = np.random.random_sample(size=shape).astype(np.float32)

        tensor = Constant(name=name, values=values)
        onnx_tensor = OnnxExporter.export_tensor_proto(tensor)
        assert onnx_tensor.name == name
        assert np.all(onnx.numpy_helper.to_array(onnx_tensor) == values)
        assert onnx_tensor.data_type == onnx.TensorProto.FLOAT
        assert tuple(onnx_tensor.dims) == shape


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


    def test_export_variable_tensor(self):
        name = "variable_tensor"
        shape = (3, 224, 224)
        dtype = np.float32

        tensor = Variable(dtype=dtype, shape=shape, name=name)
        onnx_tensor = OnnxExporter.export_value_info_proto(tensor, do_type_check=True)
        assert onnx_tensor.name == name
        assert onnx_tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT

        onnx_shape = []
        for dim in onnx_tensor.type.tensor_type.shape.dim:
            onnx_shape.append(dim.dim_value)
        assert tuple(onnx_shape) == shape


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
        attrs["tensor_attr"] = Constant("testTensor", np.ones(shape=(1, 2, 3, 4), dtype=np.float32))
        attrs["floats_attr"] = [1.0, 2.0, 3.0, 4.0]
        attrs["ints_attr"] = [4, 3, 2, 1]
        attrs["strings_attr"] = ["constant", "and", "variable"]
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
                assert onnx_attr.t.SerializeToString() == OnnxExporter.export_tensor_proto(attr).SerializeToString()
            elif isinstance(attr, list):
                if isinstance(attr[0], float):
                    assert onnx_attr.floats == attr
                elif isinstance(attr[0], int):
                    assert onnx_attr.ints == attr
                elif isinstance(attr[0], str):
                    assert [s.decode() for s in onnx_attr.strings] == attr
                else:
                    raise AssertionError("Unrecognized list attribute: ({:}: {:}) of type: {:}".format(name, attr, type(attr)))
            else:
                raise AssertionError("Unrecognized attribute: ({:}: {:}) of type: {:}".format(name, attr, type(attr)))


    # See test_importers for import correctness checks
    # This function first imports an ONNX graph, and then re-exports it with no changes.
    # The exported ONNX graph should exactly match the original.
    @pytest.mark.parametrize("model", [identity_model(), lstm_model(), scan_model(), dim_param_model(), initializer_is_output_model()])
    def test_export_graph(self, model):
        onnx_graph = model.load().graph
        graph = OnnxImporter.import_graph(onnx_graph)
        exported_onnx_graph = OnnxExporter.export_graph(graph)
        imported_graph = OnnxImporter.import_graph(exported_onnx_graph)
        assert graph == imported_graph
        assert graph.opset == imported_graph.opset
        # ONNX exports the initializers in this model differently after importing - ONNX GS can't do much about this.
        if model.path != lstm_model().path:
            assert onnx_graph == exported_onnx_graph
