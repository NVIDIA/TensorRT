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
import pytest
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.ir.tensor import Constant, LazyValues, Variable
from onnx_graphsurgeon.logger.logger import G_LOGGER

G_LOGGER.severity = G_LOGGER.ULTRA_VERBOSE


class TensorBaseTests(object):
    def test_can_convert_in_place_to_constant(self):
        tensor = self.tensor.to_constant(values=np.ones((1, 3, 5, 5), dtype=np.float64))
        assert tensor is self.tensor
        assert isinstance(tensor, Constant)
        assert isinstance(self.input_node.outputs[0], Constant)
        assert isinstance(self.output_node.inputs[0], Constant)
        assert tensor.shape == (1, 3, 5, 5)
        assert tensor.dtype == np.float64
        assert np.all(self.input_node.outputs[0].values == tensor.values)
        assert np.all(self.output_node.inputs[0].values == tensor.values)

    def test_can_convert_in_place_to_variable(self):
        tensor = self.tensor.to_variable(dtype=np.float32, shape=(1, 3, 224, 224))
        assert tensor is self.tensor
        assert isinstance(tensor, Variable)
        assert isinstance(self.input_node.outputs[0], Variable)
        assert tensor.dtype == np.float32
        assert tensor.shape == (1, 3, 224, 224)
        assert self.input_node.outputs[0].dtype == tensor.dtype
        assert self.input_node.outputs[0].shape == tensor.shape

    def test_equals(self):
        assert self.tensor == self.tensor

    def test_set_inputs_updates_old_inputs(self):
        dummy = Node(op="dummy")
        self.tensor.inputs = [dummy]
        assert len(self.input_node.outputs) == 0
        assert dummy.outputs[0] == self.tensor

    def test_set_outputs_updates_old_outputs(self):
        dummy = Node(op="dummy")
        self.tensor.outputs = [dummy]
        assert len(self.output_node.inputs) == 0
        assert dummy.inputs[0] == self.tensor

    def test_can_copy_inputs_from_other_node(self):
        tensor = Variable(name="other_test_tensor")
        tensor.inputs = self.tensor.inputs
        assert tensor.inputs == self.tensor.inputs
        # Contents should be the same, but it should not just be a reference to the existing SynchronizedList
        assert tensor.inputs is not self.tensor.inputs

    def test_can_copy_outputs_from_other_node(self):
        tensor = Variable(name="other_test_tensor")
        tensor.outputs = self.tensor.outputs
        assert tensor.outputs == self.tensor.outputs
        assert tensor.outputs is not self.tensor.outputs

    def test_i(self):
        x = Variable(name="x")
        y = Variable(name="y")
        node = Node(op="Add", name="Input", inputs=[x], outputs=[y])
        assert y.i() == x

    def test_i_multiple_inputs(self):
        x = Variable(name="x")
        x2 = Variable(name="x2")
        y = Variable(name="y")
        node = Node(op="Add", name="Input", inputs=[x, x2], outputs=[y])
        assert y.i() == x
        assert y.i(1) == x2

    def test_o(self):
        x = Variable(name="x")
        y = Variable(name="y")
        node = Node(op="Add", name="Input", inputs=[x], outputs=[y])
        assert x.o() == y

    def test_o_multiple_outputs(self):
        x = Variable(name="x")
        y = Variable(name="y")
        y2 = Variable(name="y2")
        node = Node(op="Add", name="Input", inputs=[x], outputs=[y])
        node2 = Node(op="Add", name="Input", inputs=[x], outputs=[y2])
        assert x.o() == y
        assert x.o(1) == y2


class TestVariable(TensorBaseTests):
    def setup_method(self):
        self.tensor = Variable(name="test_tensor", dtype=np.float32, shape=(1, 3, 224, 224))
        self.input_node = Node(op="Add", outputs=[self.tensor])
        self.output_node = Node(op="Add", inputs=[self.tensor])

    def test_equals_name_mismatch(self):
        tensor = Variable(name="test_tensor0", dtype=np.float32, shape=(1, 3, 224, 224))
        assert not self.tensor == tensor


class TestConstant(TensorBaseTests):
    def setup_method(self):
        self.tensor = Constant(name="test_tensor", values=np.ones((1, 3, 5, 5), dtype=np.float64))
        self.input_node = Node(
            op="Add", outputs=[self.tensor]
        )  # Doesn't make sense for Constants, but needed to make base tests happy.
        self.output_node = Node(op="Add", inputs=[self.tensor])

    def test_can_get_shape(self):
        assert self.tensor.shape == (1, 3, 5, 5)

    def test_can_get_dtype(self):
        assert self.tensor.dtype == np.float64


class TestNode(object):
    def setup_method(self):
        self.input_tensor = Variable(name="x")
        self.output_tensor = Variable(name="y")
        self.node = Node(op="Add", name="Test", inputs=[self.input_tensor], outputs=[self.output_tensor])

    def test_equals(self):
        assert self.node == self.node

    def test_equals_name_mismatch(self):
        node = Node(op="Add", name="OtherTest")
        assert not self.node == node

    def test_equals_op_mismatch(self):
        node = Node(op="Subtract", name="Test")
        assert not self.node == node

    def test_equals_num_inputs_mismatch(self):
        node = Node(op="Subtract", name="Test")
        assert not self.node == node

    def test_equals(self):
        assert self.node == self.node

    def test_equals_inputs_mismatch(self):
        tensor = Variable(name="other_tensor")
        assert not self.input_tensor == tensor

        node = Node(op="Add", name="Test", inputs=[tensor])
        assert not self.node == node

    def test_set_inputs_updates_old_inputs(self):
        dummy = Variable(name="dummy")
        self.node.inputs = [dummy]
        assert len(self.input_tensor.outputs) == 0
        assert dummy.outputs[0] == self.node

    def test_set_outputs_updates_old_outputs(self):
        dummy = Variable(name="dummy")
        self.node.outputs = [dummy]
        assert len(self.output_tensor.inputs) == 0
        assert dummy.inputs[0] == self.node

    def test_can_copy_inputs_from_other_node(self):
        node = Node(op="Subtract")
        node.inputs = self.node.inputs
        assert node.inputs == self.node.inputs
        # Contents should be the same, but it should not just be a reference to the existing SynchronizedList
        assert node.inputs is not self.node.inputs

    def test_can_copy_outputs_from_other_node(self):
        node = Node(op="Subtract")
        node.outputs = self.node.outputs
        assert node.outputs == self.node.outputs
        assert node.outputs is not self.node.outputs

    def test_i(self):
        intermediate_tensor = Variable(name="intermediate")
        input_node = Node(op="Add", name="Input", inputs=[self.input_tensor], outputs=[intermediate_tensor])
        output_node = Node(op="Add", name="Out", inputs=[intermediate_tensor], outputs=[self.output_tensor])
        assert output_node.i() == input_node

    def test_i_multiple_inputs(self):
        intermediate_tensor = Variable(name="intermediate")
        intermediate_tensor2 = Variable(name="intermediate2")
        input_node = Node(op="Add", name="Input", inputs=[self.input_tensor], outputs=[intermediate_tensor])
        input_node2 = Node(op="Add", name="Input2", inputs=[self.input_tensor], outputs=[intermediate_tensor2])
        output_node = Node(
            op="Add", name="Out", inputs=[intermediate_tensor, intermediate_tensor2], outputs=[self.output_tensor]
        )
        assert output_node.i() == input_node
        assert output_node.i(1) == input_node2

    def test_o(self):
        intermediate_tensor = Variable(name="intermediate")
        input_node = Node(op="Add", name="Input", inputs=[self.input_tensor], outputs=[intermediate_tensor])
        output_node = Node(op="Add", name="Out", inputs=[intermediate_tensor], outputs=[self.output_tensor])
        assert input_node.o() == output_node

    def test_o_multiple_outputs(self):
        intermediate_tensor = Variable(name="intermediate")
        intermediate_tensor2 = Variable(name="intermediate2")
        input_node = Node(op="Add", name="Input", inputs=[self.input_tensor], outputs=[intermediate_tensor])
        output_node = Node(op="Add", name="Out", inputs=[intermediate_tensor], outputs=[self.output_tensor])
        output_node2 = Node(op="Add", name="Input2", inputs=[intermediate_tensor], outputs=[intermediate_tensor2])
        assert input_node.o() == output_node
        assert input_node.o(1) == output_node2

    def test_domain(self):
        node = Node(op="Add", domain="test")
        assert node.domain == "test"


class TestNodeIO(object):
    def setup_method(self, field_names):
        self.tensors = [
            Variable(name="test_tensor_{:}".format(i), dtype=np.float32, shape=(1, 3, 224, 224)) for i in range(10)
        ]
        self.node = Node(op="Dummy")

    def get_lists(self, field_names):
        return getattr(self.node, field_names[0]), field_names[1]

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_append(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist.append(self.tensors[0])
        assert nlist[0] == self.tensors[0]
        assert getattr(self.tensors[0], tensor_field)[0] == self.node

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_extend(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist.extend(self.tensors)
        for tensor in self.tensors:
            assert tensor in nlist
            assert getattr(tensor, tensor_field)[0] == self.node

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_insert(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist.append(self.tensors[1])
        nlist.insert(0, self.tensors[0])
        assert nlist[0] == self.tensors[0]
        assert getattr(self.tensors[0], tensor_field)[0] == self.node

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_remove(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist.append(self.tensors[0])
        nlist.remove(self.tensors[0])
        assert len(nlist) == 0
        assert len(getattr(self.tensors[0], tensor_field)) == 0

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_pop(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist.append(self.tensors[0])
        tensor = nlist.pop()
        assert len(nlist) == 0
        assert len(getattr(tensor, tensor_field)) == 0

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_pop_index(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist.extend(self.tensors)
        tensor = nlist.pop(1)
        assert self.tensors[1] not in nlist
        assert len(getattr(tensor, tensor_field)) == 0

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_del_index(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist.extend(self.tensors)
        tensor = nlist[1]
        del nlist[1]
        assert self.tensors[1] not in nlist
        assert len(getattr(tensor, tensor_field)) == 0

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_clear(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist.extend(self.tensors)
        nlist.clear()
        assert len(nlist) == 0
        assert all([len(getattr(tensor, tensor_field)) == 0 for tensor in self.tensors])

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_add(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist = nlist + self.tensors
        for tensor in self.tensors:
            assert tensor in nlist

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_iadd(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist += self.tensors
        for tensor in self.tensors:
            assert tensor in nlist
            assert getattr(tensor, tensor_field)[0] == self.node

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_setitem(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist.append(self.tensors[0])
        new_tensor = Variable("new_tensor")
        nlist[0] = new_tensor
        assert nlist[0] == new_tensor
        assert len(getattr(self.tensors[0], tensor_field)) == 0
        assert getattr(new_tensor, tensor_field)[0] == self.node

    def test_iadd_on_node_directly(self):
        t0 = Variable("t0")
        n0 = Node("", inputs=[])

        n0.inputs += [t0]
        assert len(n0.inputs) == 1
        assert n0.inputs[0] == t0


class TestTensorIO(object):
    def test_iadd_on_tensor_directly(self):
        n0 = Node("")
        t0 = Variable("t0")

        t0.inputs += [n0]
        assert len(t0.inputs) == 1
        assert t0.inputs[0] == n0


class TestLazyValues(object):
    def test_basic(self):
        shape = (1, 5, 5)
        onnx_tensor = onnx.helper.make_tensor_value_info("test", onnx.TensorProto.FLOAT, shape)
        values = LazyValues(onnx_tensor)

        assert values.dtype == np.float32
        assert tuple(values.shape) == shape
        assert values.nbytes == 100
