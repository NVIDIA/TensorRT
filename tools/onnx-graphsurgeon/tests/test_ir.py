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

from onnx_graphsurgeon.util.exception import OnnxGraphSurgeonException
from onnx_graphsurgeon.ir.tensor import Constant, Variable
from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node

import numpy as np
import pytest
import onnx
import copy

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

    def test_can_copy_outputs_from_other_node(self):
        tensor = Variable(name="other_test_tensor")
        tensor.outputs = self.tensor.outputs
        assert tensor.outputs == self.tensor.outputs

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
        self.input_node = Node(op="Add", outputs=[self.tensor]) # Doesn't make sense for Constants, but needed to make base tests happy.
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

    def test_can_copy_outputs_from_other_node(self):
        node = Node(op="Subtract")
        node.outputs = self.node.outputs
        assert node.outputs == self.node.outputs

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
        output_node = Node(op="Add", name="Out", inputs=[intermediate_tensor, intermediate_tensor2], outputs=[self.output_tensor])
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


class TestNodeIO(object):
    def setup_method(self, field_names):
        self.tensors = [Variable(name="test_tensor_{:}".format(i), dtype=np.float32, shape=(1, 3, 224, 224)) for i in range(10)]
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
            assert getattr(tensor, tensor_field)[0] == self.node

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


def tensors_linear_graph():
    inputs = [Variable(name="x")]
    intermediate0 = Variable(name="intermediate0")
    intermediate1 = Variable(name="intermediate1")
    intermediate2 = Variable(name="intermediate2")
    outputs = [Variable(name="y")]

    tensors = inputs + [intermediate0, intermediate1, intermediate2] + outputs
    tensors = {tensor.name: tensor for tensor in tensors}
    # Nodes are NOT in topo order.
    nodes = [
        Node(op="Add", name="Test0", inputs=inputs, outputs=[intermediate0]),
        Node(op="Add", name="Test1", inputs=[intermediate0], outputs=[intermediate1]),
        Node(op="Add", name="Test2", inputs=[intermediate1], outputs=[intermediate2]),
        Node(op="Add", name="Test3", inputs=[intermediate2], outputs=outputs),
    ]
    return Graph(nodes=nodes, inputs=inputs, outputs=outputs), nodes, tensors


def build_basic_graph():
    inputs = [Variable(name="x")]
    outputs = [Variable(name="y")]
    nodes = [
        Node(op="Add", name="Test", inputs=inputs, outputs=outputs),
    ]
    return Graph(nodes=nodes, inputs=inputs, outputs=outputs)


def build_two_layer_graph():
    inputs = [Variable(name="x")]
    intermediate_tensor = Variable(name="intermediate")
    outputs = [Variable(name="y")]
    nodes = [
        Node(op="Add", name="Test0", inputs=inputs, outputs=[intermediate_tensor]),
        Node(op="Add", name="Test1", inputs=[intermediate_tensor], outputs=outputs),
    ]
    return Graph(nodes=nodes, inputs=inputs, outputs=outputs)


def build_two_layer_graph_multiple_io():
    inputs = [Variable(name="x0"), Variable(name="x1")]
    intermediate_tensor = Variable(name="intermediate")
    outputs = [Variable(name="y0"), Variable(name="y1")]
    nodes = [
        Node(op="Add", name="Test0", inputs=inputs, outputs=[intermediate_tensor]),
        Node(op="Add", name="Test1", inputs=[intermediate_tensor], outputs=outputs),
    ]
    return Graph(nodes=nodes, inputs=inputs, outputs=outputs)


GRAPH_TEST_CASES = [
    build_basic_graph(),
    build_two_layer_graph(),
    build_two_layer_graph_multiple_io(),
]


def toposort_linear_graph():
    inputs = [Variable(name="x")]
    intermediate0 = Variable(name="intermediate0")
    intermediate1 = Variable(name="intermediate1")
    intermediate2 = Variable(name="intermediate2")
    outputs = [Variable(name="y")]
    # Nodes are NOT in topo order.
    nodes = [
        Node(op="Add", name="Test0", inputs=inputs, outputs=[intermediate0]),
        Node(op="Add", name="Test2", inputs=[intermediate1], outputs=[intermediate2]),
        Node(op="Add", name="Test3", inputs=[intermediate2], outputs=outputs),
        Node(op="Add", name="Test1", inputs=[intermediate0], outputs=[intermediate1]),
    ]
    expected_node_order = [nodes[0], nodes[3], nodes[1], nodes[2]]
    return Graph(nodes=nodes, inputs=inputs, outputs=outputs), expected_node_order


# Graph structure:
# x
# |
# Test0 -> out0 (graph output)
# |
# out0
# |
# Test1 -> out1 (graph output)
# |
# out1
# |
# Test2 -> out2 (graph_output)
def toposort_multi_tier_output_graph():
    inputs = [Variable(name="x")]
    outputs = [Variable(name="out0"), Variable(name="out1"), Variable(name="out2")]
    out0, out1, out2 = outputs
    nodes = [
        Node(op="Add", name="Test2", inputs=[out1], outputs=[out2]),
        Node(op="Add", name="Test0", inputs=inputs, outputs=[out0]),
        Node(op="Add", name="Test1", inputs=[out0], outputs=[out1]),
    ]
    expected_node_order = [nodes[1], nodes[2], nodes[0]]
    return Graph(nodes=nodes, inputs=inputs, outputs=outputs), expected_node_order


# Graph structure:
# x2  x1
# |   |
# Test0
# |
# int0  x0
# |    /
# Test1
# |
# int1  x3
# |    /
# Test2 -> out (graph_output)
def toposort_multi_tier_input_graph():
    inputs = [Variable(name="x0"), Variable(name="x1"), Variable(name="x2"), Variable(name="x3")]
    int0, int1 = [Variable(name="intermediate0"), Variable(name="intermediate1")]
    outputs = [Variable(name="out")]
    x0, x1, x2, x3 = inputs
    nodes = [
        Node(op="Add", name="Test2", inputs=[int1, x3], outputs=outputs),
        Node(op="Add", name="Test0", inputs=[x2, x1], outputs=[int0]),
        Node(op="Add", name="Test1", inputs=[int0, x0], outputs=[int1]),
    ]
    expected_node_order = [nodes[1], nodes[2], nodes[0]]
    return Graph(nodes=nodes, inputs=inputs, outputs=outputs), expected_node_order


TOPOSORT_TEST_CASES = [
    toposort_linear_graph(),
    toposort_multi_tier_output_graph(),
    toposort_multi_tier_input_graph(),
]

class TestGraph(object):
    def test_generate_name(self):
        graph = Graph()
        names = set()
        num_names = 100
        # This function should not return the same name more than once
        for idx in range(num_names):
            names.add(graph._generate_name("name"))
        assert len(names) == 100


    def test_register(self):
        @Graph.register()
        def add(self, a, b):
            return self.layer(op="Add", inputs=[a, b], outputs=["add_out"])

        graph = Graph()
        [output] = graph.add("a", "b")
        assert "add_out" in output.name
        assert len(graph.nodes) == 1
        assert graph.nodes[-1].op == "Add"


    def test_register_opset(self):
        @Graph.register(opsets=[11])
        def add(self, a, b):
            return self.layer(op="Add", inputs=[a, b], outputs=["add_out"])

        @Graph.register(opsets=[10])
        def add(self, a, b):
            return self.layer(op="Add-10", inputs=[a, b], outputs=["add_out"])

        graph = Graph()
        [output] = graph.add("a", "b")
        assert "add_out" in output.name
        assert len(graph.nodes) == 1
        assert graph.nodes[-1].op == "Add"

        graph_opset10 = Graph(opset=10)
        [output] = graph_opset10.add("a", "b")
        assert "add_out" in output.name
        assert len(graph_opset10.nodes) == 1
        assert graph_opset10.nodes[-1].op == "Add-10"


    def test_layer_with_attrs(self):
        graph = Graph()
        outputs = graph.layer(op="Add", name="node", attrs={"fake_attr": 0})
        assert len(graph.nodes) == 1
        assert graph.nodes[-1].op == "Add"
        assert graph.nodes[-1].name == "node"
        assert graph.nodes[-1].attrs["fake_attr"] == 0


    def test_layer_with_tensors(self):
        x0 = Variable("x0")
        x1 = Variable("x1")
        y0 = Variable("y0")
        y1 = Variable("y1")
        graph = Graph()

        outputs = graph.layer(op="Fake", inputs=[x0, x1], outputs=[y0, y1])
        assert outputs == [y0, y1]
        assert len(graph.nodes) == 1
        assert graph.nodes[-1].inputs == [x0, x1]
        assert graph.nodes[-1].outputs == outputs


    def test_layer_with_strings(self):
        x0 = "x0"
        x1 = "x1"
        y0 = "y0"
        y1 = "y1"
        graph = Graph()

        outputs = graph.layer(op="Fake", inputs=[x0, x1], outputs=[y0, y1])
        assert len(graph.nodes) == 1
        assert [prefix in tensor.name for prefix, tensor in zip([x0, x1], graph.nodes[-1].inputs)]
        assert [prefix in tensor.name for prefix, tensor in zip([y0, y1], graph.nodes[-1].outputs)]
        assert graph.nodes[-1].outputs == outputs


    def test_layer_with_arrays(self):
        x0 = np.array([1])
        x1 = np.array([1])
        y0 = "y0"
        y1 = "y1"
        graph = Graph()

        outputs = graph.layer(op="Fake", inputs=[x0, x1], outputs=[y0, y1])
        assert [prefix in tensor.name for prefix, tensor in zip([y0, y1], graph.nodes[-1].outputs)]
        assert len(graph.nodes) == 1
        assert graph.nodes[-1].inputs[0].values == x0
        assert graph.nodes[-1].inputs[1].values == x1
        assert graph.nodes[-1].outputs == outputs


    def test_tensors(self):
        graph, nodes, tensors = tensors_linear_graph()
        graph_tensors = graph.tensors()
        for name, tensor in tensors.items():
            assert name in graph_tensors
            assert tensor is graph_tensors[name]

        for name, tensor in graph_tensors.items():
            assert name in tensors
            assert tensor is tensors[name]


    def test_tensors_check_duplicates(self):
        inputs = [Variable(name="x")]
        outputs = [Variable(name="x")] # Distinct tensors with the same name
        nodes = [
            Node(op="Add", name="Test", inputs=inputs, outputs=outputs),
        ]
        graph = Graph(nodes=nodes, inputs=inputs, outputs=outputs)

        with pytest.raises(OnnxGraphSurgeonException):
            graph.tensors(check_duplicates=True)


    @pytest.mark.parametrize("graph", GRAPH_TEST_CASES)
    def test_get_used_node_ids(self, graph):
        graph_used_nodes = copy.copy(graph.nodes)
        graph_used_tensors = copy.copy(list(graph.tensors().values()))

        unused_tensor = Variable(name="Unused")
        unused_node = Node(op="Unused", inputs=[graph.inputs[0]], outputs=[unused_tensor])
        graph.nodes.append(unused_node)

        with graph.node_ids():
            used_node_ids, used_tensors = graph._get_used_node_ids()
            assert len(used_node_ids) == len(graph.nodes) - 1
            assert all([node.id in used_node_ids for node in graph_used_nodes])
            assert unused_node.id not in used_node_ids
            assert unused_tensor not in used_tensors
            assert all([used_tensor in used_tensors for used_tensor in graph_used_tensors])


    @pytest.mark.parametrize("toposort_test_case", TOPOSORT_TEST_CASES)
    def test_topologically_sort(self, toposort_test_case):
        graph, expected_node_order = toposort_test_case
        assert graph.nodes != expected_node_order
        graph.toposort()
        assert graph.nodes == expected_node_order


    def test_cleanup_multi_tier(self):
        graph, _ = toposort_multi_tier_output_graph()
        tensor = graph.outputs.pop()
        unused_node = tensor.inputs[0]
        graph.cleanup() # Should remove just the Test2 node as out1 is still an output.
        assert unused_node not in graph.nodes
        assert len(graph.nodes) == 2
        assert len(graph.outputs) == 2

        tensor_map = graph.tensors()
        assert tensor.name not in tensor_map


    def test_cleanup_intermediate_tensors(self):
        graph, _  = toposort_linear_graph()
        graph.toposort()
        graph_output = graph.outputs[0]

        dummy = Variable("dummy")
        # Add unused tensor to a node in the middle of the graph.
        # Since it does not contribute to graph outputs, it should be removed.
        graph.nodes[1].outputs.append(dummy)

        graph.cleanup()
        assert dummy not in graph.nodes[1].outputs
        assert graph.outputs[0] == graph_output # Graoh outputs will never be removed


    def test_cleanup_independent_path(self):
        graph, _ = toposort_linear_graph()
        # Build out a path totally unrelated to rest of the graph
        indep0 = Variable(name="indep0")
        indep1 = Variable(name="indep1")
        node = Node(op="IndepTest", inputs=[indep0], outputs=[indep1])
        graph.inputs.append(indep0) # Unused inputs should be removed as well
        graph.nodes.append(node)
        graph.cleanup()
        assert indep0 not in graph.inputs
        assert node not in graph.nodes

        tensor_map = graph.tensors()
        assert indep0.name not in tensor_map
        assert indep1.name not in tensor_map


    def test_deep_copy(self):
        def make_graph():
            graph, _ = toposort_multi_tier_output_graph()
            graph.outputs.pop()
            return graph

        graph = make_graph()
        new_graph = copy.deepcopy(graph)
        assert graph == new_graph

        # Running cleanup on the first graph should not affect the copy
        graph.cleanup()
        assert graph != new_graph
        assert new_graph == make_graph()


    def test_fold_constants(self):
        # Graph:
        # c = (a + b)
        # output = input + c
        # Should fold to:
        # output = input + c
        inp = Variable("input", shape=(1, 3), dtype=np.float32)
        a = Constant("a", values=np.ones(shape=(1, 3), dtype=np.float32))
        b = Constant("b", values=np.ones(shape=(1, 3), dtype=np.float32))
        c = Variable("c", shape=(1, 3), dtype=np.float32)
        out = Variable("output", shape=(1, 3), dtype=np.float32)

        nodes = [
            Node("Add", inputs=[a, b], outputs=[c]),
            Node("Add", inputs=[inp, c], outputs=[out]),
        ]
        graph = Graph(nodes=nodes, inputs=[inp], outputs=[out])

        graph.fold_constants().cleanup()

        # Extra node should be removed
        assert len(graph.nodes) == 1
        assert graph.nodes[0].inputs[0] == inp
        assert graph.nodes[0].inputs[1] == c
        # Value should be computed correctly
        assert np.all(graph.nodes[0].inputs[1].values == np.ones(shape=(1, 3), dtype=np.float32) * 2)


    def test_fold_constants_one_hop(self):
        # Graph:
        # c = (a + b)
        # e = (c + d)
        # output = input + e
        # Should fold to:
        # output = input + e
        inp = Variable("input", shape=(1, 3), dtype=np.float32)
        a = Constant("a", values=np.ones(shape=(1, 3), dtype=np.float32))
        b = Constant("b", values=np.ones(shape=(1, 3), dtype=np.float32))
        c = Variable("c", shape=(1, 3), dtype=np.float32)
        d = Constant("d", values=np.ones(shape=(1, 3), dtype=np.float32))
        e = Variable("e", shape=(1, 3), dtype=np.float32)
        out = Variable("output", shape=(1, 3), dtype=np.float32)

        nodes = [
            Node("Add", inputs=[a, b], outputs=[c]),
            Node("Add", inputs=[c, d], outputs=[e]),
            Node("Add", inputs=[inp, e], outputs=[out]),
        ]

        graph = Graph(nodes=nodes, inputs=[inp], outputs=[out])

        graph.fold_constants().cleanup()

        # Extra nodes should be removed
        assert len(graph.nodes) == 1
        assert graph.nodes[0].inputs[0] == inp
        assert graph.nodes[0].inputs[1] == e
        # Value should be computed correctly
        assert np.all(graph.nodes[0].inputs[1].values == np.ones(shape=(1, 3), dtype=np.float32) * 3)


    def test_fold_constants_no_foldable_constants(self):
        inp0 = Variable("input0", shape=(1, 3), dtype=np.float32)
        inp1 = Variable("input1", shape=(1, 3), dtype=np.float32)
        out = Variable("output", shape=(1, 3), dtype=np.float32)

        nodes = [
            Node("Add", inputs=[inp0, inp1], outputs=[out])
        ]

        graph = Graph(nodes=nodes, inputs=[inp0, inp1], outputs=[out])

        graph.fold_constants().cleanup()

        assert len(graph.nodes) == 1
        assert graph.nodes[0].inputs == [inp0, inp1]
