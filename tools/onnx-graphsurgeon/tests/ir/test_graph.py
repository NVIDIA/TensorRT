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

import numpy as np
from numpy.core.numeric import identity
import pytest
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.ir.tensor import Constant, Variable
from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.util.exception import OnnxGraphSurgeonException
from onnx_graphsurgeon.util.misc import SynchronizedList

G_LOGGER.severity = G_LOGGER.ULTRA_VERBOSE


@Graph.register()
def shape(self, inp):
    return self.layer(op="Shape", inputs=[inp], outputs=["shape_out"])[0]


@Graph.register()
def constant(self, values):
    return self.layer(op="Constant", inputs=[], outputs=["constant_out"], attrs={"value": Constant("values", values)})[0]


@Graph.register()
def identity(self, inp):
    out = self.layer(op="Identity", inputs=[inp], outputs=["identity_out"])[0]
    out.dtype = inp.dtype
    return out


@Graph.register()
def add(self, a, b, name=None):
    outputs = [Variable(name=name)] if name else ["add_out"]
    out = self.layer(op="Add", inputs=[a, b], outputs=outputs)[0]
    out.dtype = a.dtype or b.dtype
    return out


# A fake op that can be used to ensure things work even when there is an invalid
# node present in the model.
@Graph.register()
def fake(self, inp, name=None):
    outputs = [Variable(name=name)] if name else ["fake_out"]
    out = self.layer(op="Fake", inputs=[inp], outputs=outputs)[0]
    out.dtype = inp.dtype
    return out


# Generates a graph where an outer node has no outputs except
# within the subgraph. ONNX-GS should recognize that the node
# is being used, and should not remove it during cleanup().
@pytest.fixture
def nested_graph():
    inp = Variable("input")
    id_out = Variable("id_out")
    identity = Node(op="Identity", inputs=[inp], outputs=[id_out])

    # Subgraph outputs come from the parent node, but nodes in the subgraph
    # can use nodes from the outer graphs too.
    subgraph_inputs = [Variable("subgraph_inp")]
    subgraph_id_out = Variable("subgraph_id_out")
    subgraph_outputs = [Variable("subgraph_out")]

    subgraph_identity0 = Node(op="Identity", inputs=[id_out], outputs=[subgraph_id_out])
    subgraph_identity1 = Node(op="Identity", inputs=[subgraph_id_out], outputs=subgraph_outputs)

    subgraph = Graph(nodes=[subgraph_identity0, subgraph_identity1],
                        inputs=subgraph_inputs, outputs=subgraph_outputs)

    nested_out = Variable("nested_out")
    nested_node = Node(op="Nested", attrs={"body": subgraph}, inputs=[inp], outputs=[nested_out])

    yield Graph(nodes=[identity, nested_node], inputs=[inp], outputs=[nested_out])


class TestBasic(object):
    def test_generate_name(self):
        graph = Graph()
        names = set()
        num_names = 100
        # This function should not return the same name more than once
        for idx in range(num_names):
            names.add(graph._generate_name("name"))
        assert len(names) == 100


class TestRegister(object):
    def test_register(self):
        @Graph.register()
        def fake_add(self, a, b):
            return self.layer(op="Add", inputs=[a, b], outputs=["add_out"])

        graph = Graph()
        [output] = graph.fake_add("a", "b")
        assert "add_out" in output.name
        assert len(graph.nodes) == 1
        assert graph.nodes[-1].op == "Add"


    def test_register_opset(self):
        @Graph.register(opsets=[11])
        def fake_add(self, a, b):
            return self.layer(op="Add", inputs=[a, b], outputs=["add_out"])

        @Graph.register(opsets=[10])
        def fake_add(self, a, b):
            return self.layer(op="Add-10", inputs=[a, b], outputs=["add_out"])

        graph = Graph()
        [output] = graph.fake_add("a", "b")
        assert "add_out" in output.name
        assert len(graph.nodes) == 1
        assert graph.nodes[-1].op == "Add"

        graph_opset10 = Graph(opset=10)
        [output] = graph_opset10.fake_add("a", "b")
        assert "add_out" in output.name
        assert len(graph_opset10.nodes) == 1
        assert graph_opset10.nodes[-1].op == "Add-10"


class TestLayer(object):
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


    def test_layer_with_iterables(self):
        x0 = [1]
        x1 = (1, )
        y0 = "y0"
        y1 = "y1"
        graph = Graph()

        outputs = graph.layer(op="Fake", inputs=[x0, x1], outputs=[y0, y1])
        assert [prefix in tensor.name for prefix, tensor in zip([y0, y1], graph.nodes[-1].outputs)]
        assert len(graph.nodes) == 1
        assert graph.nodes[-1].inputs[0].values == x0
        assert graph.nodes[-1].inputs[1].values == x1
        assert graph.nodes[-1].outputs == outputs



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



class TestTensors(object):
    # Calling `tensors()` should not modify tensors in the graph.
    def test_tensors_does_not_modify_tensors(self):
        graph, _, _ = tensors_linear_graph()
        graph_tensors = graph.tensors()
        # Generate a new graph to compare against
        _, _, tensors = tensors_linear_graph()

        assert set(tensors.keys()) == set(graph_tensors.keys())

        for name, tensor in tensors.items():
            graph_tensor = graph_tensors[name]
            assert tensor == graph_tensor
            assert tensor.inputs == graph_tensor.inputs
            assert tensor.outputs == graph_tensor.outputs


    # Check that tensors includes tensors not attached to nodes
    def test_tensors_includes_non_node_tensors(self):
        X = Constant("X", values=np.ones(shape=(64, 64), dtype=np.float32))
        graph = Graph(inputs=[], outputs=[X])
        tensor_map = graph.tensors()
        assert "X" in tensor_map
        assert tensor_map["X"] == X


    def test_tensors_check_duplicates(self):
        inputs = [Variable(name="x")]
        outputs = [Variable(name="x")] # Distinct tensors with the same name
        nodes = [
            Node(op="Add", name="Test", inputs=inputs, outputs=outputs),
        ]
        graph = Graph(nodes=nodes, inputs=inputs, outputs=outputs)

        with pytest.raises(OnnxGraphSurgeonException):
            graph.tensors(check_duplicates=True)


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
    toposort_linear_graph,
    toposort_multi_tier_output_graph,
    toposort_multi_tier_input_graph,
]

class TestToposort(object):
    @pytest.mark.parametrize("toposort_test_case", TOPOSORT_TEST_CASES)
    def test_topologically_sort(self, toposort_test_case):
        graph, expected_node_order = toposort_test_case()
        assert graph.nodes != expected_node_order
        graph.toposort()
        assert graph.nodes == expected_node_order


    @pytest.mark.parametrize("toposort_test_case", TOPOSORT_TEST_CASES)
    def test_toposort_nested(self, toposort_test_case):
        subgraph, expected_node_order = toposort_test_case()
        assert subgraph.nodes != expected_node_order

        # Wrap the graph within a subgraph
        inp = Variable("input")
        id_out = Variable("id_out")
        identity = Node(op="Identity", inputs=[inp], outputs=[id_out])

        # Make the subgraph take an input from the outer graph node
        # If toposort tries to take the node id, it'll fault.
        subgraph.nodes[0].inputs.append(id_out)

        out = Variable("output")
        nested = Node(op="Nested", inputs=[id_out], outputs=[out], attrs={"subgraph": subgraph})

        graph = Graph(nodes=[identity, nested], inputs=[inp], outputs=[out])
        graph.toposort(recurse_subgraphs=True)

        assert subgraph.nodes == expected_node_order


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


CLEANUP_TEST_CASES = [
    build_basic_graph(),
    build_two_layer_graph(),
    build_two_layer_graph_multiple_io(),
]


class TestCleanup(object):
    @pytest.mark.parametrize("graph", CLEANUP_TEST_CASES)
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


    def test_cleanup_remove_unused_node_outputs(self):
        graph, _  = toposort_linear_graph()
        graph.toposort()
        graph_output = graph.outputs[0]

        dummy = Variable("dummy")
        # Add unused tensor to a node in the middle of the graph.
        # Since it does not contribute to graph outputs, it should be removed.
        graph.nodes[1].outputs.append(dummy)

        graph.cleanup(remove_unused_node_outputs=True)
        assert dummy not in graph.nodes[1].outputs
        assert graph.outputs[0] == graph_output # Graoh outputs will never be removed


    def test_cleanup_graph_input_producers(self):
        graph, _ = toposort_linear_graph()
        tensor_map = graph.tensors()
        assert "x" in tensor_map

        graph.inputs = [tensor_map["intermediate0"]]

        graph.cleanup()
        cleaned_tensor_map = graph.tensors()
        assert "x" not in cleaned_tensor_map


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


    def test_cleanup_nested_graph(self, nested_graph):
        nested_node = nested_graph.nodes[1]
        nested_inp = nested_node.inputs[0]
        nested_out = nested_node.outputs[0]
        subgraph = nested_node.attrs["body"]

        assert "id_out" in nested_graph.tensors()
        nested_graph.cleanup(recurse_subgraphs=True)
        # Clean up should not remove a tensor whose only output node is a subgraph.
        assert "id_out" in nested_graph.tensors()

        # Clean up should not modify the nested nodes inputs or outputs
        assert nested_node.inputs == [nested_inp]
        assert nested_node.outputs == [nested_out]

        # Next we'll clean up the subgraph by recursing from the top-level
        assert subgraph.nodes
        subgraph.outputs.clear()
        nested_graph.cleanup(recurse_subgraphs=True)
        assert not subgraph.nodes


class TestCopy(object):
    def test_copy(self):
        def make_graph():
            graph, _ = toposort_multi_tier_output_graph()
            graph.outputs.pop()
            # Deep copy should work with empty tensors
            graph.nodes[0].inputs.append(Variable.empty())
            graph.nodes[0].outputs.append(Variable.empty())
            return graph

        graph = make_graph()
        new_graph = graph.copy()
        assert graph == new_graph

        # Running cleanup on the first graph should not affect the copy
        graph.cleanup()
        assert graph != new_graph
        assert new_graph == make_graph()


    def test_copy_with_subgraph(self, nested_graph):
        new_graph = nested_graph.copy()
        assert new_graph == nested_graph

        new_subgraph = new_graph.nodes[1].attrs["body"]
        new_subgraph.nodes[0].outputs.clear()
        new_subgraph.nodes[1].inputs.clear()

        subgraph = nested_graph.nodes[1].attrs["body"]
        assert subgraph.nodes[0].outputs
        assert subgraph.nodes[1].inputs

        new_graph.outputs.clear()
        new_graph.cleanup()

        assert nested_graph.outputs
        assert len(nested_graph.nodes) == 2
        assert len(subgraph.nodes) == 2


@pytest.fixture
def simple_foldable():
    # Graph:
    # c = (a + b)
    # output = input + c
    # Should fold to:
    # output = input + c
    weights = np.ones(shape=(1, 3), dtype=np.float32)

    graph = Graph()
    inp = Variable("input", shape=(1, 3), dtype=np.float32)
    c = graph.add(weights, weights, name="c")
    out = graph.add(inp, c)

    graph.inputs = [inp]
    graph.outputs = [out]
    yield graph


@pytest.fixture
def one_hop_foldable():
    # Graph:
    # c = (a + b)
    # e = (c + d)
    # output = input + e
    # Should fold to:
    # output = input + e
    weights = np.ones(shape=(1, 3), dtype=np.float32)

    graph = Graph()
    inp = Variable("input", shape=(1, 3), dtype=np.float32)
    c = graph.add(weights, weights, name="c")
    e = graph.add(c, weights, name="e")
    out = graph.add(inp, e)

    graph.inputs = [inp]
    graph.outputs = [out]
    yield graph


@pytest.fixture
def foldable_with_invalid_node():
    # Graph
    # c = (a + b)
    # e = fake(d)
    # f = (e + c)
    # out = inp + f
    #
    # c should be folded even though e is the output of an
    # invalid node.
    weights = np.ones(shape=(1, 3), dtype=np.float32)

    graph = Graph()
    inp = Variable("input", shape=(1, 3), dtype=np.float32)
    c = graph.add(weights, weights, name="c")
    e = graph.fake(weights, name="e")
    f = graph.add(e, c, name="f")
    out = graph.add(inp, f, name="output")

    graph.inputs = [inp]
    graph.outputs = [out]
    yield graph


class TestFoldConstants(object):
    @pytest.mark.parametrize("partitioning", [None, "basic", "recursive"])
    def test_basic(self, simple_foldable, partitioning):
        inp = simple_foldable.inputs[0]

        simple_foldable.fold_constants(partitioning=partitioning).cleanup()

        # Extra node should be removed
        assert len(simple_foldable.nodes) == 1
        assert simple_foldable.nodes[0].inputs[0] == inp
        assert simple_foldable.nodes[0].inputs[1].name == "c"

        # Value should be computed correctly
        assert np.all(simple_foldable.nodes[0].inputs[1].values == np.ones(shape=(1, 3), dtype=np.float32) * 2)


    def test_one_hop(self, one_hop_foldable):
        inp = one_hop_foldable.inputs[0]

        one_hop_foldable.fold_constants().cleanup()

        # Extra nodes should be removed
        assert len(one_hop_foldable.nodes) == 1
        assert one_hop_foldable.nodes[0].inputs[0] == inp
        assert one_hop_foldable.nodes[0].inputs[1].name == "e"

        # Value should be computed correctly
        assert np.all(one_hop_foldable.nodes[0].inputs[1].values == np.ones(shape=(1, 3), dtype=np.float32) * 3)


    def test_with_invalid_nodes(self, foldable_with_invalid_node):
        foldable_with_invalid_node.fold_constants(partitioning="recursive").cleanup()

        tensor_map = foldable_with_invalid_node.tensors()

        assert len(foldable_with_invalid_node.nodes) == 3
        assert foldable_with_invalid_node.nodes[0].op == "Fake"
        assert foldable_with_invalid_node.nodes[1].op == "Add"
        assert foldable_with_invalid_node.nodes[2].op == "Add"
        assert np.all(tensor_map["c"].values == (np.ones(shape=(1, 3), dtype=np.float32) * 2))


    def test_with_invalid_nodes_no_recursive(self, foldable_with_invalid_node):
        # No folding should take place without recursive partitioning
        original = foldable_with_invalid_node.copy()
        assert foldable_with_invalid_node.fold_constants() == original


    def test_no_foldable_constants(self):
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


    def test_const_node(self):
        graph = Graph()
        values = np.ones((1, 3, 3), dtype=np.int64)
        graph.outputs = [graph.constant(values=values)]

        assert isinstance(graph.outputs[0], Variable)

        graph.fold_constants().cleanup()

        assert isinstance(graph.outputs[0], Constant)
        assert np.all(graph.outputs[0].values == values)
        assert not graph.nodes


    def test_shape_of_constant_tensor(self):
        graph = Graph()
        values = np.ones((1, 3, 3), dtype=np.int64)
        const = Constant("const", values=values)
        graph.outputs = [graph.shape(const)]

        graph.fold_constants().cleanup()

        assert not graph.nodes
        assert isinstance(graph.outputs[0], Constant)
        assert np.all(graph.outputs[0].values == (1, 3, 3))


    def test_shape_of_constant_node(self):
        graph = Graph()
        values = np.ones((1, 3, 3), dtype=np.int64)
        const = graph.constant(values=values)
        graph.outputs = [graph.shape(const)]

        graph.fold_constants().cleanup()

        assert not graph.nodes
        assert isinstance(graph.outputs[0], Constant)
        assert np.all(graph.outputs[0].values == (1, 3, 3))


    # Cannot fold shape nodes if they have dynamically shaped inputs.
    def test_shape_of_variable_tensor_dynamic_shape(self):
        graph = Graph()
        var = Variable("var", dtype=np.float32, shape=("", -1, 0, 4))
        graph.outputs = [graph.shape(var)]

        graph.fold_constants().cleanup()

        assert len(graph.nodes) == 1
        assert graph.nodes[0].op == "Shape"
        assert isinstance(graph.outputs[0], Variable)


    def test_shape_of_variable_tensor_static_shape(self):
        graph = Graph()
        var = Variable("var", dtype=np.float32, shape=(1, 3, 4))
        graph.inputs = [var]
        graph.outputs = [graph.shape(var)]

        graph.fold_constants().cleanup()

        assert not graph.nodes
        assert isinstance(graph.outputs[0], Constant)
        assert np.all(graph.outputs[0].values == (1, 3, 4))


    def test_shape_of_variable_tensor_multiple_shapes(self):
        graph = Graph()
        var = Variable("var", dtype=np.float32, shape=(1, 3, 4))
        var2 = Variable("var2", dtype=np.float32, shape=tuple()) # Scalar
        graph.inputs = [var, var2]
        graph.outputs = [graph.shape(var), graph.identity(var), graph.shape(var2)]

        graph.fold_constants().cleanup()

        assert len(graph.nodes) == 1
        assert graph.nodes[0].op == "Identity"
        assert isinstance(graph.outputs[0], Constant)
        assert np.all(graph.outputs[0].values == (1, 3, 4))
        assert isinstance(graph.outputs[2], Constant)
        assert np.all(graph.outputs[2].values == tuple())


    def test_shape_of_variable_tensor_static_shape_no_fold(self):
        graph = Graph()
        var = Variable("var", dtype=np.float32, shape=(1, 3, 4))
        graph.inputs = [var]
        graph.outputs = [graph.shape(var)]

        graph.fold_constants(fold_shapes=False).cleanup()

        assert len(graph.nodes) == 1
        assert graph.nodes[0].op == "Shape"
        assert isinstance(graph.outputs[0], Variable)


class TestIO(object):
    def test_io_cannot_be_sync_list_on_init(self):
        inp = Variable("input0", shape=(1, 3), dtype=np.float32)
        out = Variable("input1", shape=(1, 3), dtype=np.float32)

        node = Node("Add", inputs=[inp], outputs=[out])
        assert isinstance(node.inputs, SynchronizedList)
        assert isinstance(node.outputs, SynchronizedList)

        graph = Graph(nodes=[node], inputs=node.inputs, outputs=node.outputs)
        assert not isinstance(graph.inputs, SynchronizedList)
        assert not isinstance(graph.outputs, SynchronizedList)


    def test_io_cannot_be_sync_list_on_assign(self):
        inp = Variable("input0", shape=(1, 3), dtype=np.float32)
        out = Variable("input1", shape=(1, 3), dtype=np.float32)

        node = Node("Add", inputs=[inp], outputs=[out])
        assert isinstance(node.inputs, SynchronizedList)
        assert isinstance(node.outputs, SynchronizedList)

        graph = Graph(nodes=[node], inputs=[], outputs=[])
        graph.inputs = node.inputs
        graph.outputs = node.outputs

        assert not isinstance(graph.inputs, SynchronizedList)
        assert not isinstance(graph.outputs, SynchronizedList)
