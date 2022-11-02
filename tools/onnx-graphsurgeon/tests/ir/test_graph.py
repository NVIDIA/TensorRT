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

import copy

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.ir.tensor import Constant, LazyValues, Variable
from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.util import misc
from onnx_graphsurgeon.util.exception import OnnxGraphSurgeonException
from onnx_graphsurgeon.util.misc import SynchronizedList
from onnx_models import const_foldable, shape_cast_elision

G_LOGGER.severity = G_LOGGER.ULTRA_VERBOSE


@Graph.register()
def shape(self, inp):
    return self.layer(op="Shape", inputs=[inp], outputs=["shape_out"])[0]


@Graph.register()
def cast(self, inp, to):
    return self.layer(op="Cast", inputs=[inp], outputs=["cast_out"], attrs={"to": to})[0]


@Graph.register()
def constant(self, values):
    return self.layer(op="Constant", inputs=[], outputs=["constant_out"], attrs={"value": Constant("values", values)})[
        0
    ]


@Graph.register()
def identity(self, inp):
    out = self.layer(op="Identity", inputs=[inp], outputs=["identity_out"])[0]
    out.dtype = inp.dtype
    return out


@Graph.register()
def relu(self, inp):
    out = self.layer(op="Relu", inputs=[inp], outputs=["relu_out"])[0]
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


@gs.Graph.register()
def gather(self, data, indices):
    return self.layer(op="Gather", inputs=[data, indices], outputs=["gather_out"])[0]


@gs.Graph.register()
def slice(self, data, starts=None, ends=None, axes=None, steps=None):
    inputs = []
    for inp in [data, starts, ends, axes, steps]:
        if inp is None:
            break
        inputs.append(inp)

    return self.layer(op="Slice", inputs=inputs, outputs=["slice_out"])[0]


@gs.Graph.register()
def nested(self, inp, graph):
    return self.layer(op="Nested", inputs=[inp], outputs=["nested_out"], attrs={"body": graph})[0]


@gs.Graph.register()
def if_op(self, cond, then_graph, else_graph):
    return self.layer(
        op="If", inputs=[cond], outputs=["if_out"], attrs={"then_branch": then_graph, "else_branch": else_graph}
    )[0]


@gs.Graph.register()
def tile(self, inp, repeats):
    out = self.layer(op="Tile", inputs=[inp, repeats], outputs=["tile_out"])[0]
    out.dtype = inp.dtype
    return out


@gs.Graph.register()
def dequantize_linear(self, inp, scale, zero_point, axis=1):
    out = self.layer(
        op="DequantizeLinear", inputs=[inp, scale, zero_point], outputs=["dequantize_linear_out"], attrs={"axis": axis}
    )[0]
    out.dtype = np.float32
    return out


@gs.Graph.register()
def quantize_linear(self, inp, out_scale, out_zero_point, axis=1):
    out = self.layer(
        op="QuantizeLinear",
        inputs=[inp, out_scale, out_zero_point],
        outputs=["quantize_linear_out"],
        attrs={"axis": axis},
    )[0]
    out.dtype = np.int8
    return out


# Generates a graph where an outer node has no outputs except
# within the subgraph. ONNX-GS should recognize that the node
# is being used, and should not remove it during cleanup().
def make_nested_graph():
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

    subgraph = Graph(nodes=[subgraph_identity0, subgraph_identity1], inputs=subgraph_inputs, outputs=subgraph_outputs)

    nested_out = Variable("nested_out")
    nested_node = Node(op="Nested", attrs={"body": subgraph}, inputs=[inp], outputs=[nested_out])

    return Graph(nodes=[identity, nested_node], inputs=[inp], outputs=[nested_out])


@pytest.fixture
def nested_graph():
    yield make_nested_graph()


class TestBasic(object):
    def test_generate_name(self):
        graph = Graph()
        names = set()
        num_names = 100
        # This function should not return the same name more than once
        for idx in range(num_names):
            names.add(graph._generate_name("name"))
        assert len(names) == 100

    def test_equal(self, nested_graph):
        assert nested_graph == nested_graph

    def test_equal_inputs_unequal(self):
        g0 = make_nested_graph()
        g1 = make_nested_graph()

        g0.inputs.append(Variable("test"))

        assert not (g0 == g1)

    def test_equal_outputs_unequal(self):
        g0 = make_nested_graph()
        g1 = make_nested_graph()

        g0.outputs.append(Variable("test"))

        assert not (g0 == g1)

    def test_equal_nested_unequal(self):
        g0 = make_nested_graph()
        g1 = make_nested_graph()

        # Changing the nested subgraph should make the graphs unequal
        g0.nodes[1].inputs[0].name = "subgraph_inp_modified"

        assert not (g0 == g1)


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
        x1 = (1,)
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
        outputs = [Variable(name="x")]  # Distinct tensors with the same name
        nodes = [
            Node(op="Add", name="Test", inputs=inputs, outputs=outputs),
        ]
        graph = Graph(nodes=nodes, inputs=inputs, outputs=outputs)

        with pytest.raises(OnnxGraphSurgeonException):
            graph.tensors(check_duplicates=True)

    def test_tensors_with_duplicates_check_disabled(self):
        inputs = [Variable(name="x")]
        outputs = [Variable(name="x")]  # Distinct tensors with the same name
        nodes = [
            Node(op="Add", name="Test", inputs=inputs, outputs=outputs),
        ]
        graph = Graph(nodes=nodes, inputs=inputs, outputs=outputs)

        # This should *not* throw
        graph.tensors(check_duplicates=False)


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


# Graph structure:
# x0
# |
# Add
# |
# x1
# |
# Relu
# |
# x2
#
# If:
#   Then:
#          x1  x2
#           |  |
#           Add
#            |
#           res
#   Else:
#          x1  x2
#           |  |
#           Add
#            |
#           res
#  |
# out
#
# In this graph, the subgraph of If implicitly depends on x1/x2 from the outer graph, so the parent If
# node must come after the outer Add/ReLU nodes.
# If we fail to consider such implicit inputs, the If will remain the first node.
def toposort_implicit_subgraph_inputs_graph():
    def make_var(name):
        return Variable(name, shape=(1, 1), dtype=np.float32)

    # Main graph
    inputs = [make_var("x0")]
    const = Constant(name="const", values=np.array([[1.5]], dtype=np.float32))
    cond = Constant(name="cond", values=np.array([True]))
    x1, x2 = [make_var("x1"), make_var("x2")]
    outputs = [make_var("out")]

    # Subgraphs for If
    subgraph_outputs = [make_var("res")]
    subgraph_nodes = [Node(op="Add", name="SubgraphTest0", inputs=[x1, x2], outputs=subgraph_outputs)]
    subgraph = Graph(nodes=subgraph_nodes, outputs=subgraph_outputs)

    nodes = [
        Node(
            op="If",
            name="Test2",
            inputs=[cond],
            outputs=outputs,
            attrs={"then_branch": subgraph, "else_branch": subgraph},
        ),
        Node(op="Relu", name="Test1", inputs=[x1], outputs=[x2]),
        Node(op="Add", name="Test0", inputs=inputs + [const], outputs=[x1]),
    ]
    expected_node_order = [nodes[2], nodes[1], nodes[0]]
    return Graph(nodes=nodes, inputs=inputs, outputs=outputs), expected_node_order


TOPOSORT_TEST_CASES = [
    toposort_linear_graph,
    toposort_multi_tier_output_graph,
    toposort_multi_tier_input_graph,
    toposort_implicit_subgraph_inputs_graph,
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

    def test_multi_tier(self):
        graph, _ = toposort_multi_tier_output_graph()
        tensor = graph.outputs.pop()
        unused_node = tensor.inputs[0]
        graph.cleanup()  # Should remove just the Test2 node as out1 is still an output.
        assert unused_node not in graph.nodes
        assert len(graph.nodes) == 2
        assert len(graph.outputs) == 2

        tensor_map = graph.tensors()
        assert tensor.name not in tensor_map

    def test_remove_unused_node_outputs(self):
        graph, _ = toposort_linear_graph()
        graph.toposort()
        graph_output = graph.outputs[0]

        dummy = Variable("dummy")
        # Add unused tensor to a node in the middle of the graph.
        # Since it does not contribute to graph outputs, it should be removed.
        graph.nodes[1].outputs.append(dummy)

        graph.cleanup(remove_unused_node_outputs=True)
        assert dummy not in graph.nodes[1].outputs
        assert graph.outputs[0] == graph_output  # Graoh outputs will never be removed

    def test_graph_input_producers(self):
        graph, _ = toposort_linear_graph()
        tensor_map = graph.tensors()
        assert "x" in tensor_map

        graph.inputs = [tensor_map["intermediate0"]]

        graph.cleanup()
        cleaned_tensor_map = graph.tensors()
        assert "x" not in cleaned_tensor_map

    @pytest.mark.parametrize("remove_unused_graph_inputs", [True, False])
    def test_independent_path(self, remove_unused_graph_inputs):
        graph, _ = toposort_linear_graph()
        # Build out a path totally unrelated to rest of the graph
        indep0 = Variable(name="indep0")
        indep1 = Variable(name="indep1")
        node = Node(op="IndepTest", inputs=[indep0], outputs=[indep1])
        graph.nodes.append(node)
        graph.inputs.append(indep0)
        graph.cleanup(remove_unused_graph_inputs=remove_unused_graph_inputs)
        assert indep0 not in graph.inputs or not remove_unused_graph_inputs
        assert node not in graph.nodes or not remove_unused_graph_inputs

        tensor_map = graph.tensors()
        assert indep0.name not in tensor_map or not remove_unused_graph_inputs
        assert indep1.name not in tensor_map or not remove_unused_graph_inputs

    def test_nested_graph(self, nested_graph):
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

    def test_node_used_only_in_nested_graph(self):
        X = Variable("X", dtype=np.float32, shape=(1,))
        Y = Variable("Y", dtype=np.float32, shape=(1,))
        graph = Graph(inputs=[X, Y])

        X_p = graph.identity(X)  # X_p is only used by the subgraph, not in the outer graph.

        subgraph_inp = Variable("subgraph_input", dtype=np.float32, shape=(1,))
        subgraph = Graph(inputs=[subgraph_inp])
        subgraph.outputs = [subgraph.add(subgraph_inp, X_p)]

        graph.outputs = [graph.nested(Y, subgraph)]

        graph.cleanup(remove_unused_graph_inputs=True)

        assert graph.nodes[0].op == "Identity"
        assert graph.nodes[0].inputs == [X]

    def test_input_is_output(self):
        graph = Graph()

        A = Variable("A", dtype=np.float32, shape=(1, 1))
        B = Variable("B", dtype=np.float32, shape=(1, 1))

        C = graph.add(A, B)

        graph.inputs = [A, B]
        graph.outputs = [C, B, A]  # Out of order w/ respect to Add node inputs

        # Graph should remain unchanged after cleanup, including I/O tensors.
        graph.cleanup()

        assert graph.inputs == [A, B]
        assert graph.outputs == [C, B, A]
        assert len(graph.nodes) == 1
        assert graph.nodes[0].inputs == [A, B]
        assert graph.nodes[0].outputs == [C]


class TestCopy(object):
    def test_basic(self):
        graph = Graph(
            nodes=[Node(op="Test")],
            inputs=[Variable("test")],
            outputs=[Variable("test")],
            name="test-name",
            doc_string="test-docstring",
            import_domains=["fake-import-domain"],
            opset=-1,
        )
        new_graph = graph.copy()

        assert new_graph == graph
        assert new_graph.nodes == graph.nodes
        assert new_graph.inputs == graph.inputs
        assert new_graph.outputs == graph.outputs
        assert new_graph.name == graph.name
        assert new_graph.doc_string == graph.doc_string
        assert new_graph.import_domains == graph.import_domains
        assert new_graph.opset == graph.opset

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

        id_out = new_subgraph.nodes[0].inputs[0]
        assert id_out.name == "id_out"
        assert len(id_out.inputs) == 1
        assert id_out.inputs[0].op == "Identity"
        assert id_out.inputs[0].inputs[0].name == "input"

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

    # If the subgraph has a tensor with the same name as the outer graph,
    # the subgraph copy should include a copy of the subgraph tensor, not the outer
    # graph tensor.
    def test_copy_with_subgraph_dup_tensors(self):
        inp = Variable("input", dtype=np.float32, shape=(4, 5))
        graph = Graph(inputs=[inp])

        # We'll use shape to distinguish inner/outer tensor
        subgraph_inp = Variable("input", dtype=np.float32, shape=(1, 2))
        subgraph = Graph(inputs=[subgraph_inp])

        graph.outputs = [graph.nested(inp, subgraph)]

        graph_copy = graph.copy()
        assert graph_copy.nodes[0].attrs["body"].inputs[0].shape == (1, 2)

    def test_copy_with_subgraph_dup_const_tensors(self):
        inp = Constant("input", values=np.ones(dtype=np.float32, shape=(4, 5)))
        graph = Graph()

        # We'll use shape to distinguish inner/outer tensor
        subgraph_inp = Constant("input", values=np.ones(dtype=np.float32, shape=(1, 2)))
        subgraph = Graph()
        subgraph.outputs = [subgraph.identity(subgraph_inp)]

        graph.outputs = [graph.nested(inp, subgraph)]

        graph_copy = graph.copy()
        assert graph_copy.nodes[0].attrs["body"].nodes[0].inputs[0].shape == (1, 2)


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
    out = graph.add(inp, c, name="out")

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

        simple_foldable.fold_constants(partitioning=partitioning).cleanup(remove_unused_graph_inputs=True)

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

        nodes = [Node("Add", inputs=[inp0, inp1], outputs=[out])]

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
        var = Variable("var", dtype=np.float32, shape=("", -1, 0, 4))
        graph = Graph(inputs=[var])
        graph.outputs = [graph.shape(var)]

        graph.fold_constants().cleanup()

        assert len(graph.nodes) == 1
        assert graph.nodes[0].op == "Shape"
        assert isinstance(graph.outputs[0], Variable)

    def test_shape_of_variable_tensor_static_shape(self):
        var = Variable("var", dtype=np.float32, shape=(1, 3, 4))
        graph = Graph(inputs=[var])
        graph.inputs = [var]
        graph.outputs = [graph.shape(var)]

        graph.fold_constants().cleanup()

        assert not graph.nodes
        assert isinstance(graph.outputs[0], Constant)
        assert np.all(graph.outputs[0].values == (1, 3, 4))

    def test_shape_of_variable_tensor_multiple_shapes(self):
        graph = Graph()
        var = Variable("var", dtype=np.float32, shape=(1, 3, 4))
        var2 = Variable("var2", dtype=np.float32, shape=tuple())  # Scalar
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

    # Constant folding should not cause constant tensors in the model to be loaded.
    def test_no_load_constants(self):
        graph = gs.import_onnx(const_foldable().load())

        new_graph = graph.fold_constants()

        def check_no_const_loaded(graph):
            num_lazy_constants = 0
            for tensor in graph.tensors().values():
                if isinstance(tensor, Constant) and isinstance(tensor._values, LazyValues):
                    num_lazy_constants += 1
            assert num_lazy_constants == 3  # Graph starts with 3 constants - none should be loaded.

        check_no_const_loaded(graph)
        check_no_const_loaded(new_graph)

    @pytest.mark.parametrize(
        "shape, indices",
        [
            (("batch", 3, "height", "width"), 1),  # Scalar indices case
            (None, 1),  # Shape not inferered case
            (("batch", 3, "height", "width"), [1]),
            (("batch", 3, "height", 224), [1, 3]),
            (("batch", 3, 224, 224), [1, 2, 3]),
        ],
    )
    def test_shape_gather(self, shape, indices):
        indices = np.array(indices)

        inp = Variable("input", dtype=np.float32, shape=shape)
        graph = Graph(inputs=[inp])

        inp_shape = graph.shape(inp)
        shape_part = graph.gather(inp_shape, indices=indices)
        graph.outputs = [
            graph.add(shape_part, shape_part),
            graph.gather(inp_shape, indices=[0]),
            graph.gather(inp_shape, indices=np.array(0)),
        ]

        graph.fold_constants()

        if shape is not None:
            assert isinstance(graph.outputs[0], Constant)
            expected_shape = np.array(shape)[indices].astype(np.int64) * 2
            assert np.all(graph.outputs[0].values == expected_shape)
        else:
            assert isinstance(graph.outputs[0], Variable)

        assert isinstance(graph.outputs[1], Variable)
        assert isinstance(graph.outputs[2], Variable)

    @pytest.mark.parametrize(
        "shape, starts, ends, axes, steps, expected",
        [
            (("batch", 3, "height", "width"), 1, 2, 0, 1, [3]),  # Scalar starts/ends case
            (("batch", 3, "height", "width"), [1], [2], [0], [1], [3]),
            (("batch", 3, 5, "width"), [1], [-1], [0], [1], [3, 5]),  # Negative ends case
            (("batch", 3, 5, 7), [1], [2000], [0], [1], [3, 5, 7]),  # Past end, ends case
            (("batch", 3, 5, 7), [-2], [4], [0], [1], [5, 7]),  # Negative starts case
            (("batch", 3, 5, 7), [-2], [4], [1], [1], None),  # Non-zero axes case
            (("batch", 3, 5, "width"), [-2], [4], [1], [1], None),  # Dynamic case
            (("batch", 3, 5, 7), [1], [4], [0], [2], [3, 7]),  # Non-one steps case
            (("batch", 3, 5, 7), [4], [0], [0], [-1], [7, 5, 3]),  # Negative steps case
        ],
    )
    def test_shape_slice(self, shape, starts, ends, axes, steps, expected):
        inp = Variable("input", dtype=np.float32, shape=shape)
        graph = Graph(inputs=[inp])

        inp_shape = graph.shape(inp)
        graph.outputs = [
            graph.slice(inp_shape, np.array(starts), np.array(ends), axes=np.array(axes), steps=np.array(steps))
        ]

        graph.fold_constants()

        if expected:
            assert isinstance(graph.outputs[0], Constant)
            assert np.all(graph.outputs[0].values == expected)
        else:
            assert isinstance(graph.outputs[0], Variable)

    # In the single input case, we should derive starts/ends/axes/steps from the attributes.
    def test_shape_slice_single_input(self):
        inp = Variable("input", dtype=np.int64, shape=(5, 6, 3, 2))
        graph = Graph(inputs=[inp])

        inp_shape = graph.shape(inp)
        graph.outputs = [graph.slice(inp_shape)]

        slice_node = graph.outputs[0].inputs[0]

        slice_node.attrs = {
            "axes": [0],
            "starts": [1],
            "ends": [3],
            "steps": [2],
        }

        graph.fold_constants()

        assert isinstance(graph.outputs[0], Constant)
        assert np.all(graph.outputs[0].values == inp.shape[1:3:2])

    def test_with_variable_conditional(self):
        cond = gs.Variable("cond", dtype=np.bool, shape=(1,))

        X = gs.Variable("X", dtype=np.float32, shape=(1,))
        Y = gs.Constant("Y", values=np.ones((1,), dtype=np.float32))
        graph = Graph(inputs=[X, cond])

        then_graph = Graph(name="Then")
        then_graph.outputs = [then_graph.add(Y, Y)]

        else_graph = Graph(name="Else")
        else_graph.outputs = [else_graph.add(X, else_graph.add(Y, Y))]

        graph.outputs = [graph.if_op(cond, then_graph, else_graph)]

        graph.fold_constants()
        graph.cleanup()

        assert len(then_graph.nodes) == 0
        assert np.all(then_graph.outputs[0].values == (Y.values * 2))

        assert len(else_graph.nodes) == 1
        assert isinstance(else_graph.nodes[0].inputs[1], Constant)
        assert np.all(else_graph.nodes[0].inputs[1].values == (Y.values * 2))

    @pytest.mark.parametrize("cond_value", [True, False])
    @pytest.mark.parametrize("flatten", [True, False])
    def test_flatten_static_conditional(self, flatten, cond_value):
        cond = gs.Constant("cond", values=np.array([cond_value], dtype=np.bool))

        X = gs.Variable("X", dtype=np.float32, shape=(1,))
        Y = gs.Variable("Y", dtype=np.float32, shape=(1,))
        graph = Graph(inputs=[X, cond])

        then_graph = Graph(name="Then")
        then_graph.outputs = [then_graph.relu(then_graph.add(Y, Y))]

        else_graph = Graph(name="Else")
        else_graph.outputs = [else_graph.add(X, else_graph.add(Y, Y))]

        if_out = graph.if_op(cond, then_graph, else_graph)
        graph.outputs = [if_out]

        graph.fold_constants(flatten_subgraphs=flatten)
        graph.cleanup()

        if flatten:
            assert len(graph.nodes) == 2
            assert graph.nodes[0].op == "Add"
            assert graph.nodes[1].op == "Relu" if cond_value else "Add"

            subgraph = then_graph if cond_value else else_graph
            # Make sure subgraph intermediate tensors are renamed
            assert graph.nodes[0].outputs[0].name == "add_out_0_subg_0_{:}".format(subgraph.name)
            assert graph.outputs[0].inputs[0] == subgraph.nodes[-1]
            assert subgraph.nodes[-1] == graph.nodes[-1]
        else:
            assert len(graph.nodes) == 1
            assert len(graph.nodes) == 1
            assert graph.nodes[0].op == "If"
            assert graph.outputs[0].inputs[0] == graph.nodes[-1]
        assert graph.outputs == [if_out]

    def test_const_inp_but_non_foldable_nested_graph(self):
        cond = gs.Constant("cond", values=np.array(True))
        X = gs.Variable("X", dtype=np.float32, shape=(1,))

        graph = Graph(inputs=[X])

        then_graph = Graph(name="Then")
        then_graph.outputs = [then_graph.add(X, X)]

        else_graph = Graph(name="Else")
        else_graph.outputs = [else_graph.add(X, else_graph.add(X, X))]

        # Even though if_op looks foldable because it has all constant inputs,
        # it's not, since its subgraphs depend on variables in the outer scope.
        graph.outputs = [graph.if_op(cond, then_graph, else_graph)]

        # This should not raise because the `If` node should be excluded from
        # constant folding.
        graph.fold_constants(error_ok=False, flatten_subgraphs=False).cleanup()

        assert graph.nodes[0].op == "If"
        assert len(then_graph.nodes) == 1
        assert len(else_graph.nodes) == 2

    def test_cast_elision(self):
        graph = gs.import_onnx(shape_cast_elision().load())
        graph.fold_constants().cleanup()
        assert not any(node.op == "Cast" for node in graph.nodes)

    def test_cast_elision_int64(self):
        X = gs.Variable("X", dtype=np.int64, shape=(1,))
        graph = Graph(inputs=[X])
        casted_x = graph.cast(X, to=onnx.TensorProto.DataType.FLOAT)
        add_out = graph.add(casted_x, casted_x)
        graph.outputs = [graph.cast(add_out, to=onnx.TensorProto.DataType.INT64)]

        graph.fold_constants().cleanup()
        assert graph.nodes[0].op == "Add"

    # Make sure we're lowering constant nodes before running cast elision
    def test_cast_elision_with_constant_node(self):
        inp = gs.Variable("inp", dtype=np.int64, shape=(1,))
        graph = Graph(inputs=[inp])

        casted_inp = graph.cast(inp, to=onnx.TensorProto.DataType.FLOAT)
        add_out = graph.add(casted_inp, graph.constant(np.array([2], dtype=np.float32)))

        casted_out = graph.cast(add_out, to=onnx.TensorProto.DataType.INT64)
        casted_out.dtype = np.int64

        graph.outputs = [casted_out]

        graph.fold_constants().cleanup()
        assert [node.op for node in graph.nodes] == ["Add"]

        add_const_inp = graph.nodes[0].inputs[1]
        assert isinstance(add_const_inp, Constant)
        assert add_const_inp.dtype == np.int64  # Should have been casted to match dtype of other inputs.

    # For a graph like:
    #
    #     inp
    #      |
    #    Cast
    #      |
    #    Add
    #     |
    #    Cast
    #     |
    #    out
    #
    # 1. We cannot remove the initial `Cast` if it is used outside the `Add` node
    # 2. We cannot perform cast elision at all if the original output of the `Add` node is
    #    used outside the subsequent `Cast` node.
    #
    @pytest.mark.parametrize("use_as_graph_output", [True, False], ids=["graph", ""])
    @pytest.mark.parametrize("use_in_other_node", [True, False], ids=["node", ""])
    # Whether to apply the effects of the first two parameters to the input `Cast` node or to the `Add` node.
    @pytest.mark.parametrize("apply_to_input_cast", [True, False], ids=["input", "output"])
    def test_cast_elision_multi_use_cast(self, use_as_graph_output, use_in_other_node, apply_to_input_cast):
        X = gs.Variable("X", dtype=np.int32, shape=(1,))
        graph = Graph(inputs=[X])
        casted_x = graph.cast(X, to=onnx.TensorProto.DataType.FLOAT)
        add_out = graph.add(casted_x, casted_x)
        uncasted_x = graph.cast(add_out, to=onnx.TensorProto.DataType.INT32)

        graph.outputs = [uncasted_x]

        mutli_use_tensor = casted_x if apply_to_input_cast else add_out
        if use_in_other_node:
            graph.outputs.append(graph.identity(mutli_use_tensor))

        if use_as_graph_output:
            graph.outputs.append(mutli_use_tensor)

        print(graph)
        graph.fold_constants().cleanup()
        ops = [node.op for node in graph.nodes]
        if use_as_graph_output or use_in_other_node:
            if apply_to_input_cast:
                assert graph.nodes[1].inputs[0] == X
                assert graph.nodes[1].outputs[0] == uncasted_x
                assert ops == ["Cast", "Add"] + (["Identity"] if use_in_other_node else [])
            else:
                assert ops == ["Cast", "Add", "Cast"] + (["Identity"] if use_in_other_node else [])
        else:
            assert ops == ["Add"]

    @pytest.mark.parametrize(
        # If layer1_num_bytes is larger than layer0_num_bytes, then it must be a multiple.
        "size_threshold, layer0_num_bytes, layer0_should_fold, layer1_num_bytes, layer1_should_fold",
        [
            # No size threshold - everything should fold.
            (
                None,
                2,
                True,
                4,
                True,
            ),
            # Monotonically increasing but under size threshold - everything should fold.
            (
                8,
                2,
                True,
                4,
                True,
            ),
            # Increasing then decreasing, but under size threshold - everything should fold.
            (
                8,
                2,
                True,
                1,
                True,
            ),
            # All tensors over size threshold - nothing should fold.
            (
                1,
                2,
                False,
                4,
                False,
            ),
            # Second tensor over size threshold - only first tensor should fold.
            (
                3,
                2,
                True,
                4,
                False,
            ),
            # First tensor over size threshold - second tensor should still fold.
            (
                3,
                4,
                False,
                2,
                True,
            ),
        ],
    )
    @pytest.mark.parametrize("push_into_subgraph", [True, False], ids=["subgraph", ""])
    def test_folding_size_threshold(
        self,
        size_threshold,
        layer0_num_bytes,
        layer0_should_fold,
        layer1_num_bytes,
        layer1_should_fold,
        push_into_subgraph,
    ):
        graph = Graph()

        shape = (1,)

        layer0_repeats = layer0_num_bytes // misc.volume(shape)
        layer0 = graph.tile(np.ones(shape, dtype=np.int8), repeats=[layer0_repeats])
        layer0.inputs[0].name = "Layer0"

        if layer1_num_bytes > layer0_num_bytes:
            layer1_repeats = layer1_num_bytes // layer0_num_bytes
            layer1 = graph.tile(layer0, repeats=[layer1_repeats])
        else:
            layer1 = graph.slice(layer0, starts=[0], ends=[layer1_num_bytes])
        layer1.inputs[0].name = "Layer1"

        graph.outputs = [layer1]

        # Make sure size_threshold option is propagated into subgraphs.
        if push_into_subgraph:
            cond = gs.Variable("cond", dtype=np.bool, shape=tuple())
            outer_graph = Graph(inputs=[cond])
            outer_graph.if_op(cond, then_graph=graph, else_graph=graph)

            outer_graph.fold_constants(size_threshold=size_threshold)
        else:
            graph.fold_constants(size_threshold=size_threshold)

        # When a tensor is folded, it is disconnected from its producer nodes
        assert len(graph.nodes[0].outputs) == (0 if layer0_should_fold else 1)
        assert len(graph.nodes[1].outputs) == (0 if layer1_should_fold else 1)

    @pytest.mark.parametrize("op", ["Q", "DQ"])
    @pytest.mark.parametrize("add_intermediate_layer", [True, False])
    def test_no_fold_qdq(self, op, add_intermediate_layer):
        dtype = np.float32 if op == "Q" else np.int8
        inp = gs.Constant("input", np.ones(shape=(1, 3, 5, 5), dtype=dtype))
        graph = Graph(inputs=[inp], opset=13)

        if add_intermediate_layer:
            inp = graph.identity(inp)

        qdq_func = graph.quantize_linear if op == "Q" else graph.dequantize_linear
        graph.outputs = [qdq_func(inp, 1.2, np.array(0, dtype=np.int8))]  # Arbitrary scale and zero-point

        graph.fold_constants().cleanup()
        assert len(graph.nodes) == 1
        assert graph.nodes[0].op == "QuantizeLinear" if op == "Q" else "DequantizeLinear"

    @pytest.mark.parametrize(
        "should_exclude_node_func,expected_node_names",
        [
            (
                lambda node: True,
                [
                    "onnx_graphsurgeon_node_1",
                    "onnx_graphsurgeon_node_3",
                    "onnx_graphsurgeon_node_5",
                    "onnx_graphsurgeon_node_7",
                ],
            ),
            (
                lambda node: node.name == "onnx_graphsurgeon_node_5",
                ["onnx_graphsurgeon_node_5", "onnx_graphsurgeon_node_7"],
            ),
            (
                lambda node: node.op == "Add",
                [
                    "onnx_graphsurgeon_node_1",
                    "onnx_graphsurgeon_node_3",
                    "onnx_graphsurgeon_node_5",
                    "onnx_graphsurgeon_node_7",
                ],
            ),
            (
                lambda node: node.op == "Relu",
                [
                    "onnx_graphsurgeon_node_3",
                    "onnx_graphsurgeon_node_5",
                    "onnx_graphsurgeon_node_7",
                ],
            ),
        ],
    )
    def test_custom_should_exclude_node(self, should_exclude_node_func, expected_node_names):
        inp = gs.Constant("input", np.ones(shape=(1, 3, 5, 5), dtype=np.float32))
        graph = Graph(inputs=[inp])

        add_0 = graph.add(inp, inp)  # onnx_graphsurgeon_node_1 -> add_out_0
        relu_0 = graph.relu(add_0)  # onnx_graphsurgeon_node_3 -> relu_out_2
        add_1 = graph.add(relu_0, relu_0)  # onnx_graphsurgeon_node_5 -> add_out_4
        relu_1 = graph.relu(add_1)  # onnx_graphsurgeon_node_7 -> relu_out_6

        graph.outputs = [relu_1]

        graph.fold_constants(should_exclude_node=should_exclude_node_func).cleanup()
        assert [node.name for node in graph.nodes] == expected_node_names


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
