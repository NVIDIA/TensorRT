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

from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.ir.tensor import Tensor, Constant, Variable
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.importers.onnx_importer import OnnxImporter

G_LOGGER.severity = G_LOGGER.ULTRA_VERBOSE

from collections import OrderedDict
import onnx.numpy_helper
from typing import List
import numpy as np
import onnx
import os

TEST_ROOT = os.path.realpath(os.path.dirname(__file__))


class Model(object):
    def __init__(
        self,
        path: str,
        inputs: List[Tensor],
        outputs: List[Tensor],
        nodes: List[Node],
        opset: int = None,
    ):
        self.path = path
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = nodes
        self.opset = opset

    def load(self):
        return onnx.load(self.path)

    def assert_equal(self, graph: Graph):
        assert graph.inputs == self.inputs
        G_LOGGER.debug("Graph inputs matched")

        # Break down fields to make debugging failures easier.
        for actual, expected in zip(graph.nodes, self.nodes):

            def check_tensor_io(actensor, extensor):
                def check_list(aclist, exlist):
                    G_LOGGER.debug("Actual node list: {:}\n\nExpected node list: {:}".format(aclist, exlist))
                    assert len(aclist) == len(exlist)
                    for acnode, exnode in zip(aclist, exlist):
                        assert acnode == exnode

                G_LOGGER.debug("Checking tensor: {:} inputs".format(actensor.name))
                check_list(actensor.inputs, extensor.inputs)
                G_LOGGER.debug("Checking tensor: {:} outputs".format(actensor.name))
                check_list(actensor.outputs, extensor.outputs)

            G_LOGGER.debug("Actual Node: {:}\n\nExpected Node: {:}".format(actual, expected))
            assert actual.op == expected.op
            assert actual.inputs == expected.inputs
            # Check I/O of input tensors
            for acinp, exinp in zip(actual.inputs, expected.inputs):
                check_tensor_io(acinp, exinp)

            assert actual.outputs == expected.outputs
            # Check I/O of output tensors
            for acout, exout in zip(actual.outputs, expected.outputs):
                check_tensor_io(acout, exout)

            assert actual.name == expected.name
            assert len(actual.attrs) == len(expected.attrs)
            for (ackey, acval), (exkey, exval) in zip(actual.attrs.items(), expected.attrs.items()):
                assert ackey == exkey
                assert acval == exval
            assert actual == expected
        G_LOGGER.debug("Graph nodes matched")

        assert graph.outputs == self.outputs
        G_LOGGER.debug("Graph outputs matched")

    def __str__(self):
        return os.path.basename(self.path)


def identity_model():
    path = os.path.join(TEST_ROOT, "models", "identity.onnx")
    model = onnx.load(path)

    x = Variable(name="x", dtype=np.float32, shape=(1, 1, 2, 2))
    y = Variable(name="y", dtype=np.float32, shape=(1, 1, 2, 2))
    node = Node(op="Identity", inputs=[x], outputs=[y])

    return Model(path, inputs=[x], outputs=[y], nodes=[node], opset=OnnxImporter.get_opset(model))


def dim_param_model():
    path = os.path.join(TEST_ROOT, "models", "dim_param.onnx")
    model = onnx.load(path)

    x = Variable(name="Input:0", dtype=np.float32, shape=("dim0", 16, 128))
    y = Variable(name="Output:0", dtype=np.float32, shape=("dim0", 16, 128))
    node = Node(op="Identity", inputs=[x], outputs=[y])

    return Model(path, inputs=[x], outputs=[y], nodes=[node], opset=OnnxImporter.get_opset(model))


def lstm_model():
    path = os.path.join(TEST_ROOT, "models", "lstm.onnx")
    model = onnx.load(path)
    onnx_graph = model.graph

    def load_initializer(index: int) -> np.ndarray:
        return onnx.numpy_helper.to_array(onnx_graph.initializer[index])

    # Optional inputs are represented by empty tensors
    X = Variable(name="X", dtype=np.float32, shape=(4, 3, 6))
    W = Constant(name="W", values=load_initializer(0))
    R = Constant(name="R", values=load_initializer(1))
    B = Constant(name="B", values=load_initializer(2))
    initial_c = Constant(name="initial_c", values=load_initializer(3))

    Y = Variable(name="Y", dtype=np.float32, shape=(4, 1, 3, 5))
    Y_h = Variable(name="Y_h", dtype=np.float32, shape=(1, 3, 5))
    Y_c = Variable(name="Y_c", dtype=np.float32, shape=(1, 3, 5))

    attrs = OrderedDict()
    attrs["direction"] = "forward"
    attrs["hidden_size"] = 5
    node = Node(
        op="LSTM",
        attrs=attrs,
        inputs=[X, W, R, B, Variable.empty(), Variable.empty(), initial_c],
        outputs=[Y, Y_h, Y_c],
    )

    # Initializers will not be included in the graph inputs.
    return Model(
        path,
        inputs=[X],
        outputs=[Y, Y_h, Y_c],
        nodes=[node],
        opset=OnnxImporter.get_opset(model),
    )


def scan_model():
    path = os.path.join(TEST_ROOT, "models", "scan.onnx")
    model = onnx.load(path)

    # Body graph
    sum_in = Variable(name="sum_in", dtype=np.float32, shape=(2,))
    next = Variable(name="next", dtype=np.float32, shape=(2,))
    sum_out = Variable(name="sum_out", dtype=np.float32, shape=(2,))
    scan_out = Variable(name="scan_out", dtype=np.float32, shape=(2,))

    body_nodes = [
        Node(op="Add", inputs=[sum_in, next], outputs=[sum_out]),
        Node(op="Identity", inputs=[sum_out], outputs=[scan_out]),
    ]
    body_graph = Graph(nodes=body_nodes, inputs=[sum_in, next], outputs=[sum_out, scan_out], name="scan_body")

    # Outer graph
    inputs = [
        Variable(name="initial", dtype=np.float32, shape=(2,)),
        Variable(name="x", dtype=np.float32, shape=(3, 2)),
    ]
    outputs = [
        Variable(name="y", dtype=np.float32, shape=(2,)),
        Variable(name="z", dtype=np.float32, shape=(3, 2)),
    ]

    attrs = OrderedDict()
    attrs["body"] = body_graph
    attrs["num_scan_inputs"] = 1
    scan_node = Node(op="Scan", inputs=inputs, outputs=outputs, attrs=attrs)
    return Model(
        path,
        inputs=inputs,
        outputs=outputs,
        nodes=[scan_node],
        opset=OnnxImporter.get_opset(model),
    )


def initializer_is_output_model():
    path = os.path.join(TEST_ROOT, "models", "initializer_is_output.onnx")
    model = onnx.load(path)

    X = Constant(name="X", values=np.ones((64, 64), dtype=np.float32))

    return Model(path, inputs=[], outputs=[X], nodes=[], opset=OnnxImporter.get_opset(model))


# Node includes a subgraph whose I/O names are the same as that of the node.
def nested_dup_names():
    path = os.path.join(TEST_ROOT, "models", "nested_dup_names.onnx")
    model = onnx.load(path)

    # Inner
    subgraph_inputs = [Variable("X", shape=(2, 2), dtype=np.float32)]
    subgraph_outputs = [Variable("Y", shape=(2, 2), dtype=np.float32)]

    subgraph_node = Node(op="Identity", inputs=subgraph_inputs, outputs=subgraph_outputs)
    subgraph = Graph(nodes=[subgraph_node], inputs=subgraph_inputs, outputs=subgraph_outputs)

    # Outer - problem happens if outer node has same I/O names as subgraph
    inputs = [Variable("X", shape=(2, 2), dtype=np.float32)]
    outputs = [Variable("Y", shape=(2, 2), dtype=np.float32)]

    node = Node(op="Nested", inputs=inputs, outputs=outputs, attrs={"body": subgraph})
    return Model(
        path,
        inputs=inputs,
        outputs=outputs,
        nodes=[node],
        opset=OnnxImporter.get_opset(model),
    )


def ext_weights():
    path = os.path.join(TEST_ROOT, "models", "ext_weights.onnx")
    model = onnx.load(path)

    inputs = [Variable("input", shape=(1, 3), dtype=np.float32)]
    outputs = [Variable("output", shape=(1, 3), dtype=np.float32)]

    a = Constant("a", values=np.ones((1, 3), dtype=np.float32))
    b = Constant("b", values=np.ones((1, 3), dtype=np.float32))
    d = Constant("d", values=np.ones((1, 3), dtype=np.float32))

    c = Variable("c")
    e = Variable("e")

    nodes = [
        Node(op="Add", inputs=[a, b], outputs=[c]),
        Node(op="Add", inputs=[c, d], outputs=[e]),
        Node(op="Add", inputs=[inputs[0], e], outputs=outputs),
    ]

    return Model(
        path,
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        opset=OnnxImporter.get_opset(model),
    )


def const_foldable():
    path = os.path.join(TEST_ROOT, "models", "const_foldable.onnx")
    return Model(path, inputs=None, outputs=None, nodes=None, opset=None)  # Only used for path.


def shape_cast_elision():
    path = os.path.join(TEST_ROOT, "models", "shape_cast_elision.onnx")
    return Model(path, inputs=None, outputs=None, nodes=None, opset=None)  # Only used for path.
