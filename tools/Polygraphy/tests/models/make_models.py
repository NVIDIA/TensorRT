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
"""
Helper utility to generate models to help test the `debug reduce`
subtool, which reduces failing ONNX models.
"""
import os

import tempfile
import numpy as np
import onnx
import subprocess
import onnx_graphsurgeon as gs
from meta import ONNX_MODELS
from polygraphy.tools.sparse import SparsityPruner

CURDIR = os.path.dirname(__file__)


@gs.Graph.register()
def identity(self, inp, **kwargs):
    out = self.layer(op="Identity", inputs=[inp], outputs=["identity_out"], **kwargs)[0]
    out.dtype = inp.dtype
    return out


@gs.Graph.register()
def add(self, a, b, **kwargs):
    return self.layer(op="Add", inputs=[a, b], outputs=["add_out"], **kwargs)[0]


@gs.Graph.register()
def div(self, a, b, **kwargs):
    return self.layer(op="Div", inputs=[a, b], outputs=["div_out"], **kwargs)[0]


@gs.Graph.register()
def sub(self, a, b, **kwargs):
    return self.layer(op="Sub", inputs=[a, b], outputs=["sub_out"], **kwargs)[0]


@gs.Graph.register()
def constant(self, values: gs.Constant, **kwargs):
    return self.layer(
        op="Constant", outputs=["constant_out"], attrs={"value": values}, **kwargs
    )[0]


@gs.Graph.register()
def reshape(self, data, shape, **kwargs):
    return self.layer(
        op="Reshape", inputs=[data, shape], outputs=["reshape_out"], **kwargs
    )[0]


@gs.Graph.register()
def matmul(self, a, b, **kwargs):
    return self.layer(op="MatMul", inputs=[a, b], outputs=["matmul_out"], **kwargs)[0]


@gs.Graph.register()
def tile(self, inp, repeats):
    return self.layer(op="Tile", inputs=[inp, repeats], outputs=["tile_out"])[0]


@gs.Graph.register()
def nonzero(self, inp, **kwargs):
    return self.layer(op="NonZero", inputs=[inp], outputs=["nonzero_out"], **kwargs)[0]


# Name range as onnx_range as range is a python built-in function.
@gs.Graph.register()
def onnx_range(self, start, limit, delta, **kwargs):
    return self.layer(
        op="Range", inputs=[start, limit, delta], outputs=["range_out"], **kwargs
    )[0]


@gs.Graph.register()
def cast(self, input, type, **kwargs):
    return self.layer(
        op="Cast", inputs=[input], attrs={"to": type}, outputs=["cast_out"], **kwargs
    )[0]


@gs.Graph.register()
def reduce_max(self, input, keep_dims, **kwargs):
    return self.layer(
        op="ReduceMax",
        inputs=[input],
        attrs={"keepdims": keep_dims},
        outputs=["reduce_max_out"],
        **kwargs,
    )[0]


@gs.Graph.register()
def conv(self, input, weights, kernel_shape, **kwargs):
    return self.layer(
        op="Conv",
        inputs=[input, weights],
        attrs={"kernel_shape": kernel_shape},
        outputs=["conv_out"],
        **kwargs,
    )[0]


@gs.Graph.register()
def split(self, inp, split, axis=0):
    return self.layer(
        op="Split",
        inputs=[inp],
        outputs=[f"split_out_{i}" for i in range(len(split))],
        attrs={"axis": axis, "split": split},
    )


@gs.Graph.register()
def transpose(self, inp, **kwargs):
    return self.layer(
        op="Transpose", inputs=[inp], outputs=["transpose_out"], **kwargs
    )[0]


@gs.Graph.register()
def quantize_linear(self, inp, y_scale, y_zero_point, **kwargs):
    return self.layer(
        op="QuantizeLinear",
        inputs=[inp, y_scale, y_zero_point],
        outputs=["quantize_linear_out"],
        **kwargs,
    )[0]


@gs.Graph.register()
def dequantize_linear(self, inp, x_scale, x_zero_point, **kwargs):
    return self.layer(
        op="DequantizeLinear",
        inputs=[inp, x_scale, x_zero_point],
        outputs=["dequantize_linear_out"],
        **kwargs,
    )[0]


def save(graph, model_name):
    path = os.path.join(CURDIR, model_name)
    print(f"Writing: {path}")
    onnx.save(gs.export_onnx(graph), path)


def make_sparse(graph):
    sparsity_pruner = SparsityPruner(gs.export_onnx(graph))
    return gs.import_onnx(sparsity_pruner.prune())


# Generates a model with multiple inputs/outputs:
#
#    X0    Y0
#    |     |
#    X1    Y1
#      \  /
#       Z0
#      /  \
#    Z1    Z2
#
def make_multi_input_output():
    DTYPE = np.float32
    SHAPE = (1,)

    X0 = gs.Variable("X0", dtype=DTYPE, shape=SHAPE)
    Y0 = gs.Variable("Y0", dtype=DTYPE, shape=SHAPE)

    graph = gs.Graph(inputs=[X0, Y0])

    X1 = graph.identity(X0)
    Y1 = graph.identity(Y0)

    Z0 = graph.add(X1, Y1)

    Z1 = graph.identity(Z0)
    Z1.dtype = DTYPE
    Z1.shape = SHAPE

    Z2 = graph.identity(Z0)
    Z2.dtype = DTYPE
    Z2.shape = SHAPE

    graph.outputs = [Z1, Z2]

    save(graph, "reducable.onnx")


make_multi_input_output()


# Generates a linear model with a Constant node and no inputs:
#
#    X0 (Constant)
#    |
#    X1 (Identity)
#    |
#    X2 (Identity)
#
def make_constant_linear():
    DTYPE = np.float32
    SHAPE = (4, 4)

    graph = gs.Graph()

    X0 = graph.constant(gs.Constant("const", values=np.ones(SHAPE, dtype=DTYPE)))
    # Explicitly clear shape to trigger the failure condition in reduce
    X0.shape = None

    X1 = graph.identity(X0)
    X2 = graph.identity(X1)
    X2.dtype = DTYPE
    X2.shape = SHAPE

    graph.outputs = [X2]

    save(graph, "reducable_with_const.onnx")


make_constant_linear()


# Generates a model whose node uses the same tensor for multiple inputs
#
#    inp
#    / \
#    Add
#     |
#    out
#
def make_dup_input():
    DTYPE = np.float32
    SHAPE = (4, 4)

    inp = gs.Variable("inp", dtype=DTYPE, shape=SHAPE)

    graph = gs.Graph(inputs=[inp])
    out = graph.add(inp, inp)
    out.dtype = DTYPE
    graph.outputs = [out]

    save(graph, "add_with_dup_inputs.onnx")


make_dup_input()


# Generates a model with a no-op reshape
#
#     inp    shape
#        \ /
#       Reshape
#         |
#        out
#
def make_no_op_reshape():
    DTYPE = np.float32
    SHAPE = (4, 4)

    data = gs.Variable("data", dtype=DTYPE, shape=SHAPE)

    graph = gs.Graph(inputs=[data])
    out = graph.reshape(data, np.array(SHAPE, dtype=np.int64))
    out.dtype = DTYPE
    graph.outputs = [out]

    save(graph, "no_op_reshape.onnx")


make_no_op_reshape()


# Generates a model that overflows FP16
#
#   inp
#    |
#  MatMul
#    |
#  Add
#   |
#  Sub
#   |
#  MatMul
#   |
#  out
#
def make_needs_constraints():
    SIZE = 256

    x = gs.Variable("x", shape=(1, 1, SIZE, SIZE), dtype=np.float32)
    I_rot90 = gs.Constant(
        name="I_rot90",
        values=np.rot90(
            np.identity(SIZE, dtype=np.float32).reshape((1, 1, SIZE, SIZE))
        ),
    )
    fp16_max = gs.Constant(
        name="fp16_max",
        values=np.array([np.finfo(np.float16).max], dtype=np.float32).reshape(
            (1, 1, 1, 1)
        ),
    )

    graph = gs.Graph(inputs=[x])
    y = graph.matmul(x, I_rot90, name="MatMul_0")
    z = graph.add(y, fp16_max, name="Add")
    w = graph.sub(z, fp16_max, name="Sub")
    u = graph.matmul(w, I_rot90, name="MatMul_1")

    u.dtype = np.float32
    graph.outputs = [u]

    save(graph, "needs_constraints.onnx")


make_needs_constraints()


# Generates a model that will become very large when constant-folded
#
#   inp
#    |
#  Tile
#    |
#   out
#
def make_constant_fold_bloater():
    graph = gs.Graph()
    # Input is 1MiB, tiled to 10MiB
    out = graph.tile(
        np.ones(shape=(1024, 256), dtype=np.float32), repeats=np.array([1, 10])
    )
    out.dtype = np.float32
    graph.outputs = [out]

    save(graph, "constant_fold_bloater.onnx")


make_constant_fold_bloater()


# Generate a model with a data-dependent shape
#
#    inp
#     |
#   NonZero
#     |
#    out
#
def make_nonzero():
    inp = gs.Variable("input", shape=(4,), dtype=np.int64)

    graph = gs.Graph(inputs=[inp])
    out = graph.nonzero(inp)
    out.dtype = np.int64
    graph.outputs = [out]

    save(graph, "nonzero.onnx")


make_nonzero()


# Generate a model where a node has multiple outputs that are graph outputs
#
#     inp
#      |
#   Identity
#      |
#     id0
#        \
#       Split
#     /       \
# split_out0   split_out1 (graph output)
#      |
#   Identity
#      |
#  id1 (graph output)
#
#
def make_multi_output():
    inp = gs.Variable("input", shape=(4, 5), dtype=np.float32)

    graph = gs.Graph(inputs=[inp])
    id0 = graph.identity(inp)
    [split_out0, split_out1] = graph.split(id0, split=[2, 2])
    id1 = graph.identity(split_out0)
    graph.outputs = [id1, split_out1]

    for out in graph.outputs:
        out.dtype = np.float32

    save(graph, "multi_output.onnx")


make_multi_output()


# Generate a model where a tensor contains unbounded DDS.
# Use Conv_0 and ReduceMax to generate a DDS scalar tensor, and send to Range as input `limit`.
# The output of Range has an unbounded shape.
#
#     input
#       |
#     Conv_0
#       |
#    ReduceMax
#       |
#     Range
#       |
#     Conv_1
#       |
#     output
#
def make_unbounded_dds():
    input = gs.Variable("Input", shape=(1, 3, 10, 10), dtype=np.float32)
    graph = gs.Graph(inputs=[input], opset=13)
    weights_0 = graph.constant(
        gs.Constant("Weights_0", values=np.ones((3, 3, 3, 3), dtype=np.float32))
    )
    weights_1 = graph.constant(
        gs.Constant("Weights_1", values=np.ones((4, 1, 1, 1), dtype=np.float32))
    )

    conv_0 = graph.conv(input, weights_0, [3, 3], name="Conv_0")
    reduce_max_0 = graph.reduce_max(conv_0, keep_dims=0, name="ReduceMax_0")

    cast_0 = graph.cast(
        reduce_max_0, getattr(onnx.TensorProto, "INT64"), name="Cast_to_int64"
    )
    range_0 = graph.onnx_range(
        np.array(0, dtype=np.int64), cast_0, np.array(1, dtype=np.int64), name="Range"
    )
    cast_1 = graph.cast(
        range_0, getattr(onnx.TensorProto, "FLOAT"), name="Cast_to_float"
    )

    reshape_1 = graph.reshape(
        cast_1, np.array([1, 1, -1, 1], dtype=np.int64), name="Reshape_1"
    )
    conv_1 = graph.conv(reshape_1, weights_1, [1, 1], name="Conv_1")

    graph.outputs = [conv_1]

    for out in graph.outputs:
        out.dtype = np.float32

    save(graph, "unbounded_dds.onnx")


make_unbounded_dds()


def make_small_matmul(name, dtype, save_sparse=False):
    M = 8
    N = 8
    K = 16
    a = gs.Variable("a", shape=(M, K), dtype=dtype)
    g = gs.Graph(inputs=[a], opset=13)
    val = np.random.uniform(-3, 3, size=K * N).astype(dtype).reshape((K, N))
    b = gs.Constant("b", values=val)
    c = g.matmul(a, b, name="matmul")
    c.dtype = dtype
    g.outputs = [c]

    save(g, name)
    if save_sparse:
        save(make_sparse(g), "sparse." + name)


make_small_matmul("matmul.onnx", np.float32, save_sparse=True)
make_small_matmul("matmul.fp16.onnx", np.float16)


def make_small_conv(name):
    N = 1
    C = 16
    H = 8
    W = 8
    K = 4
    F = 4
    a = gs.Variable("a", shape=(N, C, H, W), dtype=np.float32)
    g = gs.Graph(inputs=[a], opset=13)
    val = (
        np.random.uniform(-3, 3, size=K * C * F * F)
        .reshape((K, C, F, F))
        .astype(np.float32)
    )
    b = gs.Constant("b", values=val)
    c = g.conv(a, b, (F, F), name="conv")
    c.dtype = np.float32
    g.outputs = [c]

    save(g, name)
    save(make_sparse(g), "sparse." + name)


make_small_conv("conv.onnx")


def make_unsorted():
    inp = gs.Variable("input", shape=(1, 1), dtype=np.float32)
    graph = gs.Graph(inputs=[inp])
    graph.outputs = [graph.identity(graph.identity(inp))]

    graph.nodes = list(reversed(graph.nodes))
    save(graph, "unsorted.onnx")


make_unsorted()


def make_empty():
    g = gs.Graph(inputs=[], opset=13)
    g.outputs = []

    save(g, "empty.onnx")


make_empty()


# Builds a graph that has unused nodes and inputs.
#
# f  e
# |\  |
# H  G
# |  |
# h  g
# |
# I
# |
# i
#
# e is an unused input.
# G is an unused node.
# This graph is useful for testing if `lint` catches unused nodes and inputs.
def make_cleanable():
    e = gs.Variable(name="e", dtype=np.float32, shape=(1, 1))
    f = gs.Variable(name="f", dtype=np.float32, shape=(1, 1))
    h = gs.Variable(name="h", dtype=np.float32, shape=(1, 1))
    i = gs.Variable(name="i", dtype=np.float32, shape=(1, 1))
    g = gs.Variable(name="g", dtype=np.float32, shape=(2, 1))

    nodes = [
        gs.Node(op="Concat", name="G", inputs=[e, f], outputs=[g], attrs={"axis": 0}),
        gs.Node(op="Dropout", name="H", inputs=[f], outputs=[h]),
        gs.Node(op="Identity", name="I", inputs=[h], outputs=[i]),
    ]

    graph = gs.Graph(nodes=nodes, inputs=[e, f], outputs=[i])
    save(graph, "cleanable.onnx")


make_cleanable()


# Generates a graph with very deranged names
# Tests that the unique renaming in lint tool works
def make_renamable():
    a = gs.Variable(name="a", dtype=np.float32, shape=(1, 1))
    b = gs.Variable(name="b", dtype=np.float32, shape=(1, 1))
    c = gs.Variable(name="c", dtype=np.float32, shape=(1, 1))
    d = gs.Variable(name="d", dtype=np.float32, shape=(1, 1))
    e = gs.Variable(name="e", dtype=np.float32, shape=(2, 1))

    nodes = [
        gs.Node(op="Identity", name="", inputs=[a], outputs=[b]),
        gs.Node(
            op="Dropout", name="polygraphy_unnamed_node_0", inputs=[b], outputs=[c]
        ),
        gs.Node(
            op="Identity", name="polygraphy_unnamed_node_0_0", inputs=[c], outputs=[d]
        ),
        gs.Node(op="Dropout", name="", inputs=[d], outputs=[e]),
    ]

    graph = gs.Graph(nodes=nodes, inputs=[a], outputs=[e])
    save(graph, "renamable.onnx")


make_renamable()

####### Generate some invalid models #######

### Graphs whose errors are data-dependent ###


# Generats an invalid graph with multiple parallel bad nodes.
# The graph is invalid due to multiple parallel nodes failing.
# This is is the graph:
#    A    B    C    D  E    F    G
#     \  /      \  /    \  /      \
#    MatMul_0* Add_0*  MatMul_1 NonZero
#        \        /        \    /
#         MatMul_2       MatMul_3*
#               \       /
#                \     /
#                Add_1
#                  |
#                output
# The graph is invalid because MatMul_0, Add_0 and MatMul_3 all will fail.
# MatMul_0 should fail because A and B are not compatible.
# Add_0 should fail because C and D are not compatible.
# MatMul_3 should fail because result of MatMul2 and the Data-dependent shape of output of
# NonZero are not compatible.
#
# This graph is useful for testing if `lint` catches multiple parallel bad nodes that may/may not be data-dependent.
#
def make_bad_graph_with_parallel_invalid_nodes():
    DTYPE = np.float32
    BAD_DIM = 3

    graph = gs.Graph(name="bad_graph_with_parallel_invalid_nodes")

    A = gs.Variable("A", dtype=DTYPE, shape=(1, BAD_DIM))
    B = gs.Variable("B", dtype=DTYPE, shape=(4, 4))
    mm_ab_out = graph.matmul(
        A, B, name="MatMul_0"
    )  # This node will fail because A and B are not compatible.

    C = gs.Variable("C", dtype=DTYPE, shape=(BAD_DIM, 4))
    D = gs.Variable("D", dtype=DTYPE, shape=(4, 1))
    add_cd_out = graph.add(
        C, D, name="Add_0"
    )  # This node will fail because C and D are not compatible.

    pre_out_1 = graph.matmul(mm_ab_out, add_cd_out, name="MatMul_2")

    E = gs.Variable("E", dtype=DTYPE, shape=(1, 4))
    F = gs.Variable("F", dtype=DTYPE, shape=(4, 1))
    mm_ef_out = graph.matmul(E, F, name="MatMul_1")
    mm_ef_out_int64 = graph.cast(
        mm_ef_out, onnx.TensorProto.INT64, name="cast_to_int64"
    )

    G = gs.Variable("G", dtype=np.int64, shape=(4, 4))
    nz_g_out = graph.nonzero(G, name="NonZero")  # `nz_g_out` shape is data-dependent.

    pre_out_2 = graph.matmul(
        mm_ef_out_int64, nz_g_out, name="MatMul_3"
    )  # This node will fail because `mm_ef_out_int64` and `nz_g_out` are not compatible.
    pre_out_2_float = graph.cast(
        pre_out_2, getattr(onnx.TensorProto, "FLOAT"), name="cast_to_float"
    )

    out = graph.add(pre_out_1, pre_out_2_float, name="Add_1")
    out.dtype = DTYPE

    graph.inputs = [A, B, C, D, E, F, G]
    graph.outputs = [out]

    save(graph, "bad_graph_with_parallel_invalid_nodes.onnx")


make_bad_graph_with_parallel_invalid_nodes()


# Generates the following graph:
#                 cond
#                  |
#                 If
#                  |
#             z (x or y)
#              \   |
#               MatMul
#                  |
#               output
# If `cond` is True, then `x` is used, otherwise `y` is used.
# `x` is compatible with `z`, while `y` is NOT compatible with `z`.
# Based on the value of `cond`, the graph may be valid or invalid.
#
# This graph is useful to check whether the error message is caught or not at runtime based on data input.
#
def make_bad_graph_conditionally_invalid():
    X = [[4.0], [3.0]]  # shape (2, 1), compatible with Z for MatMul
    Y = [2.0, 4.0]  # shape (2,), incompatible with Z for MatMul
    Z = [[2.0, 4.0]]  # shape (1, 2)

    cond = gs.Variable(
        "cond", dtype=np.bool_, shape=(1,)
    )  # input to If, True or False based on user input.

    graph = gs.Graph(name="bad_graph_conditionally_invalid")

    x = gs.Constant("x", values=np.array(X, dtype=np.float32))
    y = gs.Constant("y", values=np.array(Y, dtype=np.float32))

    then_out = gs.Variable("then_out", dtype=np.float32, shape=None)
    else_out = gs.Variable("else_out", dtype=np.float32, shape=None)

    then_const_node = gs.Node(
        op="Constant", inputs=[], outputs=[then_out], attrs={"value": x}
    )  # node for `then_branch` Graph
    else_const_node = gs.Node(
        op="Constant", inputs=[], outputs=[else_out], attrs={"value": y}
    )  # node for `else_branch` Graph

    then_body = gs.Graph(
        nodes=[then_const_node], name="then_body", inputs=[], outputs=[then_out]
    )  # Graph for `then_branch`
    else_body = gs.Graph(
        nodes=[else_const_node], name="else_body", inputs=[], outputs=[else_out]
    )  # Graph for `else_branch`

    res = gs.Variable("res", dtype=np.float32, shape=None)  # shape is data-dependent

    if_node = gs.Node(
        op="If",
        name="If_Node",
        inputs=[cond],
        outputs=[res],
        attrs={"then_branch": then_body, "else_branch": else_body},
    )
    graph.nodes = [if_node]

    out = graph.matmul(
        res, gs.Constant("z", values=np.array(Z, dtype=np.float32)), name="MatMul"
    )
    out.dtype = np.float32

    graph.inputs = [cond]
    graph.outputs = [out]

    save(graph, "bad_graph_conditionally_invalid.onnx")


make_bad_graph_conditionally_invalid()


### Bad GraphProto ###
### Graphs that break the ONNX Specification for GraphProto ###


# Generates a model where the GraphProto has no name.
#
# This is invalid as ONNX Specification requires that the GraphProto has a name.
#
def make_bad_graph_with_no_name():
    DTYPE = np.float32
    SHAPE = (4, 4)

    inp = gs.Variable("inp", dtype=DTYPE, shape=SHAPE)

    graph = gs.Graph(inputs=[inp], name="")
    out = graph.add(inp, inp)
    out.dtype = DTYPE
    graph.outputs = [out]

    save(graph, "bad_graph_with_no_name.onnx")


make_bad_graph_with_no_name()


# Generates a model where the GraphProto has no imports.
#
# This is invalid as ONNX Specification requires that the GraphProto has at least one import.
#
def make_bad_graph_with_no_import_domains():
    DTYPE = np.float32
    SHAPE = (4, 4)

    inp = gs.Variable("inp", dtype=DTYPE, shape=SHAPE)

    graph = gs.Graph(inputs=[inp], import_domains=[])
    out = graph.add(inp, inp)
    out.dtype = DTYPE
    graph.outputs = [out]

    save(graph, "bad_graph_with_no_import_domains.onnx")


make_bad_graph_with_no_import_domains()


# Generates a model where the inputs (value info) of graph are duplicates.
#
# This is invalid as ONNX Specification requires that the (value info) inputs of a graph are unique.
#
#    inp
#    / \
#    Add
#     |
#    out
#
def make_bad_graph_with_dup_value_info():
    DTYPE = np.float32
    SHAPE = (4, 4)

    inp = gs.Variable("inp", dtype=DTYPE, shape=SHAPE)

    graph = gs.Graph(inputs=[inp, inp])
    out = graph.add(inp, inp)
    out.dtype = DTYPE
    graph.outputs = [out]

    save(graph, "bad_graph_with_dup_value_info.onnx")


make_bad_graph_with_dup_value_info()


# Generates a model with mult-level errors.
# The model is invalid because of graph-level error (no name) and node-level error (incompatible inputs).
def make_bad_graph_multi_level_errors():
    DTYPE = np.float32
    SHAPE = (4, 5)

    inp1 = gs.Variable("inp1", dtype=DTYPE, shape=SHAPE)
    inp2 = gs.Variable("inp2", dtype=DTYPE, shape=SHAPE)

    graph = gs.Graph(inputs=[inp1, inp2], name="")  # graph-level error: empty name
    out = graph.matmul(inp1, inp2)  # node-level error: incompatible inputs
    out.dtype = DTYPE
    out.shape = []  # we need to specify this so GS creates valid ONNX model.
    graph.outputs = [out]

    save(graph, "bad_graph_with_multi_level_errors.onnx")


make_bad_graph_multi_level_errors()


# Generates a model where graph has multiple node names with same non-empty string.
def make_bad_graph_with_duplicate_node_names():
    DTYPE = np.float32
    SHAPE = (4, 5)

    inp = gs.Variable("inp", dtype=DTYPE, shape=SHAPE)

    graph = gs.Graph(inputs=[inp], name="bad_graph_with_duplicate_node_names")
    inter1 = graph.identity(inp, name="identical")
    out = graph.identity(
        inter1, name="identical"
    )  # node-level error: duplicate node names
    graph.outputs = [out]

    save(graph, "bad_graph_with_duplicate_node_names.onnx")


make_bad_graph_with_duplicate_node_names()


# Generates a model where the graph has a subgraph matching toyPlugin's graph pattern
def make_graph_with_subgraph_matching_toy_plugin():
    i0 = gs.Variable(name="i0", dtype=np.float32)
    i1 = gs.Variable(name="i1", dtype=np.float32)
    i2 = gs.Variable(name="i2", dtype=np.float32)
    i3 = gs.Variable(name="i3", dtype=np.float32)
    i4 = gs.Variable(name="i4", dtype=np.float32)

    o1 = gs.Variable(name="o1", dtype=np.float32)
    o2 = gs.Variable(name="o2", dtype=np.float32)

    O_node = gs.Node(op="O", inputs=[i0], outputs=[i1], name="n1")
    A_node = gs.Node(op="A", inputs=[i1], outputs=[i2], name="n2")
    B_node = gs.Node(op="B", inputs=[i1], outputs=[i3], name="n3")
    C_node = gs.Node(op="C", inputs=[i2, i3], outputs=[i4], attrs={"x": 1}, name="n4")
    D_node = gs.Node(op="D", inputs=[i4], outputs=[o1], name="n5")
    E_node = gs.Node(op="E", inputs=[i4], outputs=[o2], name="n6")

    graph = gs.Graph(
        nodes=[O_node, A_node, B_node, C_node, D_node, E_node],
        inputs=[i0],
        outputs=[o1, o2],
    )

    save(graph, "toy_subgraph.onnx")


make_graph_with_subgraph_matching_toy_plugin()


# Generates the following Graph
#
# The input to the Transpose op is an initializer
#
#    Transpose
#       |
#      MatMul
#       |
#      out
#
def make_transpose_matmul():
    M = 8
    N = 8
    K = 16
    a = gs.Variable("a", shape=(M, K), dtype=np.float32)
    g = gs.Graph(inputs=[a], opset=13)
    val = np.random.uniform(-3, 3, size=K * N).astype(np.float32).reshape((N, K))
    b = gs.Constant("b", values=val)
    b_transpose = g.transpose(b, name="transpose")
    c = g.matmul(a, b_transpose, name="matmul")
    c.dtype = np.float32
    g.outputs = [c]

    save(g, "transpose_matmul.onnx")


make_transpose_matmul()


# Generates the following Graph
#
# The input to the QuantizeLinear op is an initializer
#
#    QuantizeLinear
#       |
#    DequantizeLinear
#       |
#      Conv
#       |
#      out
#
def make_qdq_conv():
    x = (
        np.random.uniform(-3, 3, size=3 * 3 * 130)
        .astype(np.float32)
        .reshape((1, 3, 3, 130))
    )
    y_scale = np.array([2, 4, 5], dtype=np.float32)
    y_zero_point = np.array([84, 24, 196], dtype=np.uint8)
    x_const = gs.Constant("x", values=x)
    y_scale_const = gs.Constant("y_scale", values=y_scale)
    y_zero_point_const = gs.Constant("y_zero_point", values=y_zero_point)

    weight = gs.Constant("Weights_0", values=np.ones((3, 3, 3, 3), dtype=np.float32))

    g = gs.Graph(inputs=[], opset=13)
    q_layer = g.quantize_linear(x_const, y_scale_const, y_zero_point_const)
    dq_layer = g.dequantize_linear(q_layer, y_scale_const, y_zero_point_const)
    out = g.conv(dq_layer, weight, [3, 3], name="Conv_0")
    out.dtype = np.float32
    g.outputs = [out]

    save(g, "qdq_conv.onnx")


make_qdq_conv()


def make_weightless_network(model_name):
    ipath = ONNX_MODELS[model_name].path
    opath = os.path.join(CURDIR, "weightless." + model_name + ".onnx")
    cmd = [f"polygraphy surgeon weight-strip {ipath} -o {opath}"]
    subprocess.run(cmd, shell=True)


make_weightless_network("matmul.fp16")
make_weightless_network("matmul.bf16")
make_weightless_network("sparse.matmul")
make_weightless_network("conv")
make_weightless_network("sparse.conv")
make_weightless_network("transpose_matmul")
make_weightless_network("qdq_conv")
