#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import onnx
import onnx_graphsurgeon as gs

CURDIR = os.path.dirname(__file__)


@gs.Graph.register()
def identity(self, inp, **kwargs):
    return self.layer(op="Identity", inputs=[inp], outputs=["identity_out"], **kwargs)[0]


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
    return self.layer(op="Constant", outputs=["constant_out"], attrs={"value": values}, **kwargs)[0]


@gs.Graph.register()
def reshape(self, data, shape, **kwargs):
    return self.layer(op="Reshape", inputs=[data, shape], outputs=["reshape_out"], **kwargs)[0]


@gs.Graph.register()
def matmul(self, a, b, **kwargs):
    return self.layer(op="MatMul", inputs=[a, b], outputs=["matmul_out"], **kwargs)[0]


@gs.Graph.register()
def tile(self, inp, repeats):
    return self.layer(op="Tile", inputs=[inp, repeats], outputs=["tile_out"])[0]


@gs.Graph.register()
def nonzero(self, inp):
    return self.layer(op="NonZero", inputs=[inp], outputs=["nonzero_out"])[0]


# Name range as onnx_range as range is a python built-in function.
@gs.Graph.register()
def onnx_range(self, start, limit, delta, **kwargs):
    return self.layer(op="Range", inputs=[start, limit, delta], outputs=["range_out"], **kwargs)[0]


@gs.Graph.register()
def cast(self, input, type, **kwargs):
    return self.layer(op="Cast", inputs=[input], attrs={"to": type}, outputs=["cast_out"], **kwargs)[0]


@gs.Graph.register()
def reduce_max(self, input, keep_dims, **kwargs):
    return self.layer(op="ReduceMax", inputs=[input], attrs={"keepdims": keep_dims}, outputs=["reduce_max_out"], **kwargs)[0]


@gs.Graph.register()
def conv(self, input, weights, kernel_shape, **kwargs):
    return self.layer(op="Conv", inputs=[input, weights], attrs={"kernel_shape": kernel_shape}, outputs=["conv_out"], **kwargs)[0]


@gs.Graph.register()
def split(self, inp, split, axis=0):
    return self.layer(
        op="Split",
        inputs=[inp],
        outputs=[f"split_out_{i}" for i in range(len(split))],
        attrs={"axis": axis, "split": split},
    )


def save(graph, model_name):
    path = os.path.join(CURDIR, model_name)
    print(f"Writing: {path}")
    onnx.save(gs.export_onnx(graph), path)


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
        name="I_rot90", values=np.rot90(np.identity(SIZE, dtype=np.float32).reshape((1, 1, SIZE, SIZE)))
    )
    fp16_max = gs.Constant(
        name="fp16_max", values=np.array([np.finfo(np.float16).max], dtype=np.float32).reshape((1, 1, 1, 1))
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
    out = graph.tile(np.ones(shape=(1024, 256), dtype=np.float32), repeats=np.array([1, 10]))
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
    weights_0 = graph.constant(gs.Constant("Weights_0", values=np.ones((3, 3, 3, 3), dtype=np.float32)))
    weights_1 = graph.constant(gs.Constant("Weights_1", values=np.ones((4, 1, 1, 1), dtype=np.float32)))

    conv_0 = graph.conv(input, weights_0, [3, 3], name="Conv_0")
    reduce_max_0 = graph.reduce_max(conv_0, keep_dims=0, name="ReduceMax_0")

    cast_0 = graph.cast(reduce_max_0, getattr(onnx.TensorProto, "INT64"), name="Cast_to_int64")
    range_0 = graph.onnx_range(np.array(0, dtype=np.int64), cast_0, np.array(1, dtype=np.int64), name="Range")
    cast_1 = graph.cast(range_0, getattr(onnx.TensorProto, "FLOAT"), name="Cast_to_float")

    reshape_1 = graph.reshape(cast_1, np.array([1, 1, -1, 1], dtype=np.int64), name="Reshape_1")
    conv_1 = graph.conv(reshape_1, weights_1, [1, 1], name="Conv_1")

    graph.outputs = [conv_1]

    for out in graph.outputs:
        out.dtype = np.float32

    save(graph, "unbounded_dds.onnx")


make_unbounded_dds()
