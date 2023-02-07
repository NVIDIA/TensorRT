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
def make_nonzero():
    inp = gs.Variable("input", shape=(4,), dtype=np.int64)

    graph = gs.Graph(inputs=[inp])
    out = graph.nonzero(inp)
    out.dtype = np.int64
    graph.outputs = [out]

    save(graph, "nonzero.onnx")


make_nonzero()
