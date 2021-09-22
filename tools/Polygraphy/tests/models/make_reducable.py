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
def identity(self, inp):
    return self.layer(op="Identity", inputs=[inp], outputs=["identity_out"])[0]


@gs.Graph.register()
def add(self, a, b):
    return self.layer(op="Add", inputs=[a, b], outputs=["add_out"])[0]


@gs.Graph.register()
def constant(self, values: gs.Constant):
    return self.layer(op="Constant", outputs=["constant_out"], attrs={"value": values})[0]


def save(graph, model_name):
    path = os.path.join(CURDIR, model_name)
    print("Writing: {:}".format(path))
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
