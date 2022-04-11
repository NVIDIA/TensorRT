#!/usr/bin/env python3
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

import onnx_graphsurgeon as gs
import numpy as np
import onnx

# Computes Y = x0 + (a * x1 + b)

shape = (1, 3, 224, 224)
# Inputs
x0 = gs.Variable(name="x0", dtype=np.float32, shape=shape)
x1 = gs.Variable(name="x1", dtype=np.float32, shape=shape)

# Intermediate tensors
a = gs.Constant("a", values=np.ones(shape=shape, dtype=np.float32))
b = gs.Constant("b", values=np.ones(shape=shape, dtype=np.float32))
mul_out = gs.Variable(name="mul_out")
add_out = gs.Variable(name="add_out")

# Outputs
Y = gs.Variable(name="Y", dtype=np.float32, shape=shape)

nodes = [
    # mul_out = a * x1
    gs.Node(op="Mul", inputs=[a, x1], outputs=[mul_out]),
    # add_out = mul_out + b
    gs.Node(op="Add", inputs=[mul_out, b], outputs=[add_out]),
    # Y = x0 + add
    gs.Node(op="Add", inputs=[x0, add_out], outputs=[Y]),
]

graph = gs.Graph(nodes=nodes, inputs=[x0, x1], outputs=[Y])
onnx.save(gs.export_onnx(graph), "model.onnx")
