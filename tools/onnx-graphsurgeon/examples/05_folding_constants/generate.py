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

# Computes outputs = input + ((a + b) + d)

shape = (1, 3)
# Inputs
input = gs.Variable("input", shape=shape, dtype=np.float32)

# Intermediate tensors
a = gs.Constant("a", values=np.ones(shape=shape, dtype=np.float32))
b = gs.Constant("b", values=np.ones(shape=shape, dtype=np.float32))
c = gs.Variable("c")
d = gs.Constant("d", values=np.ones(shape=shape, dtype=np.float32))
e = gs.Variable("e")

# Outputs
output = gs.Variable("output", shape=shape, dtype=np.float32)

nodes = [
    # c = (a + b)
    gs.Node("Add", inputs=[a, b], outputs=[c]),
    # e = (c + d)
    gs.Node("Add", inputs=[c, d], outputs=[e]),
    # output = input + e
    gs.Node("Add", inputs=[input, e], outputs=[output]),
]

graph = gs.Graph(nodes=nodes, inputs=[input], outputs=[output])
onnx.save(gs.export_onnx(graph), "model.onnx")
