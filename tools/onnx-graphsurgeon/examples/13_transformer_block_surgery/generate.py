#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import onnx
import onnx_graphsurgeon as gs


hidden_size = 8
num_heads = 2
head_dim = hidden_size // num_heads
sequence_length = 4

tokens = gs.Variable("tokens", dtype=np.float32, shape=(1, sequence_length, hidden_size))
identity_out = gs.Variable(
    "identity_out", dtype=np.float32, shape=(1, sequence_length, hidden_size)
)
split_shape = gs.Constant(
    "split_shape", values=np.array([1, sequence_length, num_heads, head_dim], dtype=np.int64)
)
merged_shape = gs.Constant(
    "merged_shape", values=np.array([1, sequence_length, hidden_size], dtype=np.int64)
)
split_heads = gs.Variable(
    "split_heads", dtype=np.float32, shape=(1, sequence_length, num_heads, head_dim)
)
heads_first = gs.Variable(
    "heads_first", dtype=np.float32, shape=(1, num_heads, sequence_length, head_dim)
)
tokens_again = gs.Variable(
    "tokens_again", dtype=np.float32, shape=(1, sequence_length, num_heads, head_dim)
)
merged = gs.Variable("merged", dtype=np.float32, shape=(1, sequence_length, hidden_size))

weights = gs.Constant(
    "projection_weight",
    values=np.linspace(-0.5, 0.5, hidden_size * hidden_size, dtype=np.float32).reshape(
        hidden_size, hidden_size
    ),
)
bias = gs.Constant(
    "projection_bias",
    values=np.linspace(-0.1, 0.1, hidden_size, dtype=np.float32),
)
projected = gs.Variable("projected", dtype=np.float32, shape=(1, sequence_length, hidden_size))
output = gs.Variable("output", dtype=np.float32, shape=(1, sequence_length, hidden_size))

nodes = [
    gs.Node("Identity", inputs=[tokens], outputs=[identity_out]),
    gs.Node("Reshape", inputs=[identity_out, split_shape], outputs=[split_heads]),
    gs.Node(
        "Transpose",
        attrs={"perm": [0, 2, 1, 3]},
        inputs=[split_heads],
        outputs=[heads_first],
    ),
    gs.Node(
        "Transpose",
        attrs={"perm": [0, 2, 1, 3]},
        inputs=[heads_first],
        outputs=[tokens_again],
    ),
    gs.Node("Reshape", inputs=[tokens_again, merged_shape], outputs=[merged]),
    gs.Node("MatMul", inputs=[merged, weights], outputs=[projected]),
    gs.Node("Add", inputs=[projected, bias], outputs=[output]),
]

graph = gs.Graph(nodes=nodes, inputs=[tokens], outputs=[output], opset=18)
model = gs.export_onnx(graph.cleanup().toposort())
onnx.checker.check_model(model)
onnx.save(model, "model.onnx")
