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


N, C, H, W = 1, 8, 4, 4
NUM_GROUPS = 4

graph = gs.Graph(ir_version=10, opset=17)

x = gs.Variable("input", dtype=np.float32, shape=(N, C, H, W))
graph.inputs = [x]

target_shape = gs.Constant("reshape_target", values=np.array([N, NUM_GROUPS, -1], dtype=np.int64))
reshape_in = gs.Variable("reshape_in_out", dtype=np.float32)
graph.nodes.append(gs.Node(op="Reshape", inputs=[x, target_shape], outputs=[reshape_in]))

inst_scale = gs.Constant("inst_scale", values=np.ones((NUM_GROUPS,), dtype=np.float32))
inst_bias = gs.Constant("inst_bias", values=np.zeros((NUM_GROUPS,), dtype=np.float32))
inst_out = gs.Variable("inst_out", dtype=np.float32)
graph.nodes.append(
    gs.Node(
        op="InstanceNormalization",
        attrs={"epsilon": 1e-5},
        inputs=[reshape_in, inst_scale, inst_bias],
        outputs=[inst_out],
    )
)

shape_out = gs.Variable("shape_out", dtype=np.int64)
graph.nodes.append(gs.Node(op="Shape", inputs=[x], outputs=[shape_out]))

reshape_back_out = gs.Variable("reshape_back_out", dtype=np.float32)
graph.nodes.append(gs.Node(op="Reshape", inputs=[inst_out, shape_out], outputs=[reshape_back_out]))

gamma = gs.Constant("gamma", values=np.random.rand(C).astype(np.float32).reshape(1, C, 1, 1))
beta = gs.Constant("beta", values=np.random.rand(C).astype(np.float32).reshape(1, C, 1, 1))
mul_out = gs.Variable("mul_out", dtype=np.float32)
add_out = gs.Variable("output", dtype=np.float32, shape=(N, C, H, W))
graph.nodes.append(gs.Node(op="Mul", inputs=[reshape_back_out, gamma], outputs=[mul_out]))
graph.nodes.append(gs.Node(op="Add", inputs=[mul_out, beta], outputs=[add_out]))

graph.outputs = [add_out]
onnx.save(gs.export_onnx(graph), "model.onnx")
