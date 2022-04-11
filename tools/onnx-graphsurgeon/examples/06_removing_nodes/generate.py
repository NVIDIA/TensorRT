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

# Inputs
x = gs.Variable(name="x", dtype=np.float32, shape=(1, 3, 224, 224))

# Intermediate tensors
i0 = gs.Variable(name="i0")
i1 = gs.Variable(name="i1")

# Outputs
y = gs.Variable(name="y", dtype=np.float32)

nodes = [
    gs.Node(op="Identity", inputs=[x], outputs=[i0]),
    gs.Node(op="FakeNodeToRemove", inputs=[i0], outputs=[i1]),
    gs.Node(op="Identity", inputs=[i1], outputs=[y]),
]

graph = gs.Graph(nodes=nodes, inputs=[x], outputs=[y])
onnx.save(gs.export_onnx(graph), "model.onnx")
