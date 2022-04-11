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

# Register functions to make graph generation easier
@gs.Graph.register()
def min(self, *args):
    return self.layer(op="Min", inputs=args, outputs=["min_out"])[0]


@gs.Graph.register()
def max(self, *args):
    return self.layer(op="Max", inputs=args, outputs=["max_out"])[0]


@gs.Graph.register()
def identity(self, inp):
    return self.layer(op="Identity", inputs=[inp], outputs=["identity_out"])[0]


# Generate the graph
graph = gs.Graph()

graph.inputs = [gs.Variable("input", shape=(4, 4), dtype=np.float32)]

# Clip values to [0, 6]
MIN_VAL = np.array(0, np.float32)
MAX_VAL = np.array(6, np.float32)

# Add identity nodes to make the graph structure a bit more interesting
inp = graph.identity(graph.inputs[0])
max_out = graph.max(graph.min(inp, MAX_VAL), MIN_VAL)
graph.outputs = [
    graph.identity(max_out),
]

# Graph outputs must include dtype information
graph.outputs[0].to_variable(dtype=np.float32, shape=(4, 4))

onnx.save(gs.export_onnx(graph), "model.onnx")
