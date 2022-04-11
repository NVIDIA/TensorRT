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

graph = gs.import_onnx(onnx.load("model.onnx"))

# 1. Remove the `b` input of the add node
first_add = [node for node in graph.nodes if node.op == "Add"][0]
first_add.inputs = [inp for inp in first_add.inputs if inp.name != "b"]

# 2. Change the Add to a LeakyRelu
first_add.op = "LeakyRelu"
first_add.attrs["alpha"] = 0.02

# 3. Add an identity after the add node
identity_out = gs.Variable("identity_out", dtype=np.float32)
identity = gs.Node(op="Identity", inputs=first_add.outputs, outputs=[identity_out])
graph.nodes.append(identity)

# 4. Modify the graph output to be the identity output
graph.outputs = [identity_out]

# 5. Remove unused nodes/tensors, and topologically sort the graph
# ONNX requires nodes to be topologically sorted to be considered valid.
# Therefore, you should only need to sort the graph when you have added new nodes out-of-order.
# In this case, the identity node is already in the correct spot (it is the last node,
# and was appended to the end of the list), but to be on the safer side, we can sort anyway.
graph.cleanup().toposort()

onnx.save(gs.export_onnx(graph), "modified.onnx")
