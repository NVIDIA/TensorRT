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
import onnx


# Here we'll register a function to do all the subgraph-replacement heavy-lifting.
# NOTE: Since registered functions are entirely reusable, it may be a good idea to
# refactor them into a separate module so you can use them across all your models.
@gs.Graph.register()
def replace_with_clip(self, inputs, outputs):
    # Disconnect output nodes of all input tensors
    for inp in inputs:
        inp.outputs.clear()

    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()

    # Insert the new node.
    return self.layer(op="Clip", inputs=inputs, outputs=outputs)


# Now we'll do the actual replacement
graph = gs.import_onnx(onnx.load("model.onnx"))

tmap = graph.tensors()
# You can figure out the input and output tensors using Netron. In our case:
# Inputs: [inp, MIN_VAL, MAX_VAL]
# Outputs: [max_out]
inputs = [tmap["identity_out_0"], tmap["onnx_graphsurgeon_constant_5"], tmap["onnx_graphsurgeon_constant_2"]]
outputs = [tmap["max_out_6"]]

graph.replace_with_clip(inputs, outputs)

# Remove the now-dangling subgraph.
graph.cleanup().toposort()

# That's it!
onnx.save(gs.export_onnx(graph), "replaced.onnx")
