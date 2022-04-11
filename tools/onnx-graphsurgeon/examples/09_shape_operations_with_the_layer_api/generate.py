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


# Register operators we'll need.
# NOTE: Since all the ops used here only have a single output, we return the
# first output directly instead of returning the list of outputs.
@gs.Graph.register()
def shape(self, a):
    return self.layer(op="Shape", inputs=[a], outputs=["shape_out_gs"])[0]


@gs.Graph.register()
def reduce_prod(self, a, axes, keepdims=True):
    return self.layer(
        op="ReduceProd", inputs=[a], attrs={"axes": axes, "keepdims": int(keepdims)}, outputs=["reduce_prod_out_gs"]
    )[0]


@gs.Graph.register()
def reshape(self, data, shape):
    return self.layer(op="Reshape", inputs=[data, shape], outputs=["reshape_out_gs"])[0]


@gs.Graph.register()
def gather(self, data, indices):
    return self.layer(op="Gather", inputs=[data, indices], outputs=["gather_out_gs"])[0]


@gs.Graph.register()
def concat(self, inputs, axis=0):
    return self.layer(op="Concat", inputs=inputs, attrs={"axis": axis}, outputs=["concat_out_gs"])[0]


# Create the graph.
graph = gs.Graph()

# First we set up the inputs, using gs.Tensor.DYNAMIC to specify dynamic dimensions.
graph.inputs = [gs.Variable(name="data", dtype=np.float32, shape=(gs.Tensor.DYNAMIC, 3, gs.Tensor.DYNAMIC, 5))]

input_shape = graph.shape(graph.inputs[0])

# Part 1 - Flattening the input by computing its volume and reshaping.
volume = graph.reduce_prod(input_shape, axes=[0])
flattened = graph.reshape(graph.inputs[0], volume)

# Part 2 - Collapsing some, but not all, dimensions. In this case, we will flatten the last 2 dimensions.
# To do so, we'll gather the last 2 dimensions, compute their volume with reduce_prod, and concatenate the
# result with the first 2 dimensions.
# NOTE: The code here is *not* specific to images, but we use NCHW notation to make it more readable.
NC = graph.gather(input_shape, indices=[0, 1])
HW = graph.gather(input_shape, indices=[2, 3])
new_shape = graph.concat([NC, graph.reduce_prod(HW, axes=[0])])
partially_flattened = graph.reshape(graph.inputs[0], new_shape)

# Finally, set up the outputs and export.
flattened.name = "flattened"  # Rename output tensor to make it easy to find.
flattened.dtype = np.float32  # NOTE: We must include dtype information for graph outputs
partially_flattened.name = "partially_flattened"
partially_flattened.dtype = np.float32

graph.outputs = [flattened, partially_flattened]
onnx.save(gs.export_onnx(graph), "model.onnx")
