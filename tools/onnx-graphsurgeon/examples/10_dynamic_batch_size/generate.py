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


##########################################################################################################
# Register functions to simplify the graph building process later on.

@gs.Graph.register()
def conv(self, inp, weights, dilations, group, strides):
    out = self.layer(
        op="Conv",
        inputs=[inp, weights],
        outputs=["conv_out"],
        attrs={"dilations": dilations, "group": group, "kernel_shape": weights.shape[2:], "strides": strides},
    )[0]
    out.dtype = inp.dtype
    return out


@gs.Graph.register()
def reshape(self, data, shape):
    out = self.layer(op="Reshape", inputs=[data, shape], outputs=["reshape_out"])[0]
    out.dtype = data.dtype
    return out


@gs.Graph.register()
def matmul(self, lhs, rhs):
    out = self.layer(op="MatMul", inputs=[lhs, rhs], outputs=["matmul_out"])[0]
    out.dtype = lhs.dtype
    return out

##########################################################################################################


# Set input
X = gs.Variable(name="input_1", dtype=np.float32, shape=(1, 3, 28, 28))
graph = gs.Graph(inputs=[X])

# Connect intermediate tensors
conv_out = graph.conv(
    X, weights=np.ones(shape=(32, 3, 3, 3), dtype=np.float32), dilations=[1, 1], group=1, strides=[1, 1]
)
reshape_out = graph.reshape(conv_out, np.array([1, 21632], dtype=np.int64))
matmul_out = graph.matmul(reshape_out, np.ones(shape=(21632, 10), dtype=np.float32))

# Set output
graph.outputs = [matmul_out]

# Save graph
onnx.save(gs.export_onnx(graph), "model.onnx")
