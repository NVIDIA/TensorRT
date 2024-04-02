#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
from typing import List

import onnx_graphsurgeon as gs
import numpy as np
import onnx


##########################################################################################################
# Register functions to simplify the graph building process later on.

opset = 18


@gs.Graph.register()
def add(self, lhs, rhs):
    out = self.layer(op="Add", inputs=[lhs, rhs], outputs=["add_out"])[0]
    out.dtype = lhs.dtype
    return out


@gs.Graph.register()
def div(self, lhs, rhs):
    out = self.layer(op="Div", inputs=[lhs, rhs], outputs=["div_out"])[0]
    out.dtype = lhs.dtype
    return out


@gs.Graph.register()
def matmul(self, lhs, rhs):
    out = self.layer(op="MatMul", inputs=[lhs, rhs], outputs=["matmul_out"])[0]
    out.dtype = lhs.dtype
    return out


@gs.Graph.register()
def constant_tensor_ref(self, ref_name, dtype):
    attr_ref = gs.Node.AttributeRef(ref_name, gs.Tensor)
    out = self.layer(
        op="Constant", outputs=["constant_out"], attrs={"value": attr_ref}
    )[0]
    out.dtype = dtype
    return out


@gs.Graph.register()
def constant_float_ref(self, ref_name):
    attr_ref = gs.Node.AttributeRef(ref_name, float)
    out = self.layer(
        op="Constant", outputs=["constant_out"], attrs={"value_float": attr_ref}
    )[0]
    out.dtype = np.float32
    return out


@gs.Graph.register()
def transpose(self, t, perm=[]):
    out = self.layer(
        op="Transpose", inputs=[t], outputs=["transpose_out"], attrs={"perm": perm}
    )[0]
    out.dtype = t.dtype
    return out


@gs.Graph.register()
def relu(self, t):
    out = self.layer(op="Relu", inputs=[t], outputs=["relu_out"])[0]
    out.dtype = t.dtype
    return out


@gs.Graph.register()
def softmax(self, t):
    out = self.layer(op="Softmax", inputs=[t], outputs=["softmax_out"])[0]
    out.dtype = t.dtype
    return out


##########################################################################################################
# Create a Function representing Attention the same way you would create a gs.Graph.
# Tensors created here are not reused outside of this Function.

attn_input_embeds = gs.Variable(
    "input", dtype=np.float32, shape=("batch", "seqlen", "emb_dim")
)
attn_attrs = {
    "Wq": None,
    "Wk": None,
    "Wv": None,
    "transpose_perm": [0, 2, 1],
    "sqrt_emb_dim": 1.0,
}
attn = gs.Function("SelfAttention", inputs=[attn_input_embeds], attrs=attn_attrs)
attn_Q = attn.matmul(
    attn_input_embeds, attn.constant_tensor_ref("Wq", dtype=np.float32)
)
attn_K = attn.matmul(
    attn_input_embeds, attn.constant_tensor_ref("Wk", dtype=np.float32)
)
attn_V = attn.matmul(
    attn_input_embeds, attn.constant_tensor_ref("Wv", dtype=np.float32)
)
attn_sqrt_emb_dim = attn.constant_float_ref("sqrt_emb_dim")
attn_perm = gs.Node.AttributeRef("transpose_perm", List[int])
attn_matrix = attn.div(
    attn.matmul(attn_Q, attn.transpose(attn_K, perm=attn_perm)), attn_sqrt_emb_dim
)
attn.outputs = [attn.matmul(attn.softmax(attn_matrix), attn_V)]
attn.opset = opset

##########################################################################################################
# Use the Function in a model.

# Model parameters
emb_dim = 4
n_layers = 4


def make_attention_attrs():
    return {
        "sqrt_emb_dim": float(math.sqrt(emb_dim)),
        "Wq": gs.Constant("Wq", np.random.randn(emb_dim, emb_dim).astype(np.float32)),
        "Wk": gs.Constant("Wk", np.random.randn(emb_dim, emb_dim).astype(np.float32)),
        "Wv": gs.Constant("Wv", np.random.randn(emb_dim, emb_dim).astype(np.float32)),
    }


# Build graph with n_layers attention blocks.
input_embeds = gs.Variable(
    "input_embeds", dtype=np.float32, shape=("batch", "seqlen", emb_dim)
)
graph = gs.Graph(inputs=[input_embeds], functions=[attn])
out = input_embeds
for _ in range(n_layers):
    next = graph.SelfAttention(inputs=[out], attrs=make_attention_attrs())[0]
    out = graph.add(out, graph.relu(next))
out.shape = input_embeds.shape
graph.outputs = [out]
graph.opset = opset

# Save graph
model = gs.export_onnx(graph)
onnx.save(model, "model.onnx")
