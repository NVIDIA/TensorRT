#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Construct a self-attention ONNX model using the ONNX GraphSurgeon layer API.

The model implements:
  1. Q/K/V linear projections (MatMul with 4096x4096 weights)
  2. Reshape to multi-head layout (seq, batch, 4096) -> (seq, batch, 32, 128)
  3. RMSNorm on Q and K
  4. Scaled dot-product attention (QK^T / sqrt(d), softmax, attn * V)
  5. Reshape back and output projection

Input/Output: (sequence_length, batch_size, 4096), float16
"""

import argparse
import math

import numpy as np
import onnx
import onnx_graphsurgeon as gs

NUM_HEADS = 32
HEAD_DIM = 128
HIDDEN_DIM = NUM_HEADS * HEAD_DIM  # 4096
OPSET = 17


# Register ONNX ops as methods on gs.Graph using the layer API.
# Each returns the output tensor(s) directly for easy chaining.

@gs.Graph.register()
def matmul(self, a, b):
    return self.layer(op="MatMul", inputs=[a, b], outputs=["matmul_out"])[0]


@gs.Graph.register()
def transpose(self, a, perm):
    return self.layer(op="Transpose", inputs=[a], attrs={"perm": perm}, outputs=["transpose_out"])[0]


@gs.Graph.register()
def reshape(self, data, shape):
    return self.layer(op="Reshape", inputs=[data, shape], attrs={"allowzero": 0}, outputs=["reshape_out"])[0]


@gs.Graph.register()
def softmax(self, a, axis=-1):
    return self.layer(op="Softmax", inputs=[a], attrs={"axis": axis}, outputs=["softmax_out"])[0]


@gs.Graph.register()
def cast(self, a, to):
    return self.layer(op="Cast", inputs=[a], attrs={"to": to}, outputs=["cast_out"])[0]


@gs.Graph.register()
def sqrt(self, a):
    return self.layer(op="Sqrt", inputs=[a], outputs=["sqrt_out"])[0]


@gs.Graph.register()
def add(self, a, b):
    return self.layer(op="Add", inputs=[a, b], outputs=["add_out"])[0]


@gs.Graph.register()
def mul(self, a, b):
    return self.layer(op="Mul", inputs=[a, b], outputs=["mul_out"])[0]


@gs.Graph.register()
def div(self, a, b):
    return self.layer(op="Div", inputs=[a, b], outputs=["div_out"])[0]


@gs.Graph.register()
def pow(self, a, b):
    return self.layer(op="Pow", inputs=[a, b], outputs=["pow_out"])[0]


@gs.Graph.register()
def reduce_mean(self, a, axes, keepdims=1):
    return self.layer(
        op="ReduceMean", inputs=[a], attrs={"axes": axes, "keepdims": keepdims},
        outputs=["reduce_mean_out"],
    )[0]


@gs.Graph.register()
def shape_op(self, a):
    return self.layer(op="Shape", inputs=[a], outputs=["shape_out"])[0]


@gs.Graph.register()
def gather(self, data, indices):
    return self.layer(op="Gather", inputs=[data, indices], attrs={"axis": 0}, outputs=["gather_out"])[0]


@gs.Graph.register()
def unsqueeze(self, a, axes):
    return self.layer(op="Unsqueeze", inputs=[a, axes], outputs=["unsqueeze_out"])[0]


@gs.Graph.register()
def concat(self, inputs, axis=0):
    return self.layer(op="Concat", inputs=inputs, attrs={"axis": axis}, outputs=["concat_out"])[0]


def build_attention_graph():
    """Build the full self-attention ONNX graph."""
    rng = np.random.default_rng(42)
    graph = gs.Graph(opset=OPSET)

    def fp16_weights(shape):
        return rng.standard_normal(shape).astype(np.float16)

    def fp32_scalar(val):
        return np.array([val], dtype=np.float32)

    axes_0 = np.array([0], dtype=np.int64)

    # Input: (seq, batch, 4096) fp16
    graph_input = gs.Variable("input", dtype=np.float16, shape=["sequence_length", "batch_size", HIDDEN_DIM])
    graph.inputs = [graph_input]

    # Q/K/V projections
    q_proj = graph.matmul(graph_input, fp16_weights((HIDDEN_DIM, HIDDEN_DIM)))
    k_proj = graph.matmul(graph_input, fp16_weights((HIDDEN_DIM, HIDDEN_DIM)))
    v_proj = graph.matmul(graph_input, fp16_weights((HIDDEN_DIM, HIDDEN_DIM)))

    # Dynamic reshape: (seq, batch, 4096) -> (seq, batch, 32, 128)
    # Build target shape [seq_dim, batch_dim, 32, 128] from input shape
    def reshape_to_heads(proj):
        inp_shape = graph.shape_op(proj)
        seq_dim = graph.unsqueeze(graph.gather(inp_shape, np.array(0, dtype=np.int64)), axes_0)
        batch_dim = graph.unsqueeze(graph.gather(inp_shape, np.array(1, dtype=np.int64)), axes_0)
        target_shape = graph.concat([
            seq_dim, batch_dim,
            np.array([NUM_HEADS], dtype=np.int64),
            np.array([HEAD_DIM], dtype=np.int64),
        ])
        return graph.reshape(proj, target_shape)

    q_4d = reshape_to_heads(q_proj)
    k_4d = reshape_to_heads(k_proj)
    v_4d = reshape_to_heads(v_proj)

    # RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
    def rmsnorm(x):
        x_fp32 = graph.cast(x, onnx.TensorProto.FLOAT)
        sq = graph.pow(x_fp32, fp32_scalar(2.0))
        mean = graph.reduce_mean(sq, axes=[-1])
        rms = graph.sqrt(graph.add(mean, fp32_scalar(1e-6)))
        inv_rms = graph.div(fp32_scalar(1.0), rms)
        normed = graph.mul(x_fp32, inv_rms)
        normed_fp16 = graph.cast(normed, onnx.TensorProto.FLOAT16)
        weight = rng.standard_normal((1, 1, 1, HEAD_DIM)).astype(np.float16)
        return graph.mul(weight, normed_fp16)

    q_norm = rmsnorm(q_4d)
    k_norm = rmsnorm(k_4d)

    # Transpose to attention layout: (seq, batch, heads, hdim) -> (batch, heads, seq, hdim)
    q_attn = graph.transpose(q_norm, perm=[1, 2, 0, 3])
    k_attn = graph.transpose(k_norm, perm=[1, 2, 0, 3])
    v_attn = graph.transpose(v_4d, perm=[1, 2, 0, 3])

    # Dynamic reshape Q/K/V to (batch, heads, -1, hdim) using shape extraction
    def reshape_attn(x):
        s = graph.shape_op(x)
        batch = graph.unsqueeze(graph.gather(s, np.array(0, dtype=np.int64)), axes_0)
        heads = graph.unsqueeze(graph.gather(s, np.array(1, dtype=np.int64)), axes_0)
        hdim = graph.unsqueeze(graph.gather(s, np.array(3, dtype=np.int64)), axes_0)
        target = graph.concat([batch, heads, np.array([-1], dtype=np.int64), hdim])
        return graph.reshape(x, target)

    q_r = reshape_attn(q_attn)
    k_r = reshape_attn(k_attn)
    v_r = reshape_attn(v_attn)

    # Scale: split sqrt(1/sqrt(head_dim)) across Q and K
    scale_val = math.sqrt(math.sqrt(1.0 / HEAD_DIM))
    scale_fp16 = np.array([scale_val], dtype=np.float16)
    q_scaled = graph.mul(q_r, scale_fp16)
    q_scaled.name = "q_scaled"

    # Transpose K: (batch, heads, seq, hdim) -> (batch, heads, hdim, seq)
    k_t = graph.transpose(k_r, perm=[0, 1, 3, 2])
    k_scaled = graph.mul(k_t, scale_fp16)

    # QK^T -> Softmax -> Attn*V
    qk = graph.matmul(q_scaled, k_scaled)
    attn_weights = graph.softmax(qk, axis=-1)
    attn_out = graph.matmul(attn_weights, v_r)

    # Reshape back: (batch, heads, seq, hdim) -> (seq, batch, 4096)
    attn_t = graph.transpose(attn_out, perm=[2, 0, 1, 3])
    attn_shape = graph.shape_op(attn_t)
    seq_dim = graph.unsqueeze(graph.gather(attn_shape, np.array(0, dtype=np.int64)), axes_0)
    batch_dim = graph.unsqueeze(graph.gather(attn_shape, np.array(1, dtype=np.int64)), axes_0)
    # Compute heads * hdim
    heads_dim = graph.gather(attn_shape, np.array(2, dtype=np.int64))
    hdim_dim = graph.gather(attn_shape, np.array(3, dtype=np.int64))
    hidden = graph.unsqueeze(graph.mul(heads_dim, hdim_dim), axes_0)
    flat_shape = graph.concat([seq_dim, batch_dim, hidden])
    attn_flat = graph.reshape(attn_t, flat_shape)

    # Output projection
    output = graph.matmul(attn_flat, fp16_weights((HIDDEN_DIM, HIDDEN_DIM)))
    output.name = "output"
    output.dtype = np.float16
    output.shape = ["sequence_length", "batch_size", HIDDEN_DIM]
    graph.outputs = [output]

    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    model.ir_version = 8

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Generate attention_sd.onnx model using the ONNX GraphSurgeon layer API"
    )
    parser.add_argument(
        "--output", type=str, default="attention_sd.onnx",
        help="Output ONNX file path (default: attention_sd.onnx)",
    )
    args = parser.parse_args()

    model = build_attention_graph()
    onnx.save(model, args.output)

    print(f"Saved model to {args.output}")
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  Initializers: {len(model.graph.initializer)}")
    print(f"  Opset: {model.opset_import[0].version}")


if __name__ == "__main__":
    main()
