#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
One-time developer script to generate tests/models/sd_attention.onnx.

Generates a synthetic self-attention ONNX model that mirrors the architecture
of a Stable Diffusion 1.5 BasicTransformerBlock self-attention sub-block, using
SD 1.5 down-block level-0 dimensions:

    d_model = 320, n_heads = 8, d_head = 40, seq_len = 256, batch_size = 1

The model operates on a flat [B, seq_len, d_model] hidden-state tensor and
performs:

    q = hidden_states @ W_q          # [1, 256, 320]
    k = hidden_states @ W_k          # [1, 256, 320]
    v = hidden_states @ W_v          # [1, 256, 320]
    k_T = Transpose(k, [0,2,1])      # [1, 320, 256]  (seq_len at dim-2)
    A = MatMul(q, k_T)               # [1, 256, 256]
    P = Softmax(A, axis=2)           # [1, 256, 256]
    c = MatMul(P, v)                 # [1, 256, 320]
    out = c @ W_o + b_o              # [1, 256, 320]

All weight tensors are embedded as ONNX initializers so the model is fully
self-contained.  Shape information is propagated via onnx shape inference so
that the sharding tool can determine tensor ranks without the --kv-rank flag.

This script does NOT require diffusers or internet access; it constructs the
model directly via onnx-graphsurgeon.

Usage (run once from the repo root or the tests/models/ directory):
    python tests/models/generate_sd_attention.py
"""
import os
import numpy as np
import onnx
import onnx_graphsurgeon as gs

CURDIR = os.path.dirname(os.path.abspath(__file__))

# SD 1.5 down-block-0 self-attention dimensions
BATCH_SIZE = 1
SEQ_LEN = 256
D_MODEL = 320

# Use a fixed seed for reproducibility
np.random.seed(42)


def _make_model() -> onnx.ModelProto:
    dtype = np.float32
    scale = 0.01  # small weights → no overflow in softmax

    # --- weight initialisers ---
    W_q = gs.Constant("W_q", np.random.randn(D_MODEL, D_MODEL).astype(dtype) * scale)
    W_k = gs.Constant("W_k", np.random.randn(D_MODEL, D_MODEL).astype(dtype) * scale)
    W_v = gs.Constant("W_v", np.random.randn(D_MODEL, D_MODEL).astype(dtype) * scale)
    W_o = gs.Constant("W_o", np.random.randn(D_MODEL, D_MODEL).astype(dtype) * scale)
    b_o = gs.Constant("b_o", np.zeros(D_MODEL, dtype=dtype))

    # --- graph input ---
    hidden_states = gs.Variable(
        "hidden_states", dtype=np.float32, shape=[BATCH_SIZE, SEQ_LEN, D_MODEL]
    )

    # --- intermediate tensors (shapes set for rank inference) ---
    q = gs.Variable("q", dtype=np.float32, shape=[BATCH_SIZE, SEQ_LEN, D_MODEL])
    k = gs.Variable("k", dtype=np.float32, shape=[BATCH_SIZE, SEQ_LEN, D_MODEL])
    v = gs.Variable("v", dtype=np.float32, shape=[BATCH_SIZE, SEQ_LEN, D_MODEL])
    k_T = gs.Variable("k_T", dtype=np.float32, shape=[BATCH_SIZE, D_MODEL, SEQ_LEN])
    attn_w = gs.Variable(
        "attn_w", dtype=np.float32, shape=[BATCH_SIZE, SEQ_LEN, SEQ_LEN]
    )
    attn_p = gs.Variable(
        "attn_p", dtype=np.float32, shape=[BATCH_SIZE, SEQ_LEN, SEQ_LEN]
    )
    attn_o = gs.Variable(
        "attn_o", dtype=np.float32, shape=[BATCH_SIZE, SEQ_LEN, D_MODEL]
    )
    proj_o = gs.Variable(
        "proj_o", dtype=np.float32, shape=[BATCH_SIZE, SEQ_LEN, D_MODEL]
    )
    output = gs.Variable(
        "output", dtype=np.float32, shape=[BATCH_SIZE, SEQ_LEN, D_MODEL]
    )

    # --- nodes ---
    nodes = [
        gs.Node("MatMul", "q_proj", inputs=[hidden_states, W_q], outputs=[q]),
        gs.Node("MatMul", "k_proj", inputs=[hidden_states, W_k], outputs=[k]),
        gs.Node("MatMul", "v_proj", inputs=[hidden_states, W_v], outputs=[v]),
        gs.Node(
            "Transpose",
            "k_transpose",
            inputs=[k],
            outputs=[k_T],
            attrs={"perm": [0, 2, 1]},
        ),
        gs.Node("MatMul", "attn_mm1", inputs=[q, k_T], outputs=[attn_w]),
        gs.Node(
            "Softmax", "attn_sm", inputs=[attn_w], outputs=[attn_p], attrs={"axis": 2}
        ),
        gs.Node("MatMul", "attn_mm2", inputs=[attn_p, v], outputs=[attn_o]),
        gs.Node("MatMul", "out_proj", inputs=[attn_o, W_o], outputs=[proj_o]),
        gs.Node("Add", "out_bias", inputs=[proj_o, b_o], outputs=[output]),
    ]

    graph = gs.Graph(
        nodes=nodes,
        inputs=[hidden_states],
        outputs=[output],
        opset=17,
    )
    graph.cleanup()

    model = gs.export_onnx(graph)
    # Propagate shapes so the sharding tool sees tensor ranks on intermediates
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


def generate_sd_attention(out_path: str | None = None) -> str:
    if out_path is None:
        out_path = os.path.join(CURDIR, "sd_attention.onnx")
    model = _make_model()
    onnx.save(model, out_path)
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    generate_sd_attention()
