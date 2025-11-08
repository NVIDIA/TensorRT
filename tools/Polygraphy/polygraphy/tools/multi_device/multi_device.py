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
from polygraphy import mod
from polygraphy.json import Decoder, Encoder, add_json_methods
from polygraphy.tools.base import Tool

gs = mod.lazy_import("onnx_graphsurgeon")

@add_json_methods("shard tensor")
class ShardTensor:
    def __init__(self, name, seq_len_idx, rank = None):
        self.name = name
        self.seq_len_idx = seq_len_idx
        self.rank = rank

@Decoder.register(ShardTensor)
def decode(dct):
    return ShardTensor(
        name = dct["name"],
        seq_len_idx=dct["seq_len_idx"],
        rank=dct.get("rank")
    )

@Encoder.register(ShardTensor)
def encode(shard_tensor):
    return {
        "name" : shard_tensor.name,
        "seq_len_idx" : shard_tensor.seq_len_idx,
        "rank" : shard_tensor.rank
    }

@add_json_methods("attention layer hint")
class AttentionLayerHint:
    def __init__(self, q, gather_kv, gather_q):
        self.q = q
        self.gather_kv = gather_kv
        self.gather_q = gather_q

@Decoder.register(AttentionLayerHint)
def decode(dct):
    return AttentionLayerHint(
        q=dct["q"],
        gather_kv=dct["gather_kv"],
        gather_q=dct["gather_q"],
    )

@Encoder.register(AttentionLayerHint)
def encode(attention_layer_hint):
    return {
        "q" : attention_layer_hint.q,
        "gather_kv" : attention_layer_hint.gather_kv,
        "gather_q" : attention_layer_hint.gather_q,
    }

@add_json_methods("shard hints")
class ShardHints:
    def __init__(self, parallelism, group_size, root, groups, attention_layers, inputs, outputs, k_seq_len_idx, v_seq_len_idx, kv_rank, scatter_op):
        self.parallelism = parallelism
        self.group_size = group_size
        self.root = root
        self.groups = groups
        self.attention_layers = attention_layers
        self.inputs = inputs
        self.outputs = outputs
        self.k_seq_len_idx = k_seq_len_idx
        self.v_seq_len_idx = v_seq_len_idx
        self.kv_rank = kv_rank
        self.scatter_op = scatter_op

@Decoder.register(ShardHints)
def decode(dct):
    return ShardHints(
        parallelism=dct["parallelism"],
        group_size=dct["group_size"],
        root=dct["root"],
        groups=dct["groups"],
        attention_layers=dct["attention_layers"],
        inputs=dct["inputs"],
        outputs=dct["outputs"],
        k_seq_len_idx=dct["k_seq_len_idx"],
        v_seq_len_idx=dct["v_seq_len_idx"],
        kv_rank=dct["kv_rank"],
        scatter_op=dct["reduce_scatter_reduce_op"],
    )

@Encoder.register(ShardHints)
def encode(shard_hints):
    return {
        "parallelism" : shard_hints.parallelism,
        "group_size" : shard_hints.group_size,
        "root" : shard_hints.root,
        "groups" : shard_hints.groups,
        "attention_layers" : shard_hints.attention_layers,
        "inputs" : shard_hints.inputs,
        "outputs" : shard_hints.outputs,
        "k_seq_len_idx" : shard_hints.k_seq_len_idx,
        "v_seq_len_idx" : shard_hints.v_seq_len_idx,
        "kv_rank" : shard_hints.kv_rank,
        "reduce_scatter_reduce_op" : shard_hints.scatter_op,
    }

def get_attention_pattern():
    """
    Returns the pattern for canonical attention layers.

    Attention layers follow the pattern:

    Q    K
    |    |
    MatMul
      |
    SoftMax
      |
      |  V
      |  |
    MatMul
      |
    Output
    """
        
    pattern = gs.GraphPattern()
    q = pattern.variable()
    k = pattern.variable()
    v = pattern.variable()

    matmul_1 = pattern.add("MatMul1", "MatMul", inputs=[q, k])
    softmax = pattern.add("Softmax", "Softmax", inputs=[matmul_1])
    matmul_2 = pattern.add("MatMul2", "MatMul", inputs=[softmax, v])
    pattern.set_output_tensors([matmul_2])
    return pattern

class MultiDevice(Tool):
    """
    Multi-Device related operations on an onnx model.
    """

    def __init__(self):
        super().__init__("multi-device")
    
    def get_subtools_impl(self):
        # Avoid circular dependency
        from polygraphy.tools.multi_device.subtool.shard import Shard

        return "Multi-Device Subtools", [
            Shard()
        ]
