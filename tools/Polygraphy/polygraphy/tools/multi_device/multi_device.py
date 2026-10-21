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


@add_json_methods("dist collective configuration")
class DistCollective:
    def __init__(
        self,
        group_size=0,
        root=-1,
        nb_rank=1,
        groups=None,
        reduce_scatter_reduce_op="max",
    ):
        self.group_size = group_size
        self.root = root
        self.nb_rank = nb_rank
        self.groups = groups if groups else []
        self.reduce_scatter_reduce_op = reduce_scatter_reduce_op

    def _common_attrs(self):
        return {
            "group_size": self.group_size,
            "root": self.root,
            "nb_rank": self.nb_rank,
            # Commented out until empty list '[]' can be inferred to be onnx.AttributeProto.INTS
            # "groups" : self.groups,
        }

    def all_gather_attrs(self):
        return {
            **self._common_attrs(),
            **{"reduce_op": "none", "collective_operation": "all_gather"},
        }

    def all_reduce_attrs(self):
        return {
            **self._common_attrs(),
            **{"reduce_op": "sum", "collective_operation": "all_reduce"},
        }

    def reduce_scatter_attrs(self):
        return {
            **self._common_attrs(),
            **{
                "reduce_op": self.reduce_scatter_reduce_op,
                "collective_operation": "reduce_scatter",
            },
        }


@Encoder.register(DistCollective)
def encode(dist_collective):
    return {
        **dist_collective._common_attrs(),
        **{
            "reduce_op": dist_collective.reduce_scatter_reduce_op,
            "groups": dist_collective.groups,
        },
    }


@Decoder.register(DistCollective)
def decode(dct):
    return DistCollective(
        group_size=dct.get("group_size"),
        root=dct.get("root"),
        nb_rank=dct.get("nb_rank"),
        groups=dct.get("groups"),
        reduce_scatter_reduce_op=dct.get("reduce_op"),
    )


@add_json_methods("fused attention replacement")
class FusedAttention:
    def __init__(
        self, is_causal, q_shuffle=None, k_shuffle=None, v_shuffle=None, nb_rank=1
    ):
        self.is_causal = is_causal
        self.q_shuffle = q_shuffle
        self.k_shuffle = k_shuffle
        self.v_shuffle = v_shuffle
        self.nb_rank = nb_rank

    def make_attrs(self):
        return {
            "is_causal": int(self.is_causal),
            "scale": 1.0,
            "TRT_decomposable": 0,
            "nb_rank": int(self.nb_rank),
        }


@Encoder.register(FusedAttention)
def encode(fused):
    return {
        "is_causal": fused.is_causal,
        "q_shuffle": fused.q_shuffle,
        "k_shuffle": fused.k_shuffle,
        "v_shuffle": fused.v_shuffle,
        "nb_rank": fused.nb_rank,
    }


@Decoder.register(FusedAttention)
def decode(dct):
    return FusedAttention(
        is_causal=dct["is_causal"],
        q_shuffle=dct.get("q_shuffle"),
        k_shuffle=dct.get("k_shuffle"),
        v_shuffle=dct.get("v_shuffle"),
        nb_rank=dct.get("nb_rank"),
    )


@add_json_methods("shard tensor")
class ShardTensor:
    def __init__(self, name, seq_len_idx, rank=None):
        self.name = name
        self.seq_len_idx = seq_len_idx
        self.rank = rank


@Decoder.register(ShardTensor)
def decode(dct):
    return ShardTensor(
        name=dct["name"], seq_len_idx=dct["seq_len_idx"], rank=dct.get("rank")
    )


@Encoder.register(ShardTensor)
def encode(shard_tensor):
    return {
        "name": shard_tensor.name,
        "seq_len_idx": shard_tensor.seq_len_idx,
        "rank": shard_tensor.rank,
    }


@add_json_methods("attention layer hint")
class AttentionLayerHint:
    def __init__(self, q, gather_kv, gather_q, replace=None):
        self.q = q
        self.gather_kv = gather_kv
        self.gather_q = gather_q
        self.replace = replace


@Decoder.register(AttentionLayerHint)
def decode(dct):
    return AttentionLayerHint(
        q=dct["q"],
        gather_kv=dct["gather_kv"],
        gather_q=dct["gather_q"],
        replace=dct.get("replace"),
    )


@Encoder.register(AttentionLayerHint)
def encode(attention_layer_hint):
    return {
        "q": attention_layer_hint.q,
        "gather_kv": attention_layer_hint.gather_kv,
        "gather_q": attention_layer_hint.gather_q,
        "replace": attention_layer_hint.replace,
    }


@add_json_methods("shard hints")
class ShardHints:
    def __init__(
        self,
        parallelism,
        attention_layers,
        dist_collectives,
        inputs,
        outputs,
        k_seq_len_idx,
        v_seq_len_idx,
        kv_rank,
    ):
        self.parallelism = parallelism
        self.attention_layers = attention_layers
        self.dist_collectives = dist_collectives
        self.inputs = inputs
        self.outputs = outputs
        self.k_seq_len_idx = k_seq_len_idx
        self.v_seq_len_idx = v_seq_len_idx
        self.kv_rank = kv_rank


@Decoder.register(ShardHints)
def decode(dct):
    return ShardHints(
        parallelism=dct["parallelism"],
        attention_layers=dct["attention_layers"],
        dist_collectives=dct["dist_collectives"],
        inputs=dct["inputs"],
        outputs=dct["outputs"],
        k_seq_len_idx=dct["k_seq_len_idx"],
        v_seq_len_idx=dct["v_seq_len_idx"],
        kv_rank=dct["kv_rank"],
    )


@Encoder.register(ShardHints)
def encode(shard_hints):
    return {
        "parallelism": shard_hints.parallelism,
        "attention_layers": shard_hints.attention_layers,
        "dist_collectives": shard_hints.dist_collectives,
        "inputs": shard_hints.inputs,
        "outputs": shard_hints.outputs,
        "k_seq_len_idx": shard_hints.k_seq_len_idx,
        "v_seq_len_idx": shard_hints.v_seq_len_idx,
        "kv_rank": shard_hints.kv_rank,
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


def get_attention_pattern_alt():

    pattern = gs.GraphPattern()
    q = pattern.variable()
    k = pattern.variable()
    v = pattern.variable()

    slice_starts = pattern.variable()
    slice_ends = pattern.variable()
    div_weights = pattern.variable()

    shape = pattern.add("shape", "Shape", inputs=[q])
    slice_node = pattern.add("slice", "Slice", inputs=[shape, slice_starts, slice_ends])

    cast_1 = pattern.add("cast_1", "Cast", inputs=[slice_node])
    sqrt_1 = pattern.add("sqrt_1", "Sqrt", inputs=[cast_1])
    cast_2 = pattern.add("cast_2", "Cast", inputs=[sqrt_1])
    div = pattern.add("div", "Div", inputs=[div_weights, cast_2])
    cast_3 = pattern.add("cast_3", "Cast", inputs=[div])

    sqrt_2 = pattern.add("sqrt_2", "Sqrt", inputs=[cast_3])
    sqrt_3 = pattern.add("sqrt_3", "Sqrt", inputs=[cast_3])
    mul_1 = pattern.add("mul_1", "Mul", inputs=[q, sqrt_2])
    mul_2 = pattern.add("mul_2", "Mul", inputs=[k, sqrt_3])

    matmul_1 = pattern.add("MatMul1", "MatMul", inputs=[mul_1, mul_2])
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

        return "Multi-Device Subtools", [Shard()]
