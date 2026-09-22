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

import argparse
import copy
from polygraphy import mod
from polygraphy.logger.logger import G_LOGGER
from polygraphy.tools.args.backend.onnx.loader import OnnxInferShapesArgs, OnnxLoadArgs
from polygraphy.tools.base import Tool
from polygraphy.tools.args import ModelArgs
from polygraphy.tools.args import OnnxSaveArgs
from polygraphy.tools.multi_device.multi_device import FusedAttention
from polygraphy.tools.multi_device import (
    get_attention_pattern,
    get_attention_pattern_alt,
    ShardHints,
)
from polygraphy.tools.template.subtool.shard_hint import ShardHintArgs
import itertools
import os

onnx = mod.lazy_import("onnx>=1.18")
gs = mod.lazy_import("onnx_graphsurgeon")
onnx_backend = mod.lazy_import("polygraphy.backend.onnx")
np = mod.lazy_import("numpy")


class ShardableTensor:
    def __init__(
        self, tensor, graph, name_fn, dist_collective, seq_len_idx=0, rank=None
    ):
        self.graph = graph
        self.name_fn = name_fn
        self.chain = [tensor]
        self.idx = seq_len_idx
        self.rank = rank

        self.dist_collective = dist_collective

    def get_tail(self):
        return self.chain[-1]

    def _get_output_frontier(self, shard_tensor):
        output_frontier = []
        for node in self.graph.nodes:
            for tensor in node.inputs:
                if shard_tensor == tensor:
                    output_frontier.append(node)
        return output_frontier

    def _update_node_inputs(self, output_frontier, old_tensor, new_tensor):
        for node in output_frontier:
            for i, tensor in enumerate(node.inputs):
                if tensor == old_tensor:
                    node.inputs[i] = new_tensor

    def _update_tensor_outputs(self, tensor, new_node):
        tensor.outputs = [new_node]

    def _extend_node(self, op, attrs):
        """
        Add a node between the tail of this tensor chain and the nodes that consume it
        """

        # Get tensor at end of chain and what nodes it feeds to
        head_tensor = self.get_tail()
        G_LOGGER.info(f"Inserting {op} for tensor: {head_tensor.name}")
        output_frontier = self._get_output_frontier(head_tensor)

        # Make new output tensor
        tail_tensor = gs.Variable(
            name=self.name_fn(), shape=None, dtype=head_tensor.dtype
        )

        # Make tensor go to
        # T => (T -> N -> T)'
        node = gs.Node(
            name=self.name_fn(),
            op=op,
            inputs=[head_tensor],
            outputs=[tail_tensor],
            attrs=attrs,
        )
        self.graph.nodes.extend([node])

        # Move the chain forward where the new tail is the head tensor
        self.chain.append(tail_tensor)

        # Make existing nodes take the new tensor
        self._update_node_inputs(output_frontier, head_tensor, tail_tensor)

        # Make tensor feed to inserted node
        self._update_tensor_outputs(head_tensor, node)

        # If a model output was all gathered change that
        if head_tensor in self.graph.outputs:
            i = self.graph.outputs.index(head_tensor)
            self.graph.outputs[i] = tail_tensor

        return node, tail_tensor

    def extend_transpose(self, perm):
        return self._extend_node(op="Transpose", attrs={"perm": perm})

    def extend_reduce_scatter(self):
        return self._extend_node(
            op="DistCollective", attrs=self.dist_collective.reduce_scatter_attrs()
        )

    def extend_all_gather(self):
        return self._extend_node(
            op="DistCollective", attrs=self.dist_collective.all_gather_attrs()
        )

    def extend_all_reduce(self):
        return self._extend_node(
            op="DistCollective", attrs=self.dist_collective.all_reduce_attrs()
        )

    def _check_valid_idx_rank(self):
        if self.rank is None and (
            self.get_tail().shape is None or len(self.get_tail().shape) == 0
        ):
            G_LOGGER.critical(
                f"Transpose for tensor {self.get_tail().name} is missing rank information in tensor or configuration"
            )

        if self.rank is None:
            G_LOGGER.info(
                f"Specified rank for {self.get_tail().name} was missing but shape was provided in model"
            )
            self.rank = len(self.get_tail().shape)

        if self.idx >= self.rank:
            G_LOGGER.critical(
                f"Transpose idx for tensor {self.get_tail().name} is out of range for rank"
            )

    def _make_swap(self):
        # NCCL dist collective only supports 0th dimension so some dist collectives need a swap
        perm_rank = self.rank if self.rank else len(self.get_tail().shape)
        perm = [i for i in range(perm_rank)]
        perm[0] = self.idx
        perm[self.idx] = 0

        return perm

    def _maybe_extend_dist_collective(self, fn):
        if self.idx == 0 or self.idx is None:
            return fn()
        else:
            # Make sure perm will work
            self._check_valid_idx_rank()
            perm = self._make_swap()

            self.extend_transpose(perm)
            fn()
            return self.extend_transpose(perm)

    def maybe_extend_reduce_scatter_non_zero_idx(self):
        return self._maybe_extend_dist_collective(self.extend_reduce_scatter)

    def maybe_extend_all_gather_non_zero_idx(self):
        return self._maybe_extend_dist_collective(self.extend_all_gather)


class TPManager:
    def __init__(self, graph, nb_ranks, raw_initializers) -> None:
        self.nb_ranks = nb_ranks

        # Materialize generators so _get_tp_tensors and _get_cp_tensors can both iterate
        self.attentions = TPManager.tp_match_attention(graph)
        self.mlps = TPManager.tp_match_mlp(graph)

        self.graph = graph

        self.weights = {
            k: v for k, v in graph.tensors().items() if isinstance(v, gs.Constant)
        }

        # GraphSurgeon casts fp4 to float32 so we need to preserve
        # the original data type for checking later
        self.initializer_dtypes = {t.name: t.data_type for t in raw_initializers}

        self.sharded_weights = {i: dict() for i in range(self.nb_ranks)}

    def _insert_all_reduces(self, dist_collective, name_fn):
        for cp_tensor in map(
            lambda t: ShardableTensor(t, self.graph, name_fn, dist_collective),
            self._get_cp_tensors(),
        ):
            cp_tensor.extend_all_reduce()

    def _get_tensors(self, tensor_type):
        tensors = []
        for pattern in itertools.chain(*[self.attentions, self.mlps]):
            tensors.extend(tensor_type(pattern))
        return tensors

    def _get_tp_tensors(self):
        return list(
            map(
                lambda tensor_idx: WeightShardDescriptor(tensor_idx[0], tensor_idx[1]),
                self._get_tensors(lambda t: t.get_tp_tensors()),
            )
        )

    def _get_cp_tensors(self):
        return self._get_tensors(lambda t: t.get_cp_tensors())

    def _get_slice_indices(self, tp_tensor, rank):
        slice_size = tp_tensor.get_new_dim(self.nb_ranks)
        start = rank * slice_size
        end = start + slice_size

        return start, end

    def _slice_weights(self):
        self.fp4_weights = set()

        for tp_tensor in self._get_tp_tensors():
            weight = self.weights[tp_tensor.name]

            # GraphSurgeon packs fp4 in a way that doubles the byte size and breaks
            # export, so we have to skip loading it into GraphSurgeon and deal with slicing
            # once the graph is exported
            if self.initializer_dtypes[tp_tensor.name] == onnx.TensorProto.FLOAT4E2M1:
                self.fp4_weights.add(tp_tensor)
                continue

            # Load the weight of this tensor, safely since
            # we know its not fp4
            weight = weight.values

            # Get the slice this weight will use on rank i
            for i in range(0, self.nb_ranks):
                start, end = self._get_slice_indices(tp_tensor, i)

                # Get the slice
                if tp_tensor.shard_idx == 0:
                    weight_slice = weight[start:end, :].copy()
                elif tp_tensor.shard_idx == 1:
                    weight_slice = weight[:, start:end].copy()
                else:
                    G_LOGGER.critical(
                        f"Unsupported shard dimension for {tp_tensor.name}"
                    )

                if i == 0:
                    G_LOGGER.info(
                        f"{tp_tensor.name} going from {weight.shape} -> {weight_slice.shape}"
                    )

                self.sharded_weights[i].update({tp_tensor.name: weight_slice})

    # Workaround for graphPattern not handling complex patterns
    @staticmethod
    def _find_downstream_op(graph, tensor, op, max_depth=20):
        """
        Recursive DFS graph search for a specific op type, traversing in downstream direction through the dataflow graph.
        """

        return TPManager._find_op(
            graph, tensor, op, lambda n: n.inputs, lambda n: n.outputs, max_depth
        )

    @staticmethod
    def _find_upstream_op(graph, tensor, op, max_depth=20):
        """
        Recursive DFS graph search for a specific op type, traversing in upstream direction through the dataflow graph.
        """

        return TPManager._find_op(
            graph, tensor, op, lambda n: n.outputs, lambda n: n.inputs, max_depth
        )

    @staticmethod
    def _find_op(graph, tensor, op, node_fn, tensor_fn, max_depth):

        if max_depth == 0:
            return None

        nodes = [node for node in graph.nodes if tensor in node_fn(node)]
        for node in nodes:
            G_LOGGER.verbose(f"Checking node {node.name} has {op}")
            if node.op == op:
                G_LOGGER.info(f"Found node {node.name}")
                return node
            for inp in tensor_fn(node):
                G_LOGGER.verbose(f"Next tensor: {inp.name}")
                found = TPManager._find_op(
                    graph, inp, op, node_fn, tensor_fn, max_depth - 1
                )
                if found is not None:
                    return found

        return None

    @staticmethod
    def _get_quantized_weights(graph, tensor):
        dl_2 = TPManager._find_upstream_op(graph, tensor, "DequantizeLinear")
        dl_1 = TPManager._find_upstream_op(graph, dl_2.inputs[1], "DequantizeLinear")

        weight_f4 = dl_2.inputs[0]
        weight_f8_scale = dl_1.inputs[0]

        return weight_f4, weight_f8_scale

    # Get all attention layers
    @staticmethod
    def tp_match_attention(graph):
        # Get all attention plugins, which are the starting point for where TP sharding happens
        attention_plugins = [
            node for node in graph.nodes if node.op == "AttentionPlugin"
        ]
        attentions = []

        # Go up until you hit a matmul
        for attention in attention_plugins:

            # Get the either the concat where stacked QKV feed into attention,
            # or attention itself if concat isn't present
            start_node = (
                TPManager._find_upstream_op(graph, attention.inputs[0], "Concat", 2)
                or attention
            )

            # Get QKV and O from here
            q_matmul = TPManager._find_upstream_op(
                graph, start_node.inputs[0], "MatMul"
            )
            k_matmul = TPManager._find_upstream_op(
                graph, start_node.inputs[1], "MatMul"
            )
            v_matmul = TPManager._find_upstream_op(
                graph, start_node.inputs[2], "MatMul"
            )

            output_matmul = TPManager._find_downstream_op(
                graph, attention.outputs[0], "MatMul"
            )

            q_proj = q_matmul.inputs[1]
            k_proj = k_matmul.inputs[1]
            v_proj = v_matmul.inputs[1]
            o_proj = output_matmul.inputs[1]
            output = output_matmul.outputs[0]
            # Past key immediately follows stacked qkv or separate
            past = attention.inputs[1 if len(attention.inputs) == 6 else 3]
            present = attention.outputs[1]

            # Non quantized attention
            if all([isinstance(t, gs.Constant) for t in [q_proj, k_proj, v_proj]]):
                attentions.append(
                    TPAttentionWrapper(
                        attention,
                        [q_proj],
                        [k_proj],
                        [v_proj],
                        [o_proj],
                        output,
                        past,
                        present,
                    )
                )

            # Quantized Attention
            elif any([node.op == "DequantizeLinear" for node in graph.nodes]):
                attentions.append(
                    TPAttentionWrapper(
                        attention,
                        TPManager._get_quantized_weights(graph, q_proj),
                        TPManager._get_quantized_weights(graph, k_proj),
                        TPManager._get_quantized_weights(graph, v_proj),
                        TPManager._get_quantized_weights(graph, o_proj),
                        output,
                        past,
                        present,
                        quantized=True,
                    )
                )
            else:
                G_LOGGER.critical("Graph is missing TP-shardable attentions")

        return attentions

    # Get all MLPs
    @staticmethod
    def tp_match_mlp(graph):
        sigmoid_nodes = [node for node in graph.nodes if node.op == "Sigmoid"]
        mlps = []

        for sigmoid in sigmoid_nodes:
            gate_matmul = TPManager._find_upstream_op(
                graph, sigmoid.inputs[0], "MatMul"
            )
            mul = TPManager._find_downstream_op(
                graph,
                TPManager._find_downstream_op(graph, sigmoid.outputs[0], "Mul").outputs[
                    0
                ],
                "Mul",
            )
            up_matmul = TPManager._find_upstream_op(graph, mul.inputs[1], "MatMul")
            down_matmul = TPManager._find_downstream_op(graph, mul.outputs[0], "MatMul")

            gate_proj = gate_matmul.inputs[1]
            up_proj = up_matmul.inputs[1]
            down_proj = down_matmul.inputs[1]
            output = down_matmul.outputs[0]

            if all(
                [isinstance(t, gs.Constant) for t in [up_proj, down_proj, gate_proj]]
            ):
                mlps.append(TPMLPWrapper([up_proj], [down_proj], [gate_proj], output))
            elif any([node.op == "DequantizeLinear" for node in graph.nodes]):
                mlps.append(
                    TPMLPWrapper(
                        TPManager._get_quantized_weights(graph, up_proj),
                        TPManager._get_quantized_weights(graph, down_proj),
                        TPManager._get_quantized_weights(graph, gate_proj),
                        output,
                        quantized=True,
                    )
                )
            else:
                G_LOGGER.critical("Graph is missing TP-shardable MLPs")

        return mlps

    def _update_attentions(self):
        for attention in self.attentions:
            attention.update_attention_attrs(self.nb_ranks)
            attention.update_kv_cache(self.nb_ranks)

    def tp_shard(self, dist_collective, name_fn):
        # Create map to match rank to slice of original weights
        self._slice_weights()

        # Add all reduces to each CP tensor
        self._insert_all_reduces(dist_collective, name_fn)

        # Update attentions and kv caches
        self._update_attentions()

    def get_rank_graph(self, rank):
        tensors = self.graph.tensors()

        for name, weight_slice in self.sharded_weights[rank].items():
            tensors[name].values = weight_slice

        return self.graph

    def cache_fp4_raw_data(self, onnx_model):
        """
        Cache the original weights for FP4 tensors from ONNX
        before any per-rank mutation occurs
        """
        self.fp4_raw_data = {}
        initializer_map = {t.name: t for t in onnx_model.graph.initializer}
        for fp4_tensor in self.fp4_weights:
            self.fp4_raw_data[fp4_tensor.name] = bytes(
                initializer_map[fp4_tensor.name].raw_data
            )

    def slice_fp4(self, model, rank):
        """
        Shard Fp4 tensors via directly operating on ONNX initializers
        """
        for fp4_tensor in self.fp4_weights:

            onnx_tensor = [
                t for t in model.graph.initializer if t.name == fp4_tensor.name
            ][0]

            start, end = self._get_slice_indices(fp4_tensor, rank)

            # Since fp4 is being processed as an 8 bit value,
            # the new logical shape is half as big in the last dimension
            packed_shape = list(fp4_tensor.shape)
            packed_shape[-1] //= 2

            raw_data = np.frombuffer(
                self.fp4_raw_data[fp4_tensor.name], dtype=np.uint8
            ).reshape(packed_shape)

            if fp4_tensor.shard_idx == 0:
                raw_data_slice = raw_data[start:end, :]

            # If we shard in the second dimension, we need to scale
            # by a factor of two to account for the packed shape
            elif fp4_tensor.shard_idx == 1:
                raw_data_slice = raw_data[:, start // 2 : end // 2]
            else:
                G_LOGGER.critical(f"Unsupported dimension for fp4 sharding")

            onnx_tensor.raw_data = np.ascontiguousarray(raw_data_slice).tobytes()
            del onnx_tensor.dims[:]
            onnx_tensor.dims.extend(list(fp4_tensor.get_new_shape(self.nb_ranks)))


class WeightShardDescriptor:
    def __init__(self, tensor, shard_idx):
        if tensor.shape is None:
            G_LOGGER.critical(
                f"Weights need defined shape for TP sharding. {tensor.name} is missing shape"
            )

        if type(tensor.shape[shard_idx]) is not int:
            G_LOGGER.critical(
                f"Weight {tensor.name} shard dimension should be int, got {type(tensor.shape[shard_idx])}"
            )

        self.shape = tensor.shape
        self.name = tensor.name
        self.shard_idx = shard_idx

    def get_new_dim(self, nb_ranks):
        if self.shape[self.shard_idx] % nb_ranks != 0:
            G_LOGGER.critical(
                f"{self.name} cannot be TP sharded on {nb_ranks} ranks with shape {self.shape}"
            )
        return self.shape[self.shard_idx] // nb_ranks

    def get_new_shape(self, nb_ranks):
        new_shape = copy.deepcopy(self.shape)
        new_shape[self.shard_idx] = self.get_new_dim(nb_ranks)
        return new_shape


class TPMLPWrapper:
    def __init__(self, up_proj, down_proj, gate_proj, output, quantized=False):
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.gate_proj = gate_proj
        self.output = output
        self.quantized = quantized

    def get_tp_tensors(self):
        return (
            [(u, 0 if self.quantized else 1) for u in self.up_proj]
            + [(d, 1 if self.quantized else 0) for d in self.down_proj]
            + [(g, 0 if self.quantized else 1) for g in self.gate_proj]
        )

    def get_cp_tensors(self):
        return [self.output]


class TPAttentionWrapper:
    def __init__(
        self,
        attention,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        output,
        past,
        present,
        quantized=False,
    ):
        self.attention = attention
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.output = output
        self.quantized = quantized

        self.past_kv_cache = past
        self.present_kv_cache = present

        self.q_heads = int(self.attention.attrs["num_q_heads"])
        self.kv_heads = int(self.attention.attrs["num_kv_heads"])

    def get_tp_tensors(self):
        return (
            [(q, 0 if self.quantized else 1) for q in self.q_proj]
            + [(k, 0 if self.quantized else 1) for k in self.k_proj]
            + [(v, 0 if self.quantized else 1) for v in self.v_proj]
            + [(o, 1 if self.quantized else 0) for o in self.o_proj]
        )

    def get_cp_tensors(self):
        return [self.output]

    def update_attention_attrs(self, nb_ranks):
        if (self.q_heads % nb_ranks != 0) or (self.kv_heads % nb_ranks != 0):
            G_LOGGER.critical(
                f"Q heads or KV heads are not evenly divisible by {nb_ranks}"
            )

        self.attention.attrs["num_q_heads"] = self.q_heads // nb_ranks
        self.attention.attrs["num_kv_heads"] = self.kv_heads // nb_ranks

    def update_kv_cache(self, nb_ranks):
        if isinstance(self.past_kv_cache.shape[2], int):
            self.past_kv_cache.shape[2] = int(self.past_kv_cache.shape[2] // nb_ranks)
        if isinstance(self.present_kv_cache.shape[2], int):
            self.present_kv_cache.shape[2] = int(
                self.present_kv_cache.shape[2] // nb_ranks
            )


class Shard(Tool):
    """
    Convert a SD model to a MD model using a sharding hints file.
    """

    def __init__(self):
        super().__init__("shard")

    def _shard_inputs(self, graph):
        tensors = graph.tensors()
        effective_tensors = {tensor: tensor for tensor in tensors}

        for input in self.hints.inputs:
            _, tensor_md = ShardableTensor(
                tensors[input.name],
                graph,
                self._make_name,
                self.hints.dist_collectives,
                input.seq_len_idx,
                input.rank,
            ).maybe_extend_reduce_scatter_non_zero_idx()

            # In case this feeds into attention layer, keep track of new tensor
            effective_tensors.update({input.name: tensor_md.name})

        return effective_tensors

    def _shard_outputs(self, graph):
        tensors = graph.tensors()
        for output in self.hints.outputs:
            ShardableTensor(
                tensors[output.name],
                graph,
                self._make_name,
                self.hints.dist_collectives,
                output.seq_len_idx,
                output.rank,
            ).maybe_extend_all_gather_non_zero_idx()

    def _match_subgraphs(self, pattern, graph):
        attention_patterns = {0: get_attention_pattern, 1: get_attention_pattern_alt}

        # Get all attention layers that match supported pattern(s)
        pattern = attention_patterns[pattern]()
        matches = pattern.match_all(graph)

        return matches

    def _pair_matches_to_config(self, effective_tensors, matches):
        pairs = []
        for layer in self.hints.attention_layers:
            for match in matches:
                if match.inputs[0].name == effective_tensors[layer.q]:
                    pairs.append((layer, match))
        return pairs

    def _make_name(self, prefix="", suffix=""):
        """
        Return a unique name for node or tensor.
        """

        name = f"{prefix}{'_' if prefix else ''}shard_{self.shard_count}{'_' if suffix else ''}{suffix}"
        self.shard_count += 1
        return name

    @staticmethod
    def _cleanup(attention):
        """
        Remove nodes to be replaced from attention tensors
        """

        # Get Nodes
        mm_1 = attention["MatMul1"].onnx_node
        mm_2 = attention["MatMul2"].onnx_node

        # Complex Pattern
        if len(attention.inputs) == 6:
            q, k, v, _, _, _ = attention.inputs
        else:
            q, k, v = attention.inputs

        # Take out nodes
        # These nodes need to be removed from tensors
        q.outputs.remove(mm_1)
        k.outputs.remove(mm_1)
        v.outputs.remove(mm_2)

        o = attention.outputs[0]
        o.inputs.remove(mm_2)

        return [q, k, v], [o]

    def _make_fused_attention(self, graph, inputs, outputs, config):
        # Get tensors and shuffle if needed
        q, k, v = inputs
        q_shard = ShardableTensor(
            q, graph, self._make_name, self.hints.dist_collectives
        )
        k_shard = ShardableTensor(
            k,
            graph,
            self._make_name,
            self.hints.dist_collectives,
            self.hints.k_seq_len_idx,
            self.hints.kv_rank,
        )
        v_shard = ShardableTensor(
            v,
            graph,
            self._make_name,
            self.hints.dist_collectives,
            self.hints.k_seq_len_idx,
            self.hints.kv_rank,
        )

        incoming_inputs = []
        for shardable_tensor, shuffle in [
            (q_shard, config.replace.q_shuffle),
            (k_shard, config.replace.k_shuffle),
            (v_shard, config.replace.v_shuffle),
        ]:
            if shuffle:
                shardable_tensor.extend_transpose(shuffle)
            incoming_inputs.append(shardable_tensor.get_tail())

        # Add new attention node
        attention_node = gs.Node(
            op="Attention",
            name=self._make_name(),
            inputs=incoming_inputs,
            outputs=outputs,
            attrs=config.replace.make_attrs(),
        )
        graph.nodes.insert(0, attention_node)

    def _make_unfused_attention(self, graph, subgraph, config):
        # Prevents duplicate sharding
        sharded = set()
        gather_q = config.gather_q
        gather_kv = config.gather_kv

        # Get tensors directly
        q = subgraph["MatMul1"].onnx_node.inputs[0]
        k = subgraph["MatMul1"].onnx_node.inputs[1]
        v = subgraph["MatMul2"].onnx_node.inputs[1]

        # Insert collective ops as specified
        for i, (tensor, seq_len_idx, rank) in enumerate(
            [
                (q, None, None),
                (k, self.hints.k_seq_len_idx, self.hints.kv_rank),
                (v, self.hints.v_seq_len_idx, self.hints.kv_rank),
            ]
        ):
            if [gather_q, gather_kv, gather_kv][i] and tensor.name not in sharded:
                ShardableTensor(
                    tensor,
                    graph,
                    self._make_name,
                    self.hints.dist_collectives,
                    seq_len_idx,
                    rank,
                ).maybe_extend_all_gather_non_zero_idx()
                sharded.add(tensor.name)

        # Keep track of rare case where q == k or q == v and kv are gathered
        return (q == k or q == v) and config.gather_kv

    def _cleanup_and_replace(self, graph, subgraph, config):
        replace_type = type(config.replace)
        inputs, outputs = self._cleanup(subgraph)

        if replace_type not in self.replace_table:
            G_LOGGER.critical(f"Unsupported replace type {replace_type}")

        self.replace_table[replace_type](graph, inputs, outputs, config)

    def _context_parallel_shard(self, args, graph):
        # Check if final all gather is required
        final_all_gather = any(
            [layer.gather_q for layer in self.hints.attention_layers]
        )

        # Shard dependent inputs/outputs
        effective_tensors = self._shard_inputs(graph)

        # Get matching subgraphs
        matches = self._match_subgraphs(args.attention_pattern, graph)
        G_LOGGER.info(f"Found {len(matches)} attention pattern matches in the graph.")

        # Perform sharding
        for layer, match in self._pair_matches_to_config(effective_tensors, matches):
            G_LOGGER.info(
                f"Processing attention layer: q={layer.q}, gather_q={layer.gather_q}, gather_kv={layer.gather_kv}"
            )

            if not layer.replace:
                final_all_gather |= self._make_unfused_attention(graph, match, layer)
            else:
                self._cleanup_and_replace(graph, match, layer)

        # All gather outputs if partial Q
        if not final_all_gather:
            self._shard_outputs(graph)

        return graph

    def _tensor_parallel_shard(self, graph, raw_initializers):
        manager = TPManager(
            graph, self.hints.dist_collectives.nb_rank, raw_initializers
        )
        manager.tp_shard(self.hints.dist_collectives, self._make_name)
        return manager

    @staticmethod
    def graph_cleanup(graph):
        graph.cleanup()
        graph.toposort()

    def fix_group_attribute(self, model):
        # Manually add in groups attribute since graph surgeon doesn't support
        # type inference for an empty list, which is necessary for a group configuration of '[]'
        for node in model.graph.node:
            if node.op_type == "DistCollective":
                node.attribute.append(
                    onnx.helper.make_attribute(
                        "groups",
                        self.hints.dist_collectives.groups,
                        attr_type=onnx.AttributeProto.INTS,
                    )
                )

    def get_subscriptions_impl(self):
        return [
            ModelArgs(
                model_opt_required=True,
                input_shapes_opt_name=False,
                required_model_type="onnx",
            ),
            OnnxInferShapesArgs(),
            OnnxLoadArgs(outputs_opt_prefix=False, allow_shape_inference=True),
            OnnxSaveArgs(allow_shape_inference=False, output_opt_required=True),
            ShardHintArgs(),
        ]

    def add_parser_args_impl(self, parser):
        hint_source = parser.add_mutually_exclusive_group(required=True)
        hint_source.add_argument(
            "-s",
            "--hint",
            help="Hints file to describe shardable layers.",
            type=argparse.FileType("r"),
            dest="hint_file",
            default=None,
        )
        hint_source.add_argument(
            "--one-shot",
            help="Build ShardHints directly from the model (same logic as shard-hint) without loading/saving a JSON hints file.",
            action="store_true",
            dest="one_shot",
        )

    def run_impl(self, args):
        # Reset state for each run to avoid interference between tests
        self.shard_count = 0
        self.gather_output = False

        self.replace_table = {FusedAttention: self._make_fused_attention}

        onnx_model = self.arg_groups[OnnxLoadArgs].load_onnx()
        graph = onnx_backend.gs_from_onnx(onnx_model)

        seen_names = set()
        for node in graph.nodes:
            if node.name in seen_names:
                original = node.name
                new_name = self._make_name(prefix=original)
                G_LOGGER.warning(
                    f"Duplicate node name '{original}' renamed to '{new_name}' (op={node.op})"
                )
                node.name = new_name
            seen_names.add(node.name)

        if args.one_shot:
            G_LOGGER.info("Building sharding hints in one-shot from model (no JSON).")
            self.hints = self.arg_groups[ShardHintArgs].build_shard_hints(graph)
        else:
            G_LOGGER.info(
                f"Loading sharding hints from: {args.hint_file.name if hasattr(args.hint_file, 'name') else args.hint_file}"
            )
            self.hints = ShardHints.load(args.hint_file)
        G_LOGGER.info(
            f"Loaded hints: parallelism={self.hints.parallelism}, group_size={self.hints.dist_collectives.group_size}, root={self.hints.dist_collectives.root}, groups={self.hints.dist_collectives.groups}"
        )

        if (
            self.arg_groups[OnnxSaveArgs].all_tensors_to_one_file is False
            and self.hints.parallelism == "TP"
        ):
            G_LOGGER.critical(
                "All tensors must be saved to one file if TP is used. Please omit the `--no-save-all-tensors-to-one-file` option."
            )

        if self.hints.parallelism == "CP":
            graph = self._context_parallel_shard(args, graph)
            model = gs.export_onnx(graph)
            self.fix_group_attribute(model)
            self.arg_groups[OnnxSaveArgs].save_onnx(model)
        elif self.hints.parallelism == "TP":
            base_path = self.arg_groups[OnnxSaveArgs].path
            base_dir = os.path.dirname(base_path)
            base_name = os.path.basename(base_path)
            name_stem, _ = os.path.splitext(base_name)

            tp_manager = self._tensor_parallel_shard(
                graph, onnx_model.graph.initializer
            )
            tp_manager.cache_fp4_raw_data(onnx_model)

            for i in range(self.hints.dist_collectives.nb_rank):
                graph_suffix = f"_tp{self.hints.dist_collectives.nb_rank}_rank{i}.onnx"
                weight_suffix = graph_suffix + ".data"

                G_LOGGER.info(f"Saving rank {i} graph")

                graph = tp_manager.get_rank_graph(i)
                model = gs.export_onnx(graph)

                tp_manager.slice_fp4(model, i)
                self.fix_group_attribute(model)

                # Make sure every graph ends with tp_<world>_rank_<n>
                self.arg_groups[OnnxSaveArgs].path = os.path.join(
                    base_dir, name_stem + graph_suffix
                )
                self.arg_groups[OnnxSaveArgs].external_data_path = (
                    name_stem + weight_suffix
                )
                self.arg_groups[OnnxSaveArgs].save_onnx(model)
        else:
            G_LOGGER.critical("Unsupported parallelism type")
