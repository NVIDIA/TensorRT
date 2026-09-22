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
from polygraphy.logger.logger import G_LOGGER
from polygraphy.tools.multi_device.multi_device import DistCollective, FusedAttention
from polygraphy.tools.template.subtool.base import BaseTemplateTool
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.backend.onnx.loader import OnnxLoadArgs
from polygraphy.tools.args import ModelArgs
from polygraphy.tools.multi_device import (
    get_attention_pattern,
    get_attention_pattern_alt,
    ShardHints,
    AttentionLayerHint,
    ShardTensor,
)


onnx_backend = mod.lazy_import("polygraphy.backend.onnx")

SHARD_HINT_CP_TYPES = ["native", "fused"]
REDUCE_OPS = ["sum", "prod", "min", "max", "avg"]


class ShardHintArgs(BaseArgs):
    """
    Shard Hints: configuring how models are sharded across devices.
    """

    def add_parser_args_impl(self):
        self.group.add_argument(
            "--parallelism",
            help="Type of parallelism to use",
            type=str,
            choices=["CP", "TP"],
            default="CP",
        )
        self.group.add_argument(
            "--root",
            help="Rank of root process",
            type=int,
            default=-1,
        )
        self.group.add_argument(
            "--gpus",
            help="Number of participating gpus (0 is all gpus)",
            type=int,
            default=0,
        )
        self.group.add_argument(
            "--groups",
            help="Space-separated list of NCCL group indices (omit for all groups)",
            nargs="*",
            type=int,
            default=[],
        )
        self.group.add_argument(
            "--cp-type",
            help="Sharding strategy for attention layers",
            type=str,
            choices=SHARD_HINT_CP_TYPES,
            default=SHARD_HINT_CP_TYPES[0],
        )
        self.group.add_argument(
            "--no-suggest-io",
            help="Disable suggestions of which input/output tensors need to be sharded based on attention layer dependencies",
            action="store_true",
            default=False,
        )
        self.group.add_argument(
            "--i-idx",
            help="Default index of sequence length on input tensor(s)",
            type=int,
            default=0,
        )
        self.group.add_argument(
            "--o-idx",
            help="Default index of sequence length on output tensor(s)",
            type=int,
            default=0,
        )
        self.group.add_argument(
            "--k-idx",
            help="Default index of sequence length on K tensor(s)",
            type=int,
            default=0,
        )
        self.group.add_argument(
            "--v-idx",
            help="Default index of sequence length on V tensor(s)",
            type=int,
            default=0,
        )
        self.group.add_argument(
            "--i-rank",
            help="Fallback rank of input shapes if sequence length index is > 0 and shape inference is not run",
            type=int,
            default=None,
        )
        self.group.add_argument(
            "--o-rank",
            help="Fallback rank of output shapes if sequence length index is > 0 and shape inference is not run",
            type=int,
            default=None,
        )
        self.group.add_argument(
            "--kv-rank",
            help="Fallback rank of KV shapes if sequence length index is > 0 and shape inference is not run",
            type=int,
            default=None,
        )
        self.group.add_argument(
            "--reduce-op",
            help="reduce_op for reduce_scatter operations",
            type=str,
            choices=REDUCE_OPS,
            default="max",
        )
        self.group.add_argument(
            "--attention-pattern",
            help="Which pattern to use for attention detection",
            type=int,
            choices=[0, 1],
            default=0,
        )
        self.group.add_argument(
            "--q-shuffle",
            help="Space-separated list of transpose permutation to apply to Q before attention replacement",
            nargs="*",
            type=int,
            default=None,
        )
        self.group.add_argument(
            "--k-shuffle",
            help="Space-separated list of transpose permutation to apply to K before attention replacement",
            nargs="*",
            type=int,
            default=None,
        )
        self.group.add_argument(
            "--v-shuffle",
            help="Space-separated list of transpose permutation to apply to V before attention replacement",
            nargs="*",
            type=int,
            default=None,
        )
        self.group.add_argument(
            "--is-causal",
            help="Whether or not the attention op is causal",
            action="store_true",
            default=False,
        )
        self.group.add_argument(
            "--nb-rank",
            help="Number of ranks for the attention op",
            type=int,
            default=1,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            parallelism (str): Type of parallelism ("CP" or "TP").
            root (int): Rank of root process.
            gpus (int): Number of participating GPUs (0 = all).
            groups (List[int]): NCCL group indices.
            cp_type (str): Sharding strategy for attention layers ("native" or "fused").
            no_suggest_io (bool): Whether to disable IO shard suggestions.
            i_idx (int): Default sequence length index for inputs.
            o_idx (int): Default sequence length index for outputs.
            k_idx (int): Default sequence length index for K tensors.
            v_idx (int): Default sequence length index for V tensors.
            i_rank (int): Fallback rank for input shapes.
            o_rank (int): Fallback rank for output shapes.
            kv_rank (int): Fallback rank for KV shapes.
            reduce_op (str): Reduce op for reduce_scatter operations.
            attention_pattern (int): Which attention detection pattern to use.
            q_shuffle (List[int]): Transpose permutation for Q.
            k_shuffle (List[int]): Transpose permutation for K.
            v_shuffle (List[int]): Transpose permutation for V.
            is_causal (bool): Whether the attention op is causal.
            nb_rank (int): Number of ranks for the attention op.
        """
        self.parallelism = args_util.get(args, "parallelism")
        self.root = args_util.get(args, "root")
        self.gpus = args_util.get(args, "gpus")
        self.groups = args_util.get(args, "groups")
        self.cp_type = args_util.get(args, "cp_type")
        self.no_suggest_io = args_util.get(args, "no_suggest_io")
        self.i_idx = args_util.get(args, "i_idx")
        self.o_idx = args_util.get(args, "o_idx")
        self.k_idx = args_util.get(args, "k_idx")
        self.v_idx = args_util.get(args, "v_idx")
        self.i_rank = args_util.get(args, "i_rank")
        self.o_rank = args_util.get(args, "o_rank")
        self.kv_rank = args_util.get(args, "kv_rank")
        self.reduce_op = args_util.get(args, "reduce_op")
        self.attention_pattern = args_util.get(args, "attention_pattern")
        self.q_shuffle = args_util.get(args, "q_shuffle")
        self.k_shuffle = args_util.get(args, "k_shuffle")
        self.v_shuffle = args_util.get(args, "v_shuffle")
        self.is_causal = args_util.get(args, "is_causal")
        self.nb_rank = args_util.get(args, "nb_rank")

    @staticmethod
    def _guess_seq_len_idx(tensor, default):
        if (shape := tensor.shape) is not None:
            for i, dim in enumerate(shape):
                if dim == "sequence_length" or dim == "seq_len":
                    G_LOGGER.info(
                        f"Found sequence_length index at {i} for tensor {tensor.name}"
                    )
                    return i
        return default

    @staticmethod
    def _make_fused_attention(hint_args):
        return FusedAttention(
            hint_args.is_causal,
            hint_args.q_shuffle,
            hint_args.k_shuffle,
            hint_args.v_shuffle,
            hint_args.nb_rank,
        )

    def build_shard_hints(self, graph):
        """
        Build a ShardHints object from a graph using the parsed attributes
        of this argument group.

        Args:
            graph: An onnx-graphsurgeon graph.

        Returns:
            ShardHints: The constructed shard hints.
        """
        replace_table = {SHARD_HINT_CP_TYPES[1]: ShardHintArgs._make_fused_attention}

        attention_patterns = {
            0: get_attention_pattern,
            1: get_attention_pattern_alt,
        }

        traverser = GraphTraverser(graph)
        gather_kv = self.cp_type == SHARD_HINT_CP_TYPES[0]
        attention_layers = []
        inputs = []
        outputs = []
        kv_rank = None
        get_rank = lambda t, default: len(t.shape) if t.shape else default
        replace = (
            replace_table[self.cp_type](self) if self.cp_type in replace_table else None
        )

        for match in attention_patterns[self.attention_pattern]().match_all(graph):
            q = match.inputs[0].name
            G_LOGGER.info(f"Found attention layer with Q tensor {q}")

            k = match.inputs[1]
            v = match.inputs[2]

            if not kv_rank:
                tensor = k if k.shape else v
                G_LOGGER.info(f"Trying to find KV rank...")
                kv_rank = get_rank(tensor, self.kv_rank)

                if kv_rank:
                    G_LOGGER.info(f"KV has rank of {kv_rank}")

            if not self.no_suggest_io:
                inputs.extend(traverser.get_dep_inputs(match["MatMul2"].onnx_node))
                outputs.extend(traverser.get_dep_outputs(match["MatMul2"].onnx_node))

            attention_layers.append(AttentionLayerHint(q, gather_kv, False, replace))

        convert_io = lambda tensors, default, rank=None: list(
            map(
                lambda t: ShardTensor(
                    t.name,
                    ShardHintArgs._guess_seq_len_idx(t, default),
                    get_rank(t, rank),
                ),
                tensors,
            )
        )
        inputs = convert_io(inputs, self.i_idx, self.i_rank)
        outputs = convert_io(outputs, self.o_idx, self.o_rank)
        dist_collectives = DistCollective(
            self.gpus, self.root, self.nb_rank, self.groups, self.reduce_op
        )
        return ShardHints(
            self.parallelism,
            attention_layers,
            dist_collectives,
            inputs,
            outputs,
            self.k_idx,
            self.v_idx,
            kv_rank if kv_rank else self.kv_rank,
        )


class GraphTraverser:
    def __init__(self, graph):
        self.inputs = {input.name: input for input in graph.inputs}
        self.outputs = {output.name: output for output in graph.outputs}
        self.nodes = {node.name: node for node in graph.nodes}
        self.visited_inputs = set()
        self.visited_outputs = set()
        self.reached_terminal = set()

        if len(self.nodes) != len(graph.nodes):
            G_LOGGER.critical(f"All nodes in graph need to have unique names")

        # Tensors are the output of only one node
        self.node_outputs = {
            output.name: {node.name} for node in graph.nodes for output in node.outputs
        }

        # Have to handle this slightly differently
        self.node_inputs = {
            input.name: set() for node in graph.nodes for input in node.inputs
        }
        for node in graph.nodes:
            for input in node.inputs:
                self.node_inputs[input.name].add(node.name)

    def _traverse(self, node, edges, terminal, relative, dependent, visited):
        queue = [node]
        while queue:
            cur = queue.pop()
            if cur.name in visited:
                continue

            visited.add(cur.name)

            for tensor in edges(cur):
                if tensor in terminal and tensor not in self.reached_terminal:
                    G_LOGGER.info(f"Found dependent tensor {tensor}")
                    dependent.add(tensor)
                    self.reached_terminal.add(tensor)
                elif tensor in relative:
                    for r in relative[tensor]:
                        if r not in visited:
                            queue.append(self.nodes[r])

    def get_dep_inputs(self, node):
        inputs = set()
        self._traverse(
            node,
            lambda n: [input.name for input in n.inputs],
            self.inputs,
            self.node_outputs,
            inputs,
            self.visited_inputs,
        )
        return [self.inputs[input] for input in inputs]

    def get_dep_outputs(self, node):
        outputs = set()
        self._traverse(
            node,
            lambda n: [output.name for output in n.outputs],
            self.outputs,
            self.node_inputs,
            outputs,
            self.visited_outputs,
        )
        return [self.outputs[output] for output in outputs]


class ShardHint(BaseTemplateTool):
    """
    Generate a sharding hints file
    """

    def __init__(self):
        super().__init__("shard-hints")

    def get_subscriptions_impl(self):
        return [
            ModelArgs(
                model_opt_required=True,
                input_shapes_opt_name=False,
                required_model_type="onnx",
            ),
            OnnxLoadArgs(outputs_opt_prefix=False, allow_shape_inference=False),
            ShardHintArgs(),
        ]

    def add_parser_args_impl(self, parser):
        super().add_parser_args_impl(parser)

    def run_impl(self, args):
        graph = onnx_backend.gs_from_onnx(self.arg_groups[OnnxLoadArgs].load_onnx())

        if not args.output.name.endswith(".json"):
            G_LOGGER.critical("Output file must be a json")

        hints = self.arg_groups[ShardHintArgs].build_shard_hints(graph)
        hints.save(args.output)
