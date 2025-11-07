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
from polygraphy.tools.template.subtool.base import BaseTemplateTool
from polygraphy.tools.args.backend.onnx.loader import OnnxLoadArgs
from polygraphy.tools.args import ModelArgs
from polygraphy.tools.multi_device import get_attention_pattern, ShardHints, AttentionLayerHint, ShardTensor


onnx_backend = mod.lazy_import("polygraphy.backend.onnx")


class GraphTraverser():
    def __init__(self, graph):
        self.inputs = {input.name : input for input in graph.inputs}
        self.outputs = {output.name : output for output in graph.outputs}
        self.nodes = {node.name :  node for node in graph.nodes}
        self.visited_inputs = set()
        self.visited_outputs = set()
        self.reached_terminal = set()

        if len(self.nodes) != len(graph.nodes):
            G_LOGGER.critical(f"All nodes in graph need to have unique names")

        # Tensors are the output of only one node
        self.node_outputs = {output.name: {node.name} for node in graph.nodes for output in node.outputs}

        # Have to handle this slightly differently
        self.node_inputs = {input.name : set() for node in graph.nodes for input in node.inputs}
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
            self.visited_inputs
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
            self.visited_outputs
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
        ]

    def add_parser_args_impl(self, parser):
        reduce_ops = ["sum", "prod", "min", "max", "avg"]

        super().add_parser_args_impl(parser)

        parser.add_argument(
            "--parallelism",
            help = "Type of parallelism to use",
            type = str,
            choices = ["CP"],
            default = "CP"
        )

        parser.add_argument(
            "--root",
            help = "Rank of root process",
            type=int,
            default=0
        )

        parser.add_argument(
            "--gpus",
            help = "Number of participating gpus (0 is all gpus)",
            type=int,
            default=0
        )

        parser.add_argument(
            "--groups",
            help="Space-separated list of NCCL group indices (omit for all groups)",
            nargs="*",
            type=int,
            default=[],
        )

        parser.add_argument(
            "--cp-type",
            help = "Sharding strategy for attention layers",
            type=str,
            choices=["native", "ring_attention", "fused"],
            default = "native"
        )

        parser.add_argument(
            "--no-suggest-io",
            help = "Disable suggestions of which input/output tensors need to be sharded based on attention layer dependencies",
            action='store_true',
            default=False
        )

        parser.add_argument(
            "--i-idx",
            help = "Default index of sequence length on input tensor(s)",
            type=int,
            default=0
        )

        parser.add_argument(
            "--o-idx",
            help = "Default index of sequence length on output tensor(s)",
            type=int,
            default=0
        )

        parser.add_argument(
            "--k-idx",
            help = "Default index of sequence length on K tensor(s)",
            type=int,
            default=0
        )

        parser.add_argument(
            "--v-idx",
            help = "Default index of sequence length on V tensor(s)",
            type=int,
            default=0
        )

        parser.add_argument(
            "--i-rank",
            help = "Fallback rank of input shapes if sequence length index is > 0 and shape inference is not run",
            type=int,
        )

        parser.add_argument(
            "--o-rank",
            help = "Fallback rank of output shapes if sequence length index is > 0 and shape inference is not run",
            type=int,
        )

        parser.add_argument(
            "--kv-rank",
            help = "Fallback rank of KV shapes if sequence length index is > 0 and shape inference is not run",
            type=int,
        )

        parser.add_argument(
            "--scatter-op",
            help = "reduce_op for reduce_scatter operations",
            type=str,
            choices = reduce_ops,
            default = "max"
        )

    @staticmethod
    def guess_seq_len_idx(tensor, default): 
        if (shape := tensor.shape) is not None:
            for i, dim in enumerate(shape):
                if dim == "sequence_length" or dim == "seq_len":
                    G_LOGGER.info(f"Found sequence_length index at {i} for tensor {tensor.name}")
                    return i

        return default

    def run_impl(self, args):
        graph = onnx_backend.gs_from_onnx(self.arg_groups[OnnxLoadArgs].load_onnx())

        if not args.output.name.endswith(".json"):
            G_LOGGER.critical("Output file must be a json")

        traverser = GraphTraverser(graph)
        gather_kv = args.cp_type == "native"
        attention_layers = []
        inputs = []
        outputs = []
        kv_rank = None
        get_rank = lambda t, default: len(t.shape) if t.shape else default

        for match in get_attention_pattern().match_all(graph):
            q = match.inputs[0].name
            G_LOGGER.info(f"Found attention layer with Q tensor {q}")  

            k = match.inputs[1]
            v = match.inputs[2]

            # Find rank of kv (all assumed to be same)
            if not kv_rank:
                # K or V will be same rank, use whichever has a shape (if any)
                tensor = k if k.shape else v
                G_LOGGER.info(f"Trying to find KV rank...")
                kv_rank = get_rank(tensor, args.kv_rank)

                if kv_rank:
                    G_LOGGER.info(f"KV has rank of {kv_rank}")

            # Find dependent inputs/outputs (suggestion)
            if not args.no_suggest_io:
                inputs.extend(traverser.get_dep_inputs(match["MatMul2"].onnx_node))
                outputs.extend(traverser.get_dep_outputs(match["MatMul2"].onnx_node))

            # TODO TRT-26378 https://jirasw.nvidia.com/browse/TRT-26378 add support for plugin, later myelin
            attention_layers.append(AttentionLayerHint(q, gather_kv, False))
        
        
        convert_io = lambda tensors, default, rank=None: list(map(lambda t : ShardTensor(t.name, ShardHint.guess_seq_len_idx(t, default), get_rank(t, rank)), tensors))
        inputs = convert_io(inputs, args.i_idx, args.i_rank)
        outputs = convert_io(outputs, args.o_idx, args.o_rank)
        hints = ShardHints(args.parallelism, args.gpus, args.root, args.groups, attention_layers, inputs, outputs, args.k_idx, args.v_idx, kv_rank if kv_rank else args.kv_rank, args.scatter_op)

        hints.save(args.output)
