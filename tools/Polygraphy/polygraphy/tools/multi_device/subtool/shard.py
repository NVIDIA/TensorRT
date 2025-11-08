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
from polygraphy import mod
from polygraphy.logger.logger import G_LOGGER
from polygraphy.tools.args.backend.onnx.loader import OnnxInferShapesArgs, OnnxLoadArgs
from polygraphy.tools.base import Tool
from polygraphy.tools.args import ModelArgs
from polygraphy.tools.args import OnnxSaveArgs
from polygraphy.tools.multi_device import get_attention_pattern, ShardHints

onnx = mod.lazy_import("onnx>=1.17")
gs = mod.lazy_import("onnx_graphsurgeon")
onnx_backend = mod.lazy_import("polygraphy.backend.onnx")


class Shard(Tool):
    """
    Convert a SD model to a MD model using a sharding hints file.
    """

    def __init__(self):
        super().__init__("shard")

    def _make_shuffle(self, graph, dist_node, n, rank):
        input = dist_node.inputs[0]
        output = dist_node.outputs[0]
        dist_name = dist_node.name
        dtype = input.dtype
        
        # Perm is static at runtime, we need a rank to actually do the shuffle permutation
        # DistCollective won't change rank, so either get it from input/output or use rank
        # as a fallback
        size = len(input.shape) if input.shape else (len(output.shape) if output.shape else rank)
        if size:
            if n >= size:
                G_LOGGER.critical(f"Specified seq_len_idx {n} is out of actual range {size} for DistCollective node {dist_name}")

            perm = [i for i in range(0, size)]
            perm[0], perm[n] = perm[n], perm[0]
            attrs = {"perm" : perm}

            tensor_pre = gs.Variable(name = "TensorPre_" + dist_name, shape = None, dtype = dtype)
            tensor_post = gs.Variable(name = "TensorPost_" + dist_name, shape = None, dtype = dtype)

            transpose_pre = gs.Node(name = "TransposePre_" + dist_name, op= "Transpose", inputs = [input], outputs = [tensor_pre], attrs = attrs)
            transpose_post = gs.Node(name = "TransposePost_" + dist_name, op = "Transpose", inputs = [tensor_post], outputs = [output], attrs = attrs)

            dist_node.inputs = [tensor_pre]
            dist_node.outputs = [tensor_post]
            graph.nodes.extend([transpose_pre, transpose_post])
        else:
            G_LOGGER.critical("Shape inference needs to be run with --shape-inference if any seq_len_idx is not 0 and rank is not specified")

    def _make_dist_node(self, graph, tensor, attrs, inputs, outputs, tensors, seq_len_idx, fallback):
        name = tensor.name + "_md"

        # Make new tensor and node needed
        tensor_md = gs.Variable(name = name, shape = tensor.shape, dtype = tensor.dtype)

        if inputs is None:
            inputs = [tensor_md]
        if outputs is None:
            outputs = [tensor_md]

        dist_name = "DistCollective_" + str(self.dist_count)

        node_md = gs.Node(op = "DistCollective", name = dist_name, inputs = inputs, outputs = outputs, attrs = attrs)
        
        # Update nodes affected by tensor
        for node in [n for n in graph.nodes if tensor in tensors(n)]:
            for i, t in enumerate(tensors(node)):
                if tensor == t:
                    tensors(node)[i] = tensor_md

        # Change layers that have scattered input as tensor          
        for layer in self.hints.attention_layers: 
                if layer.q == tensor.name:
                        layer.q = tensor_md.name

        if seq_len_idx:
            self._make_shuffle(graph, node_md, seq_len_idx, fallback)
                    
        self.dist_count += 1
        graph.nodes.insert(0, node_md)
        
        return tensor_md

    def _make_attrs(self, collective_operation, reduce_op):
        return {
            "collective_operation": collective_operation,
            "reduce_op": reduce_op,
            "root": self.hints.root,
            "group_size": self.hints.group_size,
        }

    def _make_all_gather(self, graph, tensor, seq_len_idx = None, rank = None):
        """
        Insert an all gather operation to tensor
        T -> T'--AG--T
        """

        G_LOGGER.info(f"Inserting all-gather for tensor: {tensor.name}")
        attrs = self._make_attrs("all_gather", "sum")
        return self._make_dist_node(graph, tensor, attrs, None, [tensor], lambda n : n.outputs, seq_len_idx, rank)

    def _make_reduce_scatter(self, graph, tensor, seq_len_idx = None, rank = None):
        """
        Insert a reduce scatter operation to tensor
        T -> T--RS--T'
        """

        G_LOGGER.info(f"Inserting reduce-scatter for tensor: {tensor.name}")
        attrs = self._make_attrs("reduce_scatter", self.hints.scatter_op)
        return self._make_dist_node(graph, tensor, attrs, [tensor], None, lambda n : n.inputs, seq_len_idx, rank)

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
        ]

    def add_parser_args_impl(self, parser):
        parser.add_argument(
            "-s",
            "--hint",
            help = "Hints file to describe shardable layers.",
            type = argparse.FileType("r"),
            dest = "hint_file",
            required = True
        )

    def run_impl(self, args):
        # Reset state for each run to avoid interference between tests
        self.dist_count = 0
        self.gather_output = False
        gathered_q = False
        
        graph = onnx_backend.gs_from_onnx(self.arg_groups[OnnxLoadArgs].load_onnx())

        G_LOGGER.info(f"Loading sharding hints from: {args.hint_file.name if hasattr(args.hint_file, 'name') else args.hint_file}")
        self.hints = ShardHints.load(args.hint_file)
        G_LOGGER.info(f"Loaded hints: parallelism={self.hints.parallelism}, group_size={self.hints.group_size}, root={self.hints.root}, groups={self.hints.groups}")

        
        # Shard dependent inputs/outputs
        tensors = graph.tensors()
        for input in self.hints.inputs:
            tensor_md = self._make_reduce_scatter(graph, tensors[input.name], input.seq_len_idx, input.rank)

            # Update attention layers if new input will be scattered tensor
            for a_l in [a_l for a_l in self.hints.attention_layers if a_l.q == tensor_md.name]:
                a_l.q = tensor_md.name

        # Get all attention layers that match supported pattern(s)
        pattern = get_attention_pattern()
        matches = pattern.match_all(graph)
        G_LOGGER.info(f"Found {len(matches)} attention pattern matches in the graph.")

        # Perform sharding
        for layer in self.hints.attention_layers:
            G_LOGGER.info(f"Processing attention layer: q={layer.q}, gather_q={layer.gather_q}, gather_kv={layer.gather_kv}")
            # Find configuration for matching attenion layer (inputs and outputs match)
            match = next((match for match in matches if match.inputs[0].name == layer.q), None)
            if match is not None:
                sharded = set()
                gather_q = layer.gather_q
                gather_kv = layer.gather_kv

                # Get tensors directly
                q = match["MatMul1"].onnx_node.inputs[0]
                k = match["MatMul1"].onnx_node.inputs[1]
                v = match["MatMul2"].onnx_node.inputs[1]
    
                # Insert collective ops as specified
                for i, (tensor, seq_len_idx, rank) in enumerate([(q, None, None), (k, self.hints.k_seq_len_idx, self.hints.kv_rank), (v, self.hints.v_seq_len_idx, self.hints.kv_rank)]): 
                    if [gather_q, gather_kv, gather_kv][i] and tensor.name not in sharded:

                        # If any q is gathered (on purpose or k == q or v == q), prevent final output from being all gathered
                        if tensor.name == layer.q and not gathered_q:
                            G_LOGGER.info(f"Q {layer.q} was gathered")
                            gathered_q = True

                        self._make_all_gather(graph, tensor, seq_len_idx, rank)
                        sharded.add(tensor.name)
            else:
                G_LOGGER.warning(f"No matching attention pattern found for layer with q={layer.q}")

        if not gathered_q:
            for output in self.hints.outputs:
                self._make_all_gather(graph, tensors[output.name], output.seq_len_idx, output.rank)

        # Cleanup and save
        graph.cleanup()
        graph.toposort()
        
        # Manually add in groups attribute since graph surgeon doesn't support
        # type inference for an empty list, which is necessary for a group configuration of '[]'
        model = gs.export_onnx(graph)
        for node in model.graph.node:
            if node.op_type == "DistCollective":
                node.attribute.append(onnx.helper.make_attribute("groups", self.hints.groups, attr_type = onnx.AttributeProto.INTS))
        
        self.arg_groups[OnnxSaveArgs].save_onnx(model)
