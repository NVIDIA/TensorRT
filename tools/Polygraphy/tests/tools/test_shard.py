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

import onnx
from polygraphy.tools.multi_device import ShardHints
from polygraphy import util
from tests.models.meta import ONNX_MODELS

class TestShard:
    
    def check_expected_dist_count(self, path, scatter_count, gather_count):
        model = onnx.load(path)

        dist_nodes = [node for node in model.graph.node if node.op_type == "DistCollective"]

        def has_collective_operation(node, op):
            for attr in node.attribute:
                if attr.name == "collective_operation" and attr.s.decode() == op:
                    return True
            return False

        gather_nodes = [node for node in dist_nodes if has_collective_operation(node, "all_gather")]
        scatter_nodes = [node for node in dist_nodes if has_collective_operation(node, "reduce_scatter")]

        assert len(gather_nodes) == gather_count
        assert len(scatter_nodes) == scatter_count

    def test_multi_attention_head_shard(
        self, poly_multi_device_shard, poly_template_shard
    ):
        # Test for a network with multiple attention heads, shard only affects specified ones
        with util.NamedTemporaryFile(suffix=".onnx") as outmodel, util.NamedTemporaryFile(mode='w+', suffix=".json") as hints:
            poly_template_shard(
                [
                    ONNX_MODELS["multi_attention"].path,
                    "-o",
                    hints.name
                ]
            )

            # Remove the second attention
            json = ShardHints.load(hints.name)
            json.inputs = json.inputs[:len(json.inputs)//2]
            json.outputs = json.outputs[:len(json.outputs)//2]
            json.attention_layers = json.attention_layers[:len(json.attention_layers)//2]
            json.save(hints.name)

            poly_multi_device_shard(
                [
                    ONNX_MODELS["multi_attention"].path,
                    "-o",
                    outmodel.name,
                    "-s",
                    hints.name
                ]
            )

            # Should only gather specified kv
            self.check_expected_dist_count(outmodel.name, 3, 3)
    
    def test_shard_same_qkv(
        self, poly_multi_device_shard, poly_template_shard
    ):
        # Test shard doesn't insert multiple all_gathers if any inputs are the same
        with util.NamedTemporaryFile(suffix=".onnx") as outmodel, util.NamedTemporaryFile(mode = "w+", suffix=".json") as hints: 
            poly_template_shard(
                [
                    ONNX_MODELS["attention_same_qkv"].path,
                    "-o",
                    hints.name
                ]
            )
            poly_multi_device_shard(
                [
                    ONNX_MODELS["attention_same_qkv"].path,
                    "-o",
                    outmodel.name,
                    "-s",
                    hints.name
                ]
            )

            # One scatter for single input, one gather for qkv (skip gather at end because q was gathered)
            self.check_expected_dist_count(outmodel.name, 1, 1)
    
    def test_shard(
        self, poly_multi_device_shard, poly_template_shard
    ):
        # Test normal sharding of kv for attention
        with util.NamedTemporaryFile(suffix=".onnx") as outmodel, util.NamedTemporaryFile(mode = "w+", suffix=".json") as hints:
            poly_template_shard(
                [
                    ONNX_MODELS["attention"].path,
                    "-o",
                    hints.name
                ]
            )
            poly_multi_device_shard(
                [
                    ONNX_MODELS["attention"].path,
                    "-o",
                    outmodel.name,
                    "-s",
                    hints.name
                ]
            )

            # 3 scatters for each input, 2 gathers for kv, one gather at end
            self.check_expected_dist_count(outmodel.name, 3, 3)
