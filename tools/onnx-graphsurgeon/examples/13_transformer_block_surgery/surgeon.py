#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Sequence

import onnx
import onnx_graphsurgeon as gs


def replace_tensor(graph: gs.Graph, old: gs.Tensor, new: gs.Tensor) -> None:
    for node in graph.nodes:
        node.inputs = [new if tensor is old else tensor for tensor in node.inputs]
    graph.outputs = [new if tensor is old else tensor for tensor in graph.outputs]


def remove_identity_nodes(graph: gs.Graph) -> int:
    removed = 0
    for node in graph.nodes:
        if node.op != "Identity" or len(node.inputs) != 1 or len(node.outputs) != 1:
            continue

        replace_tensor(graph, node.outputs[0], node.inputs[0])
        node.inputs.clear()
        node.outputs.clear()
        removed += 1

    return removed


def compose_permutations(first: Sequence[int], second: Sequence[int]) -> list[int]:
    return [first[index] for index in second]


def cancel_transpose_pairs(graph: gs.Graph) -> int:
    removed_pairs = 0

    for node in graph.nodes:
        if node.op != "Transpose" or len(node.inputs) != 1 or len(node.outputs) != 1:
            continue

        consumers = list(node.outputs[0].outputs)
        if len(consumers) != 1:
            continue

        next_node = consumers[0]
        if (
            next_node.op != "Transpose"
            or len(next_node.inputs) != 1
            or len(next_node.outputs) != 1
        ):
            continue

        first_perm = node.attrs.get("perm")
        second_perm = next_node.attrs.get("perm")
        if not isinstance(first_perm, list) or not isinstance(second_perm, list):
            continue
        if len(first_perm) != len(second_perm):
            continue

        composed = compose_permutations(first_perm, second_perm)
        if composed != list(range(len(composed))):
            continue

        replace_tensor(graph, next_node.outputs[0], node.inputs[0])
        node.inputs.clear()
        node.outputs.clear()
        next_node.inputs.clear()
        next_node.outputs.clear()
        removed_pairs += 1

    return removed_pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply conservative ONNX GraphSurgeon rewrites to a transformer-like block."
    )
    parser.add_argument("--input", required=True, help="Path to the input ONNX model")
    parser.add_argument("--output", required=True, help="Path for the cleaned ONNX model")
    args = parser.parse_args()

    graph = gs.import_onnx(onnx.load(args.input))
    removed_identities = remove_identity_nodes(graph)
    removed_transpose_pairs = cancel_transpose_pairs(graph)

    model = gs.export_onnx(graph.cleanup().toposort())
    onnx.checker.check_model(model)
    onnx.save(model, args.output)

    print(f"Removed Identity nodes: {removed_identities}")
    print(f"Removed Transpose pairs: {removed_transpose_pairs}")
    print(f"Wrote cleaned model: {args.output}")


if __name__ == "__main__":
    main()
