#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Shared graph-building helpers used by both the ONNX and TRT extractors."""
from polygraphy.tools.inspect.subtool.model.graph_data import TensorInfo


def _meta_to_tensor_infos(metadata):
    """Convert a ``TensorMetadata`` dict to a flat list of ``TensorInfo`` objects."""
    return [
        TensorInfo(
            name=n,
            dtype=m.dtype,
            shape=list(m.shape) if m.shape is not None else None,
            docstring=m.docstring,
        )
        for n, m in metadata.items()
    ]


def _build_edges(nodes, graph_inputs, graph_outputs):
    """
    Build the (producer_id, consumer_id, tensor_name) edge list for the graph.

    Input pseudo-node IDs: ``"input_{i}"``.  Output pseudo-node IDs: ``"output_{i}"``.
    Edges for initializer (weight) tensors are omitted to keep the visual clean.
    """
    tensor_to_producer = {}
    for i, ti in enumerate(graph_inputs):
        tensor_to_producer[ti.name] = f"input_{i}"
    for node in nodes:
        for out_ti in node.outputs:
            tensor_to_producer[out_ti.name] = node.node_id

    edges = []
    for node in nodes:
        for inp_ti in node.inputs:
            if inp_ti.is_initializer:
                continue
            producer = tensor_to_producer.get(inp_ti.name)
            if producer is not None:
                edges.append((producer, node.node_id, inp_ti.name))
    for i, out_ti in enumerate(graph_outputs):
        producer = tensor_to_producer.get(out_ti.name)
        if producer is not None:
            edges.append((producer, f"output_{i}", out_ti.name))
    return edges
