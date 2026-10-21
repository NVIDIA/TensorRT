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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TensorInfo:
    """Represents a single tensor (input, output, or initializer) in the model graph."""

    name: str
    dtype: Optional[Any] = None  # DataType or None
    shape: Optional[list] = None  # shape as a list; None or str entries = dynamic dims
    is_initializer: bool = False  # True if this tensor is a model weight/initializer
    docstring: Optional[str] = None  # e.g. TRT format string
    values: Any = None  # Weight values (numpy array or error string); None = not loaded


@dataclass
class TensorAttrValue:
    """Represents an ONNX TENSOR-type attribute value."""

    dtype: Any
    shape: list
    values: Any = None  # numpy array or error string; None = not loaded


@dataclass
class NodeInfo:
    """Represents a single layer/node in the model graph."""

    node_id: str  # Unique identifier (e.g. "node_0")
    name: str  # Human-readable name
    op_type: str  # Operation type (e.g. "Conv", "CONVOLUTION")
    inputs: List[TensorInfo] = field(default_factory=list)
    outputs: List[TensorInfo] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    # TRT engine extras
    origin: Optional[str] = None
    tactic: Optional[str] = None


@dataclass
class ProfileTensorInfo:
    """Describes one tensor's shape range in a TRT engine optimization profile."""

    name: str
    tensor_index: int
    is_input: bool
    min_shape: Optional[tuple] = None  # inputs only
    opt_shape: Optional[tuple] = None  # inputs only
    max_shape: Optional[tuple] = None  # inputs only
    shape: Optional[tuple] = None  # outputs only


@dataclass
class ProfileInfo:
    """TRT engine optimization profile (shape ranges for all I/O tensors)."""

    index: int
    tensor_infos: List[ProfileTensorInfo] = field(default_factory=list)


@dataclass
class GraphData:
    """
    Backend-agnostic intermediate representation of a model graph.

    Used as the single source of truth for both text formatting (``str_from_graph_data``)
    and interactive visualization (``ModelViewer``).  Extractors in ``extractors.py``
    populate this from ONNX / TRT objects; consumers in ``text.py`` and ``viewer.py``
    read from it without touching backend APIs.

    model_type values: ``"onnx"``, ``"trt_network"``, ``"trt_engine"``
    """

    title: str  # Header line, e.g. "Name: resnet50 | ONNX Opset: 13"
    model_type: str
    is_subgraph: bool = False  # True for ONNX GRAPH-type subgraph attributes
    graph_inputs: List[TensorInfo] = field(default_factory=list)
    graph_outputs: List[TensorInfo] = field(default_factory=list)
    nodes: List[NodeInfo] = field(default_factory=list)
    # Edges between nodes: (from_node_id, to_node_id, tensor_name).
    # Input pseudo-node IDs have the form "input_{i}"; output pseudo-node IDs "output_{i}".
    edges: List[Tuple[str, str, str]] = field(default_factory=list)
    initializers: List[TensorInfo] = field(default_factory=list)
    doc_string: Optional[str] = None
    # TRT engine-specific fields
    device_memory_bytes: Optional[int] = None
    num_io_tensors: int = 0
    profiles: List[ProfileInfo] = field(default_factory=list)
    # Per-profile layer lists (one inner list per optimization profile).
    per_profile_nodes: List[List[NodeInfo]] = field(default_factory=list)
    num_layers: int = 0
