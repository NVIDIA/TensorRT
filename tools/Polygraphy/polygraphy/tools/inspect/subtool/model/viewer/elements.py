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
"""Cytoscape.js element builders and string-formatting helpers."""
from polygraphy.tools.inspect.subtool.model.graph_data import (
    GraphData,
    NodeInfo,
    TensorAttrValue,
    TensorInfo,
)
from polygraphy.tools.inspect.subtool.model.viewer.colors import (
    _GRAPH_INPUT_COLOR,
    _GRAPH_OUTPUT_COLOR,
    _op_color,
)


def _darken(hex_color: str, amount: float = 0.22) -> str:
    h = hex_color.lstrip("#")
    r = int(int(h[0:2], 16) * (1 - amount))
    g = int(int(h[2:4], 16) * (1 - amount))
    b = int(int(h[4:6], 16) * (1 - amount))
    return f"#{r:02x}{g:02x}{b:02x}"


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "\u2026"


_DTYPE_SHORT = {
    "float32": "f32",
    "float16": "f16",
    "bfloat16": "bf16",
    "float64": "f64",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
    "bool": "bool",
    "string": "str",
    "complex64": "c64",
    "complex128": "c128",
}


def _has_info(val: str) -> bool:
    """Return True if *val* is a non-empty, non-placeholder string."""
    return bool(val) and val not in ("Unknown", "")


def _scalar_attrs(node: NodeInfo):
    """Yield ``(key, value)`` pairs from *node.attrs*, skipping ``GraphData`` and ``TensorAttrValue`` entries."""
    for k, v in node.attrs.items():
        if not isinstance(v, (GraphData, TensorAttrValue)):
            yield k, v


def _sg_prefix(prefix: str, node_id: str, attr_name: str) -> str:
    """Return the Cytoscape element-ID namespace for a subgraph attribute's child nodes."""
    return f"{prefix}{node_id}__sg_{attr_name}__"


def _edge_shape_label(ti: TensorInfo) -> str:
    """Shape + dtype label for edges (e.g. '1×3×224×224 f32')."""
    if ti is None:
        return ""
    parts = []
    if ti.shape is not None:
        dims = "\u00d7".join(
            "?" if (d is None or str(d) in ("-1", "?")) else str(d) for d in ti.shape
        )
        if dims:
            parts.append(dims)
    if ti.dtype is not None:
        dtype_str = str(ti.dtype)
        parts.append(_DTYPE_SHORT.get(dtype_str, dtype_str))
    return " ".join(parts)


def _esc(s: str) -> str:
    """HTML-escape a string for safe embedding in node HTML labels."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _node_canvas_label(node: NodeInfo) -> str:
    """Invisible sizing text for the Cytoscape node (HTML overlay provides the visual label)."""
    attrs = [f"{k}: {_truncate(str(v), 14)}" for k, v in _scalar_attrs(node)]
    if _has_info(node.origin):
        attrs.append(f"origin: {_truncate(str(node.origin), 14)}")
    if _has_info(node.tactic):
        attrs.append(f"tactic: {_truncate(str(node.tactic), 18)}")
    if not attrs:
        return node.op_type
    return node.op_type + "\n" + "\n".join(attrs)


def _node_body_html(node: NodeInfo) -> str:
    """Build pre-escaped HTML for the attribute body of a node's HTML overlay."""
    lines = [
        f"{_esc(k)}: {_esc(_truncate(str(v), 14))}" for k, v in _scalar_attrs(node)
    ]
    if _has_info(node.origin):
        lines.append(f"origin: {_esc(_truncate(str(node.origin), 14))}")
    if _has_info(node.tactic):
        lines.append(f"tactic: {_esc(_truncate(str(node.tactic), 18))}")
    if not lines:
        return ""
    return "<br>".join(lines)


def _build_cytoscape_elements(
    graph_data: GraphData,
    prefix: str = "",
    parent_id: str = None,
    positions: dict = None,
) -> list:
    """
    Recursively build Cytoscape.js element dicts from *graph_data*.

    Nodes with ``GraphData`` attributes (Loop, If, Scan) become **compound**
    nodes.  An intermediate container node is created for each subgraph
    attribute (e.g. ``body``, ``then_branch``) so that If's two branches
    are visually distinct.  The function recurses into each subgraph with
    the container as the new parent.

    Returns a flat list of ``{"data": {...}, "classes": "..."}`` dicts.
    """
    elements = []

    def _make_node(node_id, label, bg, border, classes, extra_data=None):
        el = {
            "data": {
                "id": node_id,
                "label": label,
                "bg": bg,
                "borderColor": border,
                **(extra_data or {}),
            },
            "classes": classes,
        }
        if positions and node_id in positions:
            x, y = positions[node_id]
            el["position"] = {"x": x, "y": y}
        if parent_id is not None:
            el["data"]["parent"] = parent_id
        return el

    # Tensor info map for edge labels.
    # TRT: the inspector gives shape on inputs but dtype may only be on graph-level
    # entries (or vice versa).  Merge all occurrences so the edge label gets the
    # best shape AND dtype available.
    def _merge_ti(existing, new_ti):
        if existing is None:
            return new_ti
        return TensorInfo(
            name=existing.name,
            dtype=existing.dtype if existing.dtype is not None else new_ti.dtype,
            shape=existing.shape if existing.shape is not None else new_ti.shape,
        )

    tensor_info_map = {}
    for node in graph_data.nodes:
        for ti in node.inputs + node.outputs:
            tensor_info_map[ti.name] = _merge_ti(tensor_info_map.get(ti.name), ti)
    for ti in graph_data.graph_inputs + graph_data.graph_outputs:
        tensor_info_map[ti.name] = _merge_ti(tensor_info_map.get(ti.name), ti)

    # Graph input pseudo-nodes
    for i, ti in enumerate(graph_data.graph_inputs):
        display_name = _truncate(ti.name, 22)
        elements.append(
            _make_node(
                f"{prefix}input_{i}",
                display_name,
                _GRAPH_INPUT_COLOR,
                _darken(_GRAPH_INPUT_COLOR),
                "graph-io input",
                extra_data={
                    "tensorName": ti.name,
                    "tensorDisplayName": display_name,
                },
            )
        )

    # Op nodes
    for node in graph_data.nodes:
        node_id = f"{prefix}{node.node_id}"
        has_subgraphs = any(isinstance(v, GraphData) for v in node.attrs.values())
        saturated, pastel = _op_color(node.op_type)
        label = _node_canvas_label(node)
        body_html = _node_body_html(node)

        classes = "compound" if has_subgraphs else "op"

        extra_data = {
            "opType": node.op_type,
            "bodyHtml": body_html,
        }

        elements.append(
            _make_node(
                node_id,
                label,
                pastel,
                _darken(saturated),
                classes,
                extra_data=extra_data,
            )
        )

        # Recurse into subgraph attributes — children go directly inside the compound node.
        # (No intermediate sg-container: that would create a second collapsible layer.)
        for attr_name, val in node.attrs.items():
            if isinstance(val, GraphData):
                elements.extend(
                    _build_cytoscape_elements(
                        val,
                        prefix=_sg_prefix(prefix, node.node_id, attr_name),
                        parent_id=node_id,
                        positions=positions,
                    )
                )

    # Graph output pseudo-nodes
    for i, ti in enumerate(graph_data.graph_outputs):
        elements.append(
            _make_node(
                f"{prefix}output_{i}",
                _truncate(ti.name, 22),
                _GRAPH_OUTPUT_COLOR,
                _darken(_GRAPH_OUTPUT_COLOR),
                "graph-io output",
            )
        )

    # Edges — 3-tuple (src, dst, tensor_name); use tensor_name for the shape label.
    seen = set()
    for src, dst, tensor_name in graph_data.edges:
        src_id, dst_id = f"{prefix}{src}", f"{prefix}{dst}"
        if (src_id, dst_id, tensor_name) not in seen:
            seen.add((src_id, dst_id, tensor_name))
            ti = tensor_info_map.get(tensor_name)
            dtype_short = ""
            if ti is not None and ti.dtype is not None:
                dtype_short = _DTYPE_SHORT.get(str(ti.dtype), str(ti.dtype))
            elements.append(
                {
                    "data": {
                        "id": f"e__{src_id}__{dst_id}__{tensor_name}",
                        "source": src_id,
                        "target": dst_id,
                        "label": _edge_shape_label(ti),
                        "dtype": dtype_short,  # preserved when profile selector updates shape
                        "tensorName": tensor_name,
                    },
                }
            )

    return elements
