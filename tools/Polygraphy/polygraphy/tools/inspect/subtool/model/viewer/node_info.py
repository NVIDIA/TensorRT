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
"""Node-detail JSON builders for the sidebar click-handler and search."""
from polygraphy.tools.inspect.subtool.model.graph_data import (
    GraphData,
    TensorAttrValue,
    TensorInfo,
)
from polygraphy.tools.inspect.subtool.model.viewer.elements import _sg_prefix


# ── Weight preview helpers ────────────────────────────────────────────────────

_WEIGHT_PREVIEW_MAX_EACH = 8  # values shown from head and tail for large tensors
_WEIGHT_PREVIEW_INLINE_MAX = 16  # show all values when total elements ≤ this


def _weight_preview(values):
    """
    Return a JSON-serializable weight-preview dict, or None if values is absent or an error.

    For tensors with ≤ ``_WEIGHT_PREVIEW_INLINE_MAX`` elements, all values are returned.
    For larger tensors, the first and last ``_WEIGHT_PREVIEW_MAX_EACH`` values are returned.
    """
    if values is None or isinstance(values, str):
        return None
    try:
        flat = values.flatten()
        total = int(flat.size)
        if total <= _WEIGHT_PREVIEW_INLINE_MAX:
            return {"values": flat.tolist(), "total": total, "truncated": False}
        head = flat[:_WEIGHT_PREVIEW_MAX_EACH].tolist()
        tail = flat[-_WEIGHT_PREVIEW_MAX_EACH:].tolist()
        return {
            "values": head + tail,
            "split": _WEIGHT_PREVIEW_MAX_EACH,
            "total": total,
            "truncated": True,
        }
    except Exception:
        return None


# ── JSON payload for sidebar click-handler ────────────────────────────────────


def _collect_subgraph_node_ids(graph_data: GraphData, prefix: str) -> list:
    """
    Return the **direct-level** node IDs for *graph_data* under *prefix*.
    """
    ids = [f"{prefix}input_{i}" for i in range(len(graph_data.graph_inputs))]
    ids += [f"{prefix}{node.node_id}" for node in graph_data.nodes]
    ids += [f"{prefix}output_{i}" for i in range(len(graph_data.graph_outputs))]
    return ids


def _build_node_json(
    graph_data: GraphData, prefix: str = "", result: dict = None
) -> dict:
    """
    Populate *result* with all node entries for *graph_data* and its nested subgraphs.
    Used by the sidebar click-handler to display node details.
    """
    if result is None:
        result = {}

    def _ti(ti: TensorInfo) -> dict:
        return {
            "name": f"Initializer | {ti.name}" if ti.is_initializer else ti.name,
            "dtype": str(ti.dtype) if ti.dtype is not None else None,
            "shape": [str(d) for d in ti.shape] if ti.shape is not None else None,
            "docstring": ti.docstring,
            "weight_preview": _weight_preview(ti.values),
        }

    def _attr_str(val) -> str:
        if isinstance(val, TensorAttrValue):
            return f"Tensor [dtype={val.dtype}, shape={val.shape}]"
        s = str(val)
        return s if len(s) <= 140 else s[:137] + "\u2026"

    for i, ti in enumerate(graph_data.graph_inputs):
        result[f"{prefix}input_{i}"] = {
            "kind": "input",
            "node_id": f"{prefix}input_{i}",
            "name": ti.name,
            "op_type": "Graph Input",
            "inputs": [],
            "outputs": [_ti(ti)],
            "attrs": [],
            "subgraphs": {},
            "origin": None,
            "tactic": None,
        }

    for node in graph_data.nodes:
        prefixed_id = f"{prefix}{node.node_id}"
        attrs_list = []
        subgraphs = {}

        for attr_name, val in node.attrs.items():
            if isinstance(val, GraphData):
                sp = _sg_prefix(prefix, node.node_id, attr_name)
                subgraphs[attr_name] = {
                    "all_node_ids": _collect_subgraph_node_ids(val, sp),
                }
                _build_node_json(val, prefix=sp, result=result)
            else:
                attrs_list.append(
                    {
                        "key": attr_name,
                        "val": _attr_str(val),
                        "weight_preview": (
                            _weight_preview(val.values)
                            if isinstance(val, TensorAttrValue)
                            else None
                        ),
                    }
                )

        result[prefixed_id] = {
            "kind": "node",
            "node_id": prefixed_id,
            "name": node.name,
            "op_type": node.op_type,
            "inputs": [_ti(ti) for ti in node.inputs],
            "outputs": [_ti(ti) for ti in node.outputs],
            "attrs": attrs_list,
            "subgraphs": subgraphs,
            "origin": node.origin,
            "tactic": node.tactic,
        }

    for i, ti in enumerate(graph_data.graph_outputs):
        result[f"{prefix}output_{i}"] = {
            "kind": "output",
            "node_id": f"{prefix}output_{i}",
            "name": ti.name,
            "op_type": "Graph Output",
            "inputs": [_ti(ti)],
            "outputs": [],
            "attrs": [],
            "subgraphs": {},
            "origin": None,
            "tactic": None,
        }

    return result


# ── Lite node JSON (served mode: minimal payload for search + extract) ─────────


def _build_node_json_lite(
    graph_data: GraphData, prefix: str = "", result: dict = None
) -> dict:
    """
    Build a slim node-info dict keyed by prefixed node ID.

    Includes only ``op_type``, ``name``, ``kind``, and basic tensor info
    (name, dtype, shape — no attrs, no weight previews, no subgraph details).
    Used as the initial ``ND_LITE`` payload in served mode; full details are
    fetched lazily from ``/node/<id>`` on first click.
    """
    if result is None:
        result = {}

    def _ti_lite(ti: TensorInfo) -> dict:
        return {
            "name": f"Initializer | {ti.name}" if ti.is_initializer else ti.name,
            "dtype": str(ti.dtype) if ti.dtype is not None else None,
            "shape": [str(d) for d in ti.shape] if ti.shape is not None else None,
        }

    for i, ti in enumerate(graph_data.graph_inputs):
        result[f"{prefix}input_{i}"] = {
            "kind": "input",
            "node_id": f"{prefix}input_{i}",
            "name": ti.name,
            "op_type": "Graph Input",
            "inputs": [],
            "outputs": [_ti_lite(ti)],
        }

    for node in graph_data.nodes:
        prefixed_id = f"{prefix}{node.node_id}"
        result[prefixed_id] = {
            "kind": "node",
            "node_id": prefixed_id,
            "name": node.name,
            "op_type": node.op_type,
            "inputs": [_ti_lite(ti) for ti in node.inputs],
            "outputs": [_ti_lite(ti) for ti in node.outputs],
        }
        # Recurse into subgraphs so their nodes are searchable.
        for attr_name, val in node.attrs.items():
            if isinstance(val, GraphData):
                _build_node_json_lite(
                    val,
                    prefix=_sg_prefix(prefix, node.node_id, attr_name),
                    result=result,
                )

    for i, ti in enumerate(graph_data.graph_outputs):
        result[f"{prefix}output_{i}"] = {
            "kind": "output",
            "node_id": f"{prefix}output_{i}",
            "name": ti.name,
            "op_type": "Graph Output",
            "inputs": [_ti_lite(ti)],
            "outputs": [],
        }

    return result
