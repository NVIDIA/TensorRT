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
"""HTML page builder and top-level JSON payload assemblers."""
import json

from polygraphy.tools.inspect.subtool.model.graph_data import GraphData, TensorInfo
from polygraphy.tools.inspect.subtool.model.viewer._templates import (
    _CDN_SCRIPTS,
    _CSS,
    _CYTOSCAPE_JS,
    _CYTOSCAPE_STYLE,
)
from polygraphy.tools.inspect.subtool.model.viewer.colors import (
    _CATEGORY_NAMES,
    _GRAPH_INPUT_COLOR,
    _GRAPH_OUTPUT_COLOR,
    _OP_CATEGORY_COLORS,
)
from polygraphy.tools.inspect.subtool.model.viewer.elements import _darken, _sg_prefix
from polygraphy.tools.inspect.subtool.model.viewer.node_info import (
    _build_node_json,
    _build_node_json_lite,
)


# ── Legend ────────────────────────────────────────────────────────────────────


def _legend_html(graph_data: GraphData) -> str:
    present_ops = {n.op_type for n in graph_data.nodes}
    items = []
    for (ops, saturated, _pastel), name in zip(_OP_CATEGORY_COLORS, _CATEGORY_NAMES):
        if ops & present_ops:
            items.append(
                f'<span class="pg-li">'
                f'<span class="pg-sw" style="background:{saturated};border:1px solid {_darken(saturated)}"></span>'
                f'<span class="pg-ll">{name}</span></span>'
            )
    for color, label in [
        (_GRAPH_INPUT_COLOR, "Input"),
        (_GRAPH_OUTPUT_COLOR, "Output"),
    ]:
        items.insert(
            0,
            f'<span class="pg-li">'
            f'<span class="pg-sw" style="background:{color};border:1px solid {_darken(color)}"></span>'
            f'<span class="pg-ll">{label}</span></span>',
        )
    return '<div id="pg-legend">' + "".join(items) + "</div>"


# ── Top-level JSON payload builders ───────────────────────────────────────────


def _build_profiles_json(graph_data: GraphData) -> list:
    """Serialize TRT optimization profiles for the JS profile selector."""
    result = []
    for prof in graph_data.profiles:
        inputs = []
        for ti in prof.tensor_infos:
            if ti.is_input:
                inputs.append(
                    {
                        "name": ti.name,
                        "min": list(ti.min_shape) if ti.min_shape else None,
                        "opt": list(ti.opt_shape) if ti.opt_shape else None,
                        "max": list(ti.max_shape) if ti.max_shape else None,
                    }
                )
        result.append({"index": prof.index, "inputs": inputs})
    return result


def _build_model_info(graph_data: GraphData) -> dict:
    """Serialize top-level graph metadata for the right-panel default view."""

    def _ti(ti: TensorInfo) -> dict:
        return {
            "name": ti.name,
            "dtype": str(ti.dtype) if ti.dtype is not None else None,
            "shape": [str(d) for d in ti.shape] if ti.shape is not None else None,
        }

    return {
        "title": graph_data.title or "Model",
        "model_type": graph_data.model_type,
        "num_nodes": len(graph_data.nodes),
        "num_inputs": len(graph_data.graph_inputs),
        "num_outputs": len(graph_data.graph_outputs),
        "inputs": [_ti(ti) for ti in graph_data.graph_inputs],
        "outputs": [_ti(ti) for ti in graph_data.graph_outputs],
    }


def _build_graph_adj(graph_data: GraphData, prefix: str = "") -> dict:
    """
    Build top-level successor and predecessor adjacency lists for JS path-finding.

    Returns ``{"succ": {id: [id, ...]}, "pred": {id: [id, ...]}}`` with prefixed
    node IDs.  Only top-level edges are included; subgraph-internal edges are not
    traversed so compound nodes remain opaque to the range-select BFS.
    """
    succ: dict = {}
    pred: dict = {}
    for i in range(len(graph_data.graph_inputs)):
        nid = f"{prefix}input_{i}"
        succ[nid] = []
        pred[nid] = []
    for node in graph_data.nodes:
        nid = f"{prefix}{node.node_id}"
        succ[nid] = []
        pred[nid] = []
    for i in range(len(graph_data.graph_outputs)):
        nid = f"{prefix}output_{i}"
        succ[nid] = []
        pred[nid] = []
    for src, dst, _ in graph_data.edges:
        s, d = f"{prefix}{src}", f"{prefix}{dst}"
        if s in succ and d in pred:
            succ[s].append(d)
            pred[d].append(s)
    return {"succ": succ, "pred": pred}


def _build_tensor_graph(graph_data: GraphData, prefix: str = "") -> dict:
    """
    Build a compact producer/consumer map for the sidebar jump buttons.

    Returns ``{"p": {tensor_name: src_id}, "c": {tensor_name: [dst_id, ...]}}``
    where all node IDs carry the given *prefix*.  Recurses into subgraphs.
    """
    producer: dict = {}
    consumers: dict = {}
    for src, dst, tensor_name in graph_data.edges:
        producer[tensor_name] = f"{prefix}{src}"
        consumers.setdefault(tensor_name, []).append(f"{prefix}{dst}")
    for node in graph_data.nodes:
        for attr_name, val in node.attrs.items():
            if isinstance(val, GraphData):
                sub = _build_tensor_graph(
                    val, _sg_prefix(prefix, node.node_id, attr_name)
                )
                producer.update(sub["p"])
                for tn, ids in sub["c"].items():
                    consumers.setdefault(tn, []).extend(ids)
    return {"p": producer, "c": consumers}


# ── HTML page builder ─────────────────────────────────────────────────────────


def _build_html(
    graph_data: GraphData,
    elements: list,
    model_path: str = None,
    is_served: bool = False,
) -> str:
    """Generate a self-contained HTML page with Cytoscape.js visualization."""
    # In served mode emit only the slim node dict; full details are fetched
    # lazily on click via /node/<id>.  In save mode embed everything so the
    # file works offline with no server.
    if is_served:
        nd_lite_json = json.dumps(_build_node_json_lite(graph_data))
    else:
        nd_lite_json = json.dumps(_build_node_json(graph_data))

    model_info_json = json.dumps(_build_model_info(graph_data))
    profiles_json = json.dumps(_build_profiles_json(graph_data))
    tensor_graph_json = json.dumps(_build_tensor_graph(graph_data))
    graph_adj_json = json.dumps(_build_graph_adj(graph_data))
    title = graph_data.title or "polygraphy inspect model"
    title_json = json.dumps(title)
    legend = _legend_html(graph_data)

    # Detect whether elements carry server-side positions (preset layout).
    has_positions = bool(elements) and "position" in elements[0]

    # Escape '$' for string.Template (model paths may contain dollar signs).
    model_path_json = json.dumps(model_path).replace("$", "$$")
    is_served_json = json.dumps(is_served).replace("$", "$$")
    has_positions_json = json.dumps(has_positions).replace("$", "$$")

    js = _CYTOSCAPE_JS.substitute(
        ND_LITE_JSON=nd_lite_json,
        MODEL_INFO_JSON=model_info_json,
        PROFILES_JSON=profiles_json,
        TENSOR_GRAPH_JSON=tensor_graph_json,
        GRAPH_ADJ_JSON=graph_adj_json,
        TITLE_JSON=title_json,
        ELEMENTS_JSON=json.dumps(elements),
        STYLE_JSON=json.dumps(_CYTOSCAPE_STYLE),
        MODEL_PATH_JSON=model_path_json,
        IS_SERVED_JSON=is_served_json,
        HAS_POSITIONS_JSON=has_positions_json,
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
{_CDN_SCRIPTS}
<style>{_CSS}</style>
</head>
<body>
<div id="pg-server-down"><div id="pg-server-down-box">
  <h2>Server disconnected</h2>
  <p>The Polygraphy viewer server is no longer reachable. You can close this tab.</p>
</div></div>
<div id="pg-loading">
  <div class="pg-spinner"></div>
  <span class="pg-loading-text">Loading graph\u2026</span>
</div>
<div id="pg-root">
<div id="pg-toolbar">
  <span id="pg-title"></span>
  <span id="pg-model-path"></span>
  <select id="pg-profile-select"></select>
</div>
<div id="pg-body">
  <div id="pg-search-panel">
    <div id="pg-search-header">
      <input id="pg-search-input" type="text" placeholder="\U0001f50d  Search\u2026 (Ctrl+F)" />
    </div>
    <div id="pg-filter-bar">
      <label class="pg-filter-item"><input type="checkbox" data-filter="op_type" checked> Op type</label>
      <label class="pg-filter-item"><input type="checkbox" data-filter="name" checked> Name</label>
      <label class="pg-filter-item"><input type="checkbox" data-filter="tensor" checked> Tensor</label>
    </div>
    <div id="pg-search-results"></div>
  </div>
  <div id="pg-canvas-wrap">
    <div id="cy"></div>
    <div id="pg-zoom-controls">
      <button class="pg-zoom-btn pg-icon-only" id="pg-zoom-in"    title="Zoom in">+</button>
      <button class="pg-zoom-btn pg-icon-only" id="pg-zoom-out"   title="Zoom out">&minus;</button>
      <button class="pg-zoom-btn" id="pg-zoom-home"  title="Reset to initial view">&#x2302;<span class="pg-btn-label">Reset</span></button>
      <button class="pg-zoom-btn" id="pg-zoom-reset" title="Fit to screen">&#x229E;<span class="pg-btn-label">Fit</span></button>
    </div>
    <div id="pg-hints"></div>
    <div id="pg-extract-panel">
      <div id="pg-extract-hdr">
        <span>Extract Subgraph</span>
        <button class="pg-extract-btn" id="pg-extract-close">&times;</button>
      </div>
      <div id="pg-extract-body">
        <div id="pg-extract-info"></div>
        <div id="pg-extract-path-row" style="display:none">
          <span id="pg-extract-path-label">Output:</span>
          <textarea id="pg-extract-path" spellcheck="false" rows="1"></textarea>
        </div>
        <div class="pg-cmd-wrap">
          <pre id="pg-extract-cmd"></pre>
          <button id="pg-extract-copy" title="Copy command">&#x2398;</button>
        </div>
        <div id="pg-extract-actions">
          <button class="pg-extract-btn primary" id="pg-extract-run" style="display:none">Extract &amp; Save</button>
          <p id="pg-extract-offline-note" style="display:none"></p>
        </div>
        <div id="pg-extract-status"></div>
      </div>
    </div>
  </div>
  <div id="pg-sidebar">
    <div id="pg-sidebar-hdr">Details</div>
    <div id="pg-details"></div>
    {legend}
  </div>
</div>
</div>
<script>{js}</script>
</body>
</html>"""
