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
"""Static CSS/JS/HTML templates for the Cytoscape.js model viewer."""
from string import Template

# ── Cytoscape stylesheet ────────────────────────────────────────────────────

_CYTOSCAPE_STYLE = [
    # Op nodes: fully invisible canvas; HTML overlay provides all visuals.
    # Selection/highlight uses underlay glow instead of border.
    {
        "selector": "node.op",
        "style": {
            "shape": "round-rectangle",
            "label": "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "font-family": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            "font-size": "9px",
            "color": "transparent",
            "background-opacity": 0,
            "border-width": 0,
            "width": "label",
            "height": "label",
            "padding": "4px",
            "text-wrap": "wrap",
            "text-max-width": "140px",
        },
    },
    {
        "selector": "node.graph-io",
        "style": {
            "shape": "round-rectangle",
            "label": "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "font-family": "'SF Mono', 'Cascadia Code', Consolas, monospace",
            "font-size": "9px",
            "color": "#1e293b",
            "background-color": "data(bg)",
            "border-width": 1.5,
            "border-color": "data(borderColor)",
            "border-opacity": 0.6,
            "width": "label",
            "height": "label",
            "padding": "6px",
        },
    },
    {
        "selector": "node.graph-io.input",
        "style": {
            "border-style": "solid",
        },
    },
    {
        "selector": "node.graph-io.output",
        "style": {
            "border-style": "solid",
        },
    },
    {
        "selector": "node.compound",
        "style": {
            "shape": "round-rectangle",
            "label": "data(label)",
            "background-color": "data(bg)",
            "background-opacity": 0.06,
            "border-width": 2,
            "border-style": "dashed",
            "border-color": "data(borderColor)",
            "text-valign": "top",
            "text-halign": "center",
            "font-family": "'Inter', -apple-system, sans-serif",
            "font-size": "11px",
            "font-weight": "bold",
            "color": "#64748b",
            "padding": "20px",
            "text-wrap": "wrap",
            "text-max-width": "200px",
        },
    },
    # When a compound node is collapsed it should look like a normal op node
    # (Cytoscape-native label since HTML labels are not used for compounds).
    {
        "selector": "node.compound.cy-expand-collapse-collapsed-node",
        "style": {
            "background-color": "data(bg)",
            "background-opacity": 1,
            "border-style": "solid",
            "border-width": 1,
            "width": "label",
            "height": "label",
            "text-valign": "center",
            "padding": "5px",
            "font-weight": "600",
            "font-size": "11px",
            "color": "#1e293b",
        },
    },
    {
        "selector": "node.sg-container",
        "style": {
            "background-opacity": 0.03,
            "background-color": "#94a3b8",
            "border-width": 1,
            "border-style": "dotted",
            "border-color": "#b0b8c4",
            "text-valign": "top",
            "text-halign": "left",
            "font-family": "'Inter', sans-serif",
            "font-size": "10px",
            "color": "#94a3b8",
            "padding": "15px",
        },
    },
    {
        "selector": "edge",
        "style": {
            "width": 1.2,
            "line-color": "#b0b8c4",
            "target-arrow-color": "#b0b8c4",
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.5,
            "curve-style": "bezier",
            "label": "data(label)",
            "font-family": "'SF Mono', 'Cascadia Code', Consolas, monospace",
            "font-size": "8px",
            "color": "#64748b",
            "text-background-color": "#f8f9fb",
            "text-background-opacity": 0.92,
            "text-background-padding": "2px",
            "text-background-shape": "roundrectangle",
            "text-border-width": 0,
            "text-margin-x": 12,
            "text-margin-y": 0,
        },
    },
    {
        "selector": "node:selected",
        "style": {
            "underlay-color": "#3b82f6",
            "underlay-opacity": 0.22,
            "underlay-padding": 8,
            "underlay-shape": "round-rectangle",
            "border-width": 2,
            "border-color": "#3b82f6",
            "border-opacity": 0.7,
        },
    },
    {
        "selector": "node.highlighted",
        "style": {
            "underlay-color": "#f97316",
            "underlay-opacity": 0.13,
            "underlay-padding": 4,
            "underlay-shape": "round-rectangle",
        },
    },
    {
        "selector": "node.pg-active",
        "style": {
            "underlay-color": "#3b82f6",
            "underlay-opacity": 0.22,
            "underlay-padding": 8,
            "underlay-shape": "round-rectangle",
            "border-width": 2,
            "border-color": "#3b82f6",
            "border-opacity": 0.7,
        },
    },
    {
        "selector": "edge.selected-internal",
        "style": {
            "line-color": "#fdba74",
            "target-arrow-color": "#fdba74",
            "width": 2.0,
            "opacity": 0.85,
        },
    },
    {
        "selector": "edge.boundary-input",
        "style": {
            "line-color": "#22c55e",
            "target-arrow-color": "#22c55e",
            "width": 2.5,
        },
    },
    {
        "selector": "edge.boundary-output",
        "style": {
            "line-color": "#ef4444",
            "target-arrow-color": "#ef4444",
            "width": 2.5,
        },
    },
    {
        "selector": "edge.pg-edge-highlighted",
        "style": {
            "line-color": "#f59e0b",
            "target-arrow-color": "#f59e0b",
            "width": 2.5,
        },
    },
]


# ── CSS ──────────────────────────────────────────────────────────────────────

_CSS = """
html, body { margin: 0; padding: 0; height: 100%; overflow: hidden;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }

#pg-root   { display: flex; flex-direction: column; height: 100vh; }

#pg-toolbar { display: flex; align-items: center; gap: 12px;
  padding: 8px 16px; background: #1e1e2e; color: #f0f0f5;
  font-size: 13px; flex-shrink: 0;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15); z-index: 10; }
#pg-title  { font-weight: 600; color: #f0f0f5; letter-spacing: -0.01em;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  max-width: 520px; flex-shrink: 0; }
#pg-model-path { display: none; font-family: 'SF Mono', 'Cascadia Code', Consolas, monospace;
  font-size: 11px; color: rgba(255,255,255,0.40); white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; min-width: 0; flex-shrink: 1;
  border-left: 1px solid rgba(255,255,255,0.15); padding-left: 12px; margin-left: 4px;
  cursor: default; }
#pg-profile-select { display: none; padding: 4px 10px; border-radius: 6px;
  border: 1px solid rgba(255,255,255,0.15); background: rgba(255,255,255,0.08);
  color: #fff; font-size: 12px; cursor: pointer; outline: none; flex-shrink: 0;
  transition: background 0.15s; }
#pg-profile-select:hover { background: rgba(255,255,255,0.14); }

#pg-body   { display: flex; flex: 1; overflow: hidden; }

/* ── Left search panel ── */
#pg-search-panel { width: 210px; flex-shrink: 0; display: flex; flex-direction: column;
  border-right: 1px solid #e2e8f0; background: #f8fafc; }
#pg-search-header { position: relative; display: flex; align-items: center;
  margin: 10px 10px 6px; }
#pg-search-input { flex: 1; min-width: 0; padding: 7px 11px;
  border-radius: 8px; border: 1px solid #e2e8f0; background: #fff;
  font-size: 12px; outline: none; color: #1e293b;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05); transition: border-color 0.15s, box-shadow 0.15s; }
#pg-search-input:focus { border-color: #93c5fd;
  box-shadow: 0 0 0 3px rgba(59,130,246,0.1); }
#pg-search-input::placeholder { color: #94a3b8; }
#pg-filter-bar { display: flex; gap: 0; margin: 0 10px 8px;
  border: 1px solid #e2e8f0; border-radius: 7px; overflow: hidden;
  background: #f8fafc; }
.pg-filter-item { flex: 1; display: flex; align-items: center; justify-content: center;
  gap: 5px; padding: 4px 6px; font-size: 10px; color: #64748b;
  cursor: pointer; user-select: none; border-right: 1px solid #e2e8f0;
  transition: background 0.12s, color 0.12s; }
.pg-filter-item:last-child { border-right: none; }
.pg-filter-item:hover { background: #eff6ff; color: #3b82f6; }
.pg-filter-item input[type=checkbox] { cursor: pointer; accent-color: #3b82f6; margin: 0; }
#pg-search-results { flex: 1; overflow-y: auto; padding: 0 4px 8px; }
.pg-sr-empty { padding: 8px 10px; font-size: 11px; color: #94a3b8; font-style: italic; }
.pg-sr-item { padding: 6px 10px; border-radius: 6px; margin-bottom: 2px;
  cursor: pointer; font-size: 11px; line-height: 1.4; transition: background 0.12s;
  display: flex; align-items: flex-start; gap: 6px; }
.pg-sr-item:hover { background: #e2e8f0; }
.pg-sr-icon { display: inline-flex; align-items: center; justify-content: center;
  width: 15px; height: 15px; border-radius: 3px; font-size: 9px; font-weight: 700;
  flex-shrink: 0; margin-top: 1px; }
.pg-sr-icon-op         { background: #dbeafe; color: #1d4ed8; }  /* op type match      */
.pg-sr-icon-name       { background: #ede9fe; color: #6d28d9; }  /* node name match    */
.pg-sr-icon-tensor-in  { background: #fef3c7; color: #92400e; }  /* input tensor match */
.pg-sr-icon-tensor-out { background: #f0fdf4; color: #166534; }  /* output tensor match*/
.pg-sr-item .pg-sr-text { min-width: 0; }
.pg-sr-item .pg-sr-op { font-weight: 600; color: #1e293b; }
.pg-sr-item .pg-sr-name { color: #64748b; font-size: 10px;
  font-family: 'SF Mono', Consolas, monospace; display: block;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

#pg-canvas-wrap { flex: 3; position: relative; min-width: 0; overflow: hidden; }
#cy { width: 100%; height: 100%; background: #f8f9fb; }

#pg-zoom-controls { position: absolute; bottom: 12px; left: 12px; z-index: 9999;
  display: flex; flex-direction: column; gap: 4px; pointer-events: none; }
.pg-zoom-btn { height: 30px; border-radius: 8px;
  border: 1px solid #e2e8f0; background: rgba(255,255,255,0.95); cursor: pointer;
  font-size: 15px; line-height: 1; color: #475569;
  box-shadow: 0 2px 6px rgba(0,0,0,0.08);
  display: flex; align-items: center; justify-content: center; gap: 5px;
  pointer-events: auto; transition: background 0.12s, box-shadow 0.12s;
  padding: 0 8px; white-space: nowrap; }
.pg-zoom-btn:hover { background: #f1f5f9; box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
.pg-zoom-btn .pg-btn-label { font-size: 10px; font-weight: 500; letter-spacing: 0.02em; }
.pg-zoom-btn.pg-icon-only { width: 30px; padding: 0; }
#pg-hints { position: absolute; top: 10px; left: 12px; z-index: 9999;
  pointer-events: none; display: flex; flex-direction: column; gap: 3px; }
#pg-hints span { font-size: 11px; font-style: italic; color: #b0b8c8; }

#pg-sidebar { width: 310px; flex-shrink: 0; display: flex; flex-direction: column;
  border-left: 1px solid #e2e8f0; background: #fff;
  box-shadow: -1px 0 4px rgba(0,0,0,0.04); }
#pg-sidebar-hdr { padding: 10px 16px; font-weight: 600; font-size: 12px;
  color: #64748b; letter-spacing: 0.04em; text-transform: uppercase;
  border-bottom: 1px solid #e2e8f0; background: #fafbfc; }
#pg-details { flex: 1; overflow-y: auto; padding: 12px 14px;
  font-size: 12px; line-height: 1.6; color: #1e293b; }

.pg-op   { font-size: 15px; font-weight: 700; color: #1e293b; margin-bottom: 2px; }
.pg-name { font-size: 11px; color: #64748b; margin-bottom: 10px; word-break: break-all; }
.pg-sec  { font-weight: 600; color: #64748b; margin: 12px 0 5px;
  text-transform: uppercase; font-size: 10px; letter-spacing: .06em;
  border-bottom: 1px solid #f1f5f9; padding-bottom: 3px; }
.pg-tensor { position: relative; padding: 5px 28px 5px 8px; margin-bottom: 4px; border-radius: 6px;
  background: #f8fafc; border: 1px solid #f1f5f9;
  font-family: 'SF Mono', Consolas, monospace; font-size: 11px; word-break: break-all; }
.pg-tname  { color: #1e293b; font-weight: 600; }
.pg-tmeta  { color: #94a3b8; }
.pg-jump-btn { position: absolute; bottom: 4px; right: 5px;
  background: none; border: 1px solid #e2e8f0;
  border-radius: 4px; cursor: pointer; padding: 1px 4px; font-size: 11px; line-height: 1.4;
  color: #94a3b8; transition: color 0.12s, background 0.12s, border-color 0.12s; }
.pg-jump-btn:hover { color: #3b82f6; background: #eff6ff; border-color: #bfdbfe; }
.pg-attr   { padding: 4px 8px; margin-bottom: 3px; border-radius: 6px;
  background: #fafbfc; border-left: 2px solid #e2e8f0;
  font-family: 'SF Mono', Consolas, monospace; font-size: 11px; word-break: break-all; }
.pg-akey   { color: #0369a1; }
.pg-aval   { color: #475569; }
.pg-trt    { margin-top: 10px; padding: 7px 9px; border-radius: 6px;
  background: #fffbeb; border: 1px solid #fef3c7;
  font-size: 11px; font-family: 'SF Mono', Consolas, monospace; }

.pg-weight-details { margin-top: 4px; }
.pg-weight-summary { font-size: 10px; color: #64748b; cursor: pointer; user-select: none; }
.pg-weight { font-family: 'SF Mono', Consolas, monospace; font-size: 11px;
  background: #f1f5f9; border-radius: 4px; padding: 4px 6px; margin-top: 3px;
  overflow-x: auto; white-space: pre-wrap; word-break: break-all; color: #334155; }

#pg-legend { display: flex; flex-wrap: wrap; gap: 6px 12px;
  padding: 8px 14px; border-top: 1px solid #e2e8f0; background: #fafbfc;
  flex-shrink: 0; }
.pg-li { display: flex; align-items: center; gap: 5px; white-space: nowrap; }
.pg-sw { width: 10px; height: 10px; border-radius: 3px; flex-shrink: 0; }
.pg-ll { font-size: 10px; color: #64748b; }

/* ── HTML node overlay cards ── */
.pg-node-card { display: flex; flex-direction: column; pointer-events: none;
  border-radius: 5px; overflow: hidden; min-width: 40px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.10), 0 0 0 1px rgba(0,0,0,0.06); }
.pg-node-hdr { padding: 2px 7px;
  font-family: 'Inter', -apple-system, sans-serif;
  font-size: 9px; font-weight: 600; color: #1e293b; text-align: center;
  white-space: nowrap; }
.pg-node-body { padding: 2px 7px 3px;
  font-family: 'SF Mono', 'Cascadia Code', Consolas, monospace;
  font-size: 7.5px; color: #475569; background: #fff; line-height: 1.4; }

/* ── Server-down overlay ── */
#pg-server-down { display: none; position: fixed; inset: 0; z-index: 99999;
  background: rgba(15,23,42,0.82); align-items: center; justify-content: center; }
#pg-server-down.visible { display: flex; }
#pg-server-down-box { background: #fff; border-radius: 12px; padding: 32px 40px;
  text-align: center; max-width: 340px; }
#pg-server-down-box h2 { margin: 0 0 8px; font-size: 17px; color: #0f172a; }
#pg-server-down-box p { margin: 0; font-size: 13px; color: #64748b; }

/* ── Extract subgraph panel ── */
#pg-extract-panel { display: none; position: absolute; bottom: 16px; right: 16px;
  z-index: 9999; width: 540px; max-height: 65vh; overflow-y: auto;
  background: #fff; border-radius: 12px; box-shadow: 0 6px 32px rgba(0,0,0,0.20);
  border: 1px solid #e2e8f0; font-size: 13px; }
#pg-extract-hdr { padding: 12px 16px; font-weight: 600; font-size: 15px;
  border-bottom: 1px solid #e2e8f0; display: flex; justify-content: space-between;
  align-items: center; }
#pg-extract-body { padding: 14px 16px; }
#pg-extract-info { margin-bottom: 10px; line-height: 1.7; font-size: 13px; }
#pg-extract-info b { color: #1e293b; }
#pg-extract-info .pg-boundary-sec { font-weight: 600; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.04em; color: #64748b;
  margin: 8px 0 4px; display: flex; align-items: center; gap: 6px; }
#pg-extract-info .pg-boundary-sec .pg-bdot { width: 8px; height: 8px;
  border-radius: 50%; flex-shrink: 0; }
.pg-tensor-pill { display: inline-block; padding: 2px 8px; margin: 2px 3px 2px 0;
  border-radius: 10px; font-family: 'SF Mono', Consolas, monospace; font-size: 11px;
  border: 1px solid; line-height: 1.4; }
.pg-tensor-pill.pg-tp-in  { background: #f0fdf4; border-color: #bbf7d0; color: #166534; }
.pg-tensor-pill.pg-tp-out { background: #fef2f2; border-color: #fecaca; color: #991b1b; }
.pg-cmd-wrap { position: relative; margin: 10px 0; }
#pg-extract-cmd { background: #1e1e2e; color: #e2e8f0; padding: 12px 36px 12px 14px;
  border-radius: 8px; font-family: 'SF Mono', Consolas, monospace;
  font-size: 12px; white-space: pre-wrap; word-break: break-all;
  max-height: 220px; overflow-y: auto; margin: 0; }
#pg-extract-copy { position: absolute; top: 8px; right: 8px;
  background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.25);
  border-radius: 6px; padding: 5px 7px; cursor: pointer; color: #cbd5e1;
  font-size: 15px; line-height: 1; transition: background 0.12s, color 0.12s; }
#pg-extract-copy:hover { background: rgba(255,255,255,0.25); color: #f1f5f9; }
#pg-extract-copy.copied { color: #4ade80; background: rgba(74,222,128,0.15);
  border-color: rgba(74,222,128,0.3); }
#pg-extract-actions { display: flex; gap: 8px; margin-top: 10px; align-items: center; }
#pg-extract-offline-note { margin: 0; font-size: 11px; font-style: italic; color: #94a3b8; }
.pg-extract-btn { padding: 7px 16px; border-radius: 8px; border: 1px solid #e2e8f0;
  background: #f8fafc; cursor: pointer; font-size: 13px; font-weight: 500;
  transition: background 0.12s; }
.pg-extract-btn:hover { background: #e2e8f0; }
.pg-extract-btn.primary { background: #3b82f6; color: #fff; border-color: #3b82f6; }
.pg-extract-btn.primary:hover { background: #2563eb; }
#pg-extract-path-row { display: flex; align-items: center; gap: 6px; margin-bottom: 10px;
  font-size: 12px; color: #475569; }
#pg-extract-path-label { font-weight: 600; flex-shrink: 0; }
#pg-extract-path { flex: 1; min-width: 0; font-family: 'SF Mono', Consolas, monospace;
  font-size: 12px; color: #1e293b; background: #f8fafc; padding: 5px 10px;
  border-radius: 6px; border: 1px solid #e2e8f0; outline: none;
  resize: none; overflow: hidden; word-break: break-all; line-height: 1.5;
  transition: border-color 0.15s, box-shadow 0.15s, background 0.15s; }
#pg-extract-path:hover { background: #fff; border-color: #cbd5e1; }
#pg-extract-path:focus { background: #fff; border-color: #93c5fd;
  box-shadow: 0 0 0 3px rgba(59,130,246,0.1); }
#pg-extract-status { margin-top: 8px; font-size: 12px; color: #64748b; }
.pg-extract-warn { margin-top: 8px; padding: 8px 10px; border-radius: 6px;
  background: #fffbeb; border: 1px solid #fef3c7; font-size: 11px; color: #92400e; }

/* ── Loading overlay ── */
#pg-loading { position: fixed; inset: 0; z-index: 99999; background: #f8f9fb;
  display: flex; align-items: center; justify-content: center;
  flex-direction: column; gap: 12px; }
.pg-spinner { width: 36px; height: 36px; border: 3px solid #e2e8f0;
  border-top-color: #3b82f6; border-radius: 50%;
  animation: pg-spin 0.7s linear infinite; }
@keyframes pg-spin { to { transform: rotate(360deg); } }
.pg-loading-text { font-size: 13px; color: #64748b; }
"""


# ── JavaScript (uses string.Template ${VAR} substitution) ────────────────────

_CYTOSCAPE_JS = Template(
    """
(function() {
  // ND_LITE: slim node dict (op_type, name, kind, basic tensor info).
  // In save mode this contains the full details; in served mode full details
  // are fetched lazily from /node/<id> and cached in ND_CACHE on first click.
  var ND_LITE = ${ND_LITE_JSON};
  var ND_CACHE = {};
  var MI = ${MODEL_INFO_JSON};
  var PROFILES = ${PROFILES_JSON};
  // TG: tensor producer/consumer map pre-built in Python from graph edges.
  // TG.p[tensorName] = producing node ID; TG.c[tensorName] = [consuming node IDs].
  // Declared in outer scope so fmtTensor can reference them immediately.
  var _tg = ${TENSOR_GRAPH_JSON};
  var tensorProducer  = _tg.p;
  var tensorConsumers = _tg.c;
  // GRAPH_ADJ: top-level successor/predecessor adjacency lists pre-built in
  // Python from graph edges.  Used for Shift+click range-select BFS.
  var _ga = ${GRAPH_ADJ_JSON};
  var adj_succ = _ga.succ;
  var adj_pred = _ga.pred;
  var ELEMENTS = ${ELEMENTS_JSON};
  var STYLE = ${STYLE_JSON};
  var MODEL_PATH = ${MODEL_PATH_JSON};
  var IS_SERVED = ${IS_SERVED_JSON};
  // HAS_POSITIONS: true when elements carry server-side positions (use preset
  // layout); false when positions are absent (fall back to dagre).
  var HAS_POSITIONS = ${HAS_POSITIONS_JSON};

  // Sidebar rendering helpers
  function esc(s) {
    return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }
  function fmtWeightPreview(wp) {
    if (!wp) return '';
    var nums = wp.values.map(function(v) {
      return (typeof v === 'number') ? (Number.isInteger(v) ? String(v) : v.toPrecision(4)) : esc(String(v));
    });
    var display;
    if (wp.truncated) {
      var head = nums.slice(0, wp.split);
      var tail = nums.slice(wp.split);
      display = '[' + head.join(', ') + ',\\u00a0\\u2026,\\u00a0' + tail.join(', ') + ']';
    } else {
      display = '[' + nums.join(', ') + ']';
    }
    var label = wp.total.toLocaleString() + ' element(s)' + (wp.truncated ? ' \\u2014 showing first/last ' + wp.split : '');
    return '<details class="pg-weight-details"><summary class="pg-weight-summary">' + label + '</summary>'
      + '<div class="pg-weight">' + display + '</div>'
      + '</details>';
  }
  function fmtTensor(t, isOutput) {
    var meta = [];
    if (t.dtype)     meta.push(t.dtype);
    if (t.shape)     meta.push(t.shape.join('\\u00d7') || 'scalar');
    if (t.docstring) meta.push(t.docstring);
    // Jump button: ↑ to producer for inputs, ↓ to first consumer for outputs.
    var jumpTarget = isOutput ? (tensorConsumers[t.name] && tensorConsumers[t.name][0])
                              : tensorProducer[t.name];
    var jumpBtn = jumpTarget
      ? '<button class="pg-jump-btn" data-target="' + esc(jumpTarget) + '" title="'
          + (isOutput ? 'Jump to consumer' : 'Jump to producer') + '">'
          + (isOutput ? '\\u2193' : '\\u2191') + '</button>'
      : '';
    return '<div class="pg-tensor">'
      + '<span class="pg-tname">' + esc(t.name) + '</span>'
      + (meta.length ? '<br><span class="pg-tmeta">' + esc(meta.join('  ')) + '</span>' : '')
      + (t.weight_preview ? fmtWeightPreview(t.weight_preview) : '')
      + jumpBtn
      + '</div>';
  }
  function fmtAttr(a) {
    return '<div class="pg-attr"><span class="pg-akey">' + esc(a.key) + '</span>'
      + ': <span class="pg-aval">' + esc(a.val) + '</span>'
      + (a.weight_preview ? fmtWeightPreview(a.weight_preview) : '')
      + '</div>';
  }
  function renderNode(d) {
    var h = '<div class="pg-op">' + esc(d.op_type) + '</div>';
    // Skip the name subtitle for I/O nodes — fmtTensor already shows the tensor name.
    if (d.name && d.kind !== 'input' && d.kind !== 'output') h += '<div class="pg-name">' + esc(d.name) + '</div>';
    if (d.kind === 'input' || d.kind === 'output') {
      // Graph I/O pseudo-nodes: the node *is* the tensor.
      // Reuse fmtTensor so the jump-to-consumer / jump-to-producer button appears.
      // input nodes "produce" their tensor (isOutput=true → jump to consumer ↓).
      // output nodes "consume" their tensor (isOutput=false → jump to producer ↑).
      var ioTensor = (d.kind === 'input') ? d.outputs[0] : d.inputs[0];
      if (ioTensor) h += fmtTensor(ioTensor, d.kind === 'input');
    } else {
      if (d.inputs  && d.inputs.length)  { h += '<div class="pg-sec">Inputs</div>';  d.inputs.forEach(function(t){h+=fmtTensor(t, false);}); }
      if (d.outputs && d.outputs.length) { h += '<div class="pg-sec">Outputs</div>'; d.outputs.forEach(function(t){h+=fmtTensor(t, true);}); }
    }
    if (d.attrs   && d.attrs.length)   { h += '<div class="pg-sec">Attributes</div>'; d.attrs.forEach(function(a){h+=fmtAttr(a);}); }
    if (d.subgraphs && Object.keys(d.subgraphs).length) {
      h += '<div class="pg-sec">Subgraphs</div>';
      Object.keys(d.subgraphs).forEach(function(attrName) {
        var sg = d.subgraphs[attrName];
        h += '<div class="pg-attr"><span class="pg-akey">' + esc(attrName) + '</span>'
           + ': <span class="pg-tmeta">' + sg.all_node_ids.length + ' nodes'
           + ' \u2014 click \u271a on the graph node to expand</span></div>';
      });
    }
    if (d.origin || d.tactic) {
      h += '<div class="pg-trt">';
      if (d.origin) h += '<b>Origin:</b> ' + esc(d.origin) + '<br>';
      if (d.tactic) h += '<b>Tactic:</b> ' + esc(d.tactic);
      h += '</div>';
    }
    return h;
  }
  function renderModelInfo() {
    var h = '<div class="pg-op">' + esc(MI.title) + '</div>';
    if (MI.model_type) h += '<div class="pg-name">Type: ' + esc(MI.model_type) + '</div>';
    h += '<div class="pg-sec">' + MI.num_nodes + ' Node(s) &nbsp;&middot;&nbsp; '
       + MI.num_inputs + ' Input(s) &nbsp;&middot;&nbsp; ' + MI.num_outputs + ' Output(s)</div>';
    if (MI.inputs && MI.inputs.length)  { h += '<div class="pg-sec">Graph Inputs</div>';  MI.inputs.forEach(function(t){h+=fmtTensor(t, true);}); }
    if (MI.outputs && MI.outputs.length){ h += '<div class="pg-sec">Graph Outputs</div>'; MI.outputs.forEach(function(t){h+=fmtTensor(t, false);}); }
    return h;
  }

  // Defer all heavy initialisation so the loading spinner paints first.
  requestAnimationFrame(function() { setTimeout(function() {

  var det   = document.getElementById('pg-details');
  var title = document.getElementById('pg-title');
  if (title) title.textContent = ${TITLE_JSON};
  var pathEl = document.getElementById('pg-model-path');
  if (pathEl && MODEL_PATH) {
    var _displayPath = MODEL_PATH.replace(/\\\\/g, '/');
    pathEl.textContent = _displayPath;
    pathEl.title = MODEL_PATH;
    pathEl.style.display = 'block';
  }

  // Show model info by default
  det.innerHTML = renderModelInfo();

  // Initialize Cytoscape
  var cy = cytoscape({
    container: document.getElementById('cy'),
    elements: ELEMENTS,
    style: STYLE,
    layout: { name: 'preset' },
    wheelSensitivity: 0.3,
    autoungrabify: true,
    boxSelectionEnabled: false,
    userZoomingEnabled: false,  // all zoom/pan handled by our custom wheel listener
    maxZoom: 4,
    minZoom: 0.1,
  });

  // Register dagre layout (kept for compound-node re-layout and fallback).
  if (typeof cytoscape !== 'undefined' && typeof cytoscapeDagre !== 'undefined') {
    cytoscape.use(cytoscapeDagre);
  }

  // Register HTML node labels for op nodes (colored header + attribute body).
  if (typeof cytoscapeNodeHtmlLabel !== 'undefined') {
    cytoscape.use(cytoscapeNodeHtmlLabel);
  }
  cy.nodeHtmlLabel([{
    query: 'node.op',
    halign: 'center',
    valign: 'center',
    tpl: function(data) {
      var h = '<div class="pg-node-card">'
        + '<div class="pg-node-hdr" style="background:'
        + (data.bg || '#e8ecf0') + '">' + esc(data.opType || '') + '</div>';
      if (data.bodyHtml) h += '<div class="pg-node-body">' + data.bodyHtml + '</div>';
      h += '</div>';
      return h;
    }
  }]);

  // Use preset positions when available (server-side layout); fall back to
  // dagre for compound-node graphs where server-side layout is skipped.
  var layoutOpts = HAS_POSITIONS
    ? { name: 'preset', fit: true, padding: 25 }
    : { name: 'dagre', rankDir: 'TB', nodeSep: 30, rankSep: 45, edgeSep: 12,
        animate: false, fit: true, padding: 25 };
  cy.layout(layoutOpts).run();

  // Register and configure expand-collapse
  if (typeof cytoscapeExpandCollapse !== 'undefined') {
    cytoscape.use(cytoscapeExpandCollapse);
  }
  // Always use dagre for expand/collapse re-layout (positions of newly
  // revealed children are not pre-computed server-side).
  var layoutByOpts = { name: 'dagre', rankDir: 'TB', nodeSep: 30, rankSep: 45,
                       edgeSep: 12, animate: false, fit: false, padding: 25 };
  var ecApi = cy.expandCollapse({
    layoutBy: function() {
      cy.layout(layoutByOpts).run();
    },
    fisheye: false,
    animate: false,
    undoable: false,
    cueEnabled: true,
    expandCollapseCueSize: 14,
    expandCollapseCuePosition: 'top-left',
    expandCollapseCueSensitivity: 1,
  });

  // Collapse all compound nodes initially
  cy.nodes('.compound, .sg-container').forEach(function(n) {
    if (n.isParent()) {
      try { ecApi.collapse(n); } catch(e) {}
    }
  });

  // Position the initial view near the graph inputs at a readable zoom level.
  // Fall back to a full fit if no input nodes exist.
  var _initialViewport = null;
  (function() {
    var inputNodes = cy.nodes('.graph-io.input');
    if (inputNodes.length === 0) {
      cy.fit(undefined, 25);
      return;
    }
    // Fixed zoom: calibrated so a hypothetical 4-char input node ("node")
    // with model-space width ~50 units renders at 75px screen pixels.
    // 75px / 50 model-units = zoom 1.5.
    var targetZoom = 1.5;
    var bb = inputNodes.boundingBox();
    // Pan so the input nodes sit ~15% from the top of the viewport.
    var vpH = cy.height();
    var vpW = cy.width();
    var _initPan = {
      x: vpW / 2 - (bb.x1 + bb.w / 2) * targetZoom,
      y: vpH * 0.15 - bb.y1 * targetZoom,
    };
    cy.viewport({ zoom: targetZoom, pan: _initPan });
    _initialViewport = { zoom: targetZoom, pan: _initPan };
  })();

  // Remove loading overlay now that the graph is ready.
  var loadingEl = document.getElementById('pg-loading');
  if (loadingEl) loadingEl.style.display = 'none';

  // Event delegation for jump buttons rendered inside the sidebar.
  // tensorProducer / tensorConsumers are pre-built in Python (see TG var above).
  det.addEventListener('click', function(e) {
    var btn = e.target.closest('.pg-jump-btn');
    if (!btn) return;
    var targetId = btn.getAttribute('data-target');
    if (!targetId) return;
    var targetNode = cy.getElementById(targetId);
    if (!targetNode || !targetNode.length) return;
    cy.nodes(':selected').unselect();
    cy.edges().removeClass('boundary-input boundary-output selected-internal');
    cy.nodes('.highlighted').removeClass('highlighted');
    cy.nodes('.pg-active').removeClass('pg-active');
    targetNode.addClass('highlighted').addClass('pg-active');
    cy.animate({ fit: { eles: targetNode, padding: 80 }, duration: 300 });
    loadNodeAndRender(targetId);
  });

  // ── Lazy node detail loader ───────────────────────────────────────────────
  // In served mode, ND_LITE contains only slim info (no attrs); full details
  // are fetched from /node/<id> on first click and cached in ND_CACHE.
  // In save mode (!IS_SERVED), ND_LITE already contains the full details.
  //
  // A monotone generation counter ensures that if the user clicks a different
  // node while a fetch is in flight, the stale response never overwrites the
  // sidebar for the new node.
  var _ndRenderGen = 0;
  function loadNodeAndRender(nid) {
    var gen = ++_ndRenderGen;
    var cached = ND_CACHE[nid];
    var lite    = ND_LITE[nid];
    if (!cached && !lite) return;

    if (cached || !IS_SERVED) {
      // Full data available immediately (cached or save-mode).
      det.innerHTML = renderNode(cached || lite);
      return;
    }

    // Served mode, not yet cached: render lite view first, then fetch.
    det.innerHTML = renderNode(lite)
      + '<div class="pg-sec" id="pg-attrs-loading" style="color:#94a3b8">Attributes \u2014 loading\u2026</div>';

    fetch('/node/' + encodeURIComponent(nid))
      .then(function(r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function(full) {
        ND_CACHE[nid] = full;
        if (_ndRenderGen === gen) {
          det.innerHTML = renderNode(full);
        }
      })
      .catch(function() {
        // On error remove the loading placeholder so the lite view is clean.
        if (_ndRenderGen === gen) {
          var pl = document.getElementById('pg-attrs-loading');
          if (pl) pl.remove();
        }
      });
  }

  // Tap on node: show sidebar details and highlight selected node
  cy.on('tap', 'node', function(evt) {
    cy.nodes('.pg-active').removeClass('pg-active');
    evt.target.addClass('pg-active');
    loadNodeAndRender(evt.target.id());
  });
  // Tap on background: restore model info and clear highlight
  cy.on('tap', function(evt) {
    if (evt.target === cy) {
      cy.nodes('.pg-active').removeClass('pg-active');
      det.innerHTML = renderModelInfo();
    }
  });


  // Left search panel: filter nodes and show clickable list
  var searchInput = document.getElementById('pg-search-input');
  var searchResults = document.getElementById('pg-search-results');
  // Persistent filter bar checkboxes
  var searchFilters = { op_type: true, name: true, tensor: true };
  document.querySelectorAll('#pg-filter-bar input[type=checkbox]').forEach(function(cb) {
    cb.addEventListener('change', function() {
      searchFilters[cb.getAttribute('data-filter')] = cb.checked;
      if (searchInput) showSearchResults(searchInput.value.trim().toLowerCase());
    });
  });
  function showSearchResults(q) {
    searchResults.innerHTML = '';
    cy.nodes('.highlighted').removeClass('highlighted');
    if (!q) {
      searchResults.innerHTML = '<div class="pg-sr-empty">Type to search nodes\u2026</div>';
      return;
    }
    var matches = [];
    cy.nodes().forEach(function(n) {
      var d = ND_LITE[n.id()];
      if (!d) return;
      var matchType = null, matchedTensor = null, matchedTensorDir = null;
      if (searchFilters.op_type && (d.op_type||'').toLowerCase().indexOf(q) >= 0) {
        matchType = 'op_type';
      } else if (searchFilters.name && (d.name||'').toLowerCase().indexOf(q) >= 0) {
        matchType = 'name';
      } else if (searchFilters.tensor) {
        // Search inputs first, then outputs; track which side matched.
        var ins = d.inputs || [], outs = d.outputs || [];
        for (var ti = 0; ti < ins.length; ti++) {
          if ((ins[ti].name||'').toLowerCase().indexOf(q) >= 0) {
            matchType = 'tensor'; matchedTensor = ins[ti].name; matchedTensorDir = 'in'; break;
          }
        }
        if (!matchType) {
          for (var ti = 0; ti < outs.length; ti++) {
            if ((outs[ti].name||'').toLowerCase().indexOf(q) >= 0) {
              matchType = 'tensor'; matchedTensor = outs[ti].name; matchedTensorDir = 'out'; break;
            }
          }
        }
      }
      if (matchType) matches.push({ node: n, d: d, matchType: matchType, matchedTensor: matchedTensor, matchedTensorDir: matchedTensorDir });
    });
    if (!matches.length) {
      searchResults.innerHTML = '<div class="pg-sr-empty">No matches.</div>';
      return;
    }
    matches.forEach(function(m) {
      var el = document.createElement('div');
      el.className = 'pg-sr-item';
      var iconCls, iconLbl, iconTitle;
      if (m.matchType === 'op_type') {
        iconCls = 'pg-sr-icon-op';   iconLbl = 'op'; iconTitle = 'Matched op type';
      } else if (m.matchType === 'name') {
        iconCls = 'pg-sr-icon-name'; iconLbl = 'id'; iconTitle = 'Matched node name';
      } else if (m.matchedTensorDir === 'out') {
        iconCls = 'pg-sr-icon-tensor-out'; iconLbl = 'T\u2192'; iconTitle = 'Matched output tensor';
      } else {
        iconCls = 'pg-sr-icon-tensor-in';  iconLbl = '\u2192T'; iconTitle = 'Matched input tensor';
      }
      el.innerHTML = '<span class="pg-sr-icon ' + iconCls + '" title="' + iconTitle + '">' + iconLbl + '</span>'
        + '<div class="pg-sr-text"><span class="pg-sr-op">' + esc(m.d.op_type) + '</span>'
        + (m.d.name ? '<span class="pg-sr-name">' + esc(m.d.name) + '</span>' : '') + '</div>';
      el.addEventListener('click', function() {
        cy.nodes('.highlighted').removeClass('highlighted');
        cy.edges('.pg-edge-highlighted').removeClass('pg-edge-highlighted');
        cy.nodes('.pg-active').removeClass('pg-active');
        if (m.matchType === 'tensor' && m.matchedTensor) {
          // Strip "Initializer | " prefix before querying edge data.
          var tn = m.matchedTensor.replace(/^Initializer \| /, '');
          var tedges = cy.edges().filter(function(e) { return e.data('tensorName') === tn; });
          if (tedges.length) {
            tedges.addClass('pg-edge-highlighted');
            tedges.sources().union(tedges.targets()).addClass('highlighted');
          }
          // Output tensor match → jump to producer; input tensor match → jump
          // to the matched node itself (the consumer shown in the result).
          var jumpId = (m.matchedTensorDir === 'out') ? tensorProducer[tn] : m.node.id();
          var jumpNode = jumpId ? cy.getElementById(jumpId) : null;
          if (jumpNode && jumpNode.length) {
            jumpNode.addClass('pg-active');
            cy.animate({ fit: { eles: jumpNode, padding: 80 }, duration: 300 });
            loadNodeAndRender(jumpId);
          } else if (tedges.length) {
            cy.animate({ fit: { eles: tedges.union(tedges.sources().union(tedges.targets())), padding: 80 }, duration: 300 });
          }
          return;
        }
        // Default: navigate to and select the node.
        m.node.addClass('highlighted').addClass('pg-active');
        cy.animate({ fit: { eles: m.node, padding: 80 }, duration: 300 });
        loadNodeAndRender(m.node.id());
      });
      searchResults.appendChild(el);
    });
    matches.forEach(function(m){ m.node.addClass('highlighted'); });
  }
  if (searchInput) {
    showSearchResults('');
    searchInput.addEventListener('input', function() {
      showSearchResults(this.value.trim().toLowerCase());
    });
  }

  // Profile selector (TRT engines with >1 profile)
  var profileSelect = document.getElementById('pg-profile-select');
  if (PROFILES.length > 1 && profileSelect) {
    PROFILES.forEach(function(p) {
      var opt = document.createElement('option');
      opt.value = p.index;
      opt.textContent = 'Profile ' + p.index;
      profileSelect.appendChild(opt);
    });
    profileSelect.style.display = 'inline-block';

    function shapeStr(arr) {
      if (!arr) return '';
      return arr.map(function(d){ return d === -1 ? '?' : d; }).join('\u00d7');
    }
    function updateForProfile(pIdx) {
      var profile = PROFILES[pIdx];
      if (!profile) return;
      profile.inputs.forEach(function(pti) {
        var optStr = shapeStr(pti.opt);
        // Only update edge labels — node labels stay as the tensor name only.
        // Updating node labels on polygon arrow shapes causes multiline overflow.
        cy.nodes('[tensorName="' + pti.name + '"]').forEach(function(n) {
          cy.edges('[source="' + n.id() + '"]').forEach(function(e) {
            var dtype = e.data('dtype');
            e.data('label', optStr + (dtype ? ' ' + dtype : ''));
          });
        });
      });
    }
    profileSelect.addEventListener('change', function() {
      updateForProfile(parseInt(this.value));
    });
    updateForProfile(0);
  }

  // Zoom buttons
  document.getElementById('pg-zoom-in').addEventListener('click', function() {
    cy.zoom({ level: cy.zoom() * 1.3, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } });
  });
  document.getElementById('pg-zoom-out').addEventListener('click', function() {
    cy.zoom({ level: cy.zoom() / 1.3, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } });
  });
  document.getElementById('pg-zoom-home').addEventListener('click', function() {
    if (_initialViewport) {
      cy.viewport(_initialViewport);
    } else {
      cy.fit(undefined, 25);
    }
  });
  document.getElementById('pg-zoom-reset').addEventListener('click', function() {
    cy.fit(undefined, 25);
  });

  // Handle scroll-wheel: Ctrl+scroll zooms (input-anchored), Shift+scroll
  // pans horizontally, plain scroll pans vertically.
  var cyContainer = document.getElementById('cy');
  cyContainer.addEventListener('wheel', function(e) {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
      var factor = e.deltaY > 0 ? 1 / 1.08 : 1.08;
      var rect = cyContainer.getBoundingClientRect();
      cy.zoom({
        level: cy.zoom() * factor,
        renderedPosition: { x: e.clientX - rect.left, y: e.clientY - rect.top }
      });
    } else if (e.shiftKey) {
      // Shift+scroll: horizontal pan (deltaY drives it so a plain vertical
      // wheel also works when shift is held, as on most platforms).
      cy.panBy({ x: -(e.deltaX || e.deltaY), y: 0 });
    } else {
      cy.panBy({ x: -e.deltaX, y: -e.deltaY });
    }
  }, { passive: false });

  // Right-click pan: works in both normal and extraction (select) mode.
  var rightDragActive = false;
  var rightDragLast = { x: 0, y: 0 };
  cyContainer.addEventListener('mousedown', function(e) {
    if (e.button !== 2) return;
    rightDragActive = true;
    rightDragLast = { x: e.clientX, y: e.clientY };
    e.preventDefault();
  });
  document.addEventListener('mousemove', function(e) {
    if (!rightDragActive) return;
    var dx = e.clientX - rightDragLast.x;
    var dy = e.clientY - rightDragLast.y;
    rightDragLast = { x: e.clientX, y: e.clientY };
    cy.panBy({ x: dx, y: dy });
  });
  document.addEventListener('mouseup', function(e) {
    if (e.button === 2) rightDragActive = false;
  });
  cyContainer.addEventListener('contextmenu', function(e) { e.preventDefault(); });

  // Persistent canvas hints (top-left)
  var hintsEl = document.getElementById('pg-hints');
  var _selectModeActive = false;  // mirrored here so updateHints() can read it
  function updateHints() {
    var lines = _selectModeActive
      ? [
          'Click and drag: select nodes',
          'Shift+Click: range select',
          'Ctrl+Click: multi-select',
          'Right-click: pan',
          'Shift+Scroll: horizontal scroll',
          'Ctrl+Scroll: zoom',
        ]
      : [
          'Click and drag: pan',
          'Right-click: pan',
          'Shift+Scroll: horizontal scroll',
          'Ctrl+Scroll: zoom',
        ];
    hintsEl.innerHTML = lines.map(function(l) { return '<span>' + l + '</span>'; }).join('');
  }
  updateHints();

  // Heartbeat: show overlay if server goes away
  if (IS_SERVED) {
    var _serverDown = false;
    setInterval(function() {
      fetch('/heartbeat').then(function(r) {
        if (!r.ok) throw new Error();
      }).catch(function() {
        if (!_serverDown) {
          _serverDown = true;
          document.getElementById('pg-server-down').classList.add('visible');
        }
      });
    }, 2000);
  }

  // Ctrl+F / Cmd+F → focus search box
  document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
      e.preventDefault();
      if (searchInput) { searchInput.focus(); searchInput.select(); }
    }
  });


  // ── Subgraph extraction (ONNX only) ──────────────────────────────────────
  var extractEnabled = (MI.model_type === 'onnx' && MODEL_PATH !== null);
  if (extractEnabled && !IS_SERVED) {
    var _offlineNote = document.getElementById('pg-extract-offline-note');
    _offlineNote.textContent = 'Copy the command above to run it in your terminal.';
    _offlineNote.style.display = 'block';
  }
  if (extractEnabled) {
    cy.boxSelectionEnabled(true);

    // Extract mode is always active for ONNX models.
    var selectMode = true;
    _selectModeActive = true;
    cy.userPanningEnabled(false);
    updateHints();

    // ── Edge-scroll during box selection drag ─────────────────────────────
    // When the mouse is near the viewport edge while dragging a selection
    // box, auto-pan the canvas so the user can extend the selection past
    // the currently visible area.
    var EDGE_ZONE  = 60;   // px from viewport edge to start scrolling
    var EDGE_SPEED = 10;   // max canvas-px per animation frame at the very edge
    var edgeDragActive = false;
    var edgeScrollDx = 0, edgeScrollDy = 0;
    var edgeScrollRaf = null;

    function edgeScrollStep() {
      if (!edgeDragActive) return;
      if (edgeScrollDx !== 0 || edgeScrollDy !== 0) {
        cy.panBy({ x: edgeScrollDx, y: edgeScrollDy });
      }
      edgeScrollRaf = requestAnimationFrame(edgeScrollStep);
    }

    cyContainer.addEventListener('mousedown', function(e) {
      if (e.button !== 0 || !selectMode) return;
      edgeDragActive = true;
      edgeScrollDx = 0;
      edgeScrollDy = 0;
      edgeScrollRaf = requestAnimationFrame(edgeScrollStep);
    });

    document.addEventListener('mousemove', function(e) {
      if (!edgeDragActive) return;
      var rect = cyContainer.getBoundingClientRect();
      var x = e.clientX - rect.left;
      var y = e.clientY - rect.top;
      // Positive dx/dy = pan canvas right/down = reveal content to the left/top.
      function vel(pos, size) {
        if (pos < EDGE_ZONE)        return  EDGE_SPEED * (1 - pos / EDGE_ZONE);
        if (pos > size - EDGE_ZONE) return -EDGE_SPEED * (1 - (size - pos) / EDGE_ZONE);
        return 0;
      }
      edgeScrollDx = vel(x, rect.width);
      edgeScrollDy = vel(y, rect.height);
    });

    document.addEventListener('mouseup', function(e) {
      if (e.button !== 0 || !edgeDragActive) return;
      edgeDragActive = false;
      edgeScrollDx = 0;
      edgeScrollDy = 0;
      if (edgeScrollRaf) { cancelAnimationFrame(edgeScrollRaf); edgeScrollRaf = null; }
    });

    // Look up tensor info (shape, dtype) from ND_LITE (always present).
    function lookupTensorInfo(tensorName, srcNodeId, tgtNodeId) {
      var nodes = [srcNodeId, tgtNodeId];
      for (var i = 0; i < nodes.length; i++) {
        var nd = ND_LITE[nodes[i]];
        if (!nd) continue;
        var lists = [nd.outputs, nd.inputs];
        for (var j = 0; j < lists.length; j++) {
          if (!lists[j]) continue;
          for (var k = 0; k < lists[j].length; k++) {
            var t = lists[j][k];
            var cleanName = t.name.replace(/^Initializer \\| /, '');
            if (cleanName === tensorName) {
              return { name: tensorName, shape: t.shape, dtype: t.dtype };
            }
          }
        }
      }
      return { name: tensorName, shape: null, dtype: null };
    }

    // Compute boundary input/output tensors for the selected nodes
    function computeBoundaryTensors(selectedNodes) {
      var selIds = {};
      selectedNodes.forEach(function(n) {
        selIds[n.id()] = true;
        // Include descendants of compound nodes
        if (n.isParent()) {
          n.descendants().forEach(function(d) { selIds[d.id()] = true; });
        }
      });

      var inputTensors = {};
      var outputTensors = {};

      // First pass: find tensors that are consumed by a node inside the selection.
      // A tensor consumed internally is an intermediate value, not a subgraph
      // output — even if it also feeds a node outside (skip connection).
      var consumedInside = {};
      cy.edges().forEach(function(e) {
        if (!!selIds[e.data('source')] && !!selIds[e.data('target')]) {
          consumedInside[e.data('tensorName')] = true;
        }
      });

      cy.edges().forEach(function(e) {
        var src = e.data('source');
        var tgt = e.data('target');
        var tName = e.data('tensorName');
        if (!tName) return;

        var srcIn = !!selIds[src];
        var tgtIn = !!selIds[tgt];

        if (!srcIn && tgtIn) {
          if (!inputTensors[tName]) inputTensors[tName] = lookupTensorInfo(tName, src, tgt);
          e.addClass('boundary-input');
        }
        // Only mark as a boundary output if the tensor is not also consumed
        // inside the selection — tensors that feed both an internal node and
        // an external one are skip-connection tensors we don't need to expose.
        if (srcIn && !tgtIn && !consumedInside[tName]) {
          if (!outputTensors[tName]) outputTensors[tName] = lookupTensorInfo(tName, src, tgt);
          e.addClass('boundary-output');
        }
        if (srcIn && tgtIn) {
          e.addClass('selected-internal');
        }
      });

      return {
        inputs: Object.values(inputTensors),
        outputs: Object.values(outputTensors)
      };
    }

    // Build a smart default output filename from model path + boundary tensor names
    function defaultOutputPath(boundary, numNodes) {
      // Get model directory and base name
      var parts = MODEL_PATH.replace(/\\\\/g, '/').split('/');
      var basename = parts.pop();
      var dir = parts.join('/');
      var stem = basename.replace(/\\.onnx$$/i, '');

      // Abbreviate tensor names: take last segment after / or ::, truncate
      function abbrev(name) {
        var s = name.split(/[/:]/).pop();
        if (s.length > 16) return s.substring(0, 8) + '..' + s.substring(s.length - 6);
        return s;
      }

      var tag = numNodes + 'n';
      if (boundary.inputs.length) {
        tag += '_in(' + boundary.inputs.slice(0, 2).map(function(t){ return abbrev(t.name); }).join('+');
        if (boundary.inputs.length > 2) tag += '+' + (boundary.inputs.length - 2) + 'more';
        tag += ')';
      }
      if (boundary.outputs.length) {
        tag += '_out(' + boundary.outputs.slice(0, 2).map(function(t){ return abbrev(t.name); }).join('+');
        if (boundary.outputs.length > 2) tag += '+' + (boundary.outputs.length - 2) + 'more';
        tag += ')';
      }

      var filename = stem + '_' + tag + '.onnx';
      return dir ? dir + '/' + filename : filename;
    }

    var currentOutputPath = '';

    // Generate the polygraphy surgeon extract command
    function generateExtractCommand(boundary, outputPath) {
      var cmd = 'polygraphy surgeon extract ' + JSON.stringify(MODEL_PATH) + ' \\\n  -o ' + outputPath;
      if (boundary.inputs.length) {
        cmd += ' \\\n  --inputs';
        boundary.inputs.forEach(function(t) {
          var shape = (t.shape && t.shape.length) ? '[' + t.shape.join(',') + ']' : 'auto';
          var dtype = t.dtype || 'auto';
          cmd += ' \\\n    ' + t.name + ':' + shape + ':' + dtype;
        });
      }
      if (boundary.outputs.length) {
        cmd += ' \\\n  --outputs';
        boundary.outputs.forEach(function(t) {
          var dtype = t.dtype || 'auto';
          cmd += ' \\\n    ' + t.name + ':' + dtype;
        });
      }
      return cmd;
    }

    // Show extract panel on box selection
    function showExtractPanel() {
      var allSelected = cy.nodes(':selected').filter(function(n) {
        return n.hasClass('op') || n.hasClass('graph-io') || n.hasClass('compound');
      });
      cy.edges().removeClass('boundary-input boundary-output selected-internal');
      if (allSelected.length === 0) {
        document.getElementById('pg-extract-panel').style.display = 'none';
        return;
      }

      // Filter out nodes inside subgraphs — surgeon extract only works on the
      // top-level ONNX graph.  Subgraph-internal node IDs contain "__sg_".
      var selected = allSelected.filter(function(n) {
        return n.id().indexOf('__sg_') === -1;
      });
      var skippedCount = allSelected.length - selected.length;
      var boundary = computeBoundaryTensors(selected);
      currentOutputPath = defaultOutputPath(boundary, selected.length);
      var cmd = generateExtractCommand(boundary, currentOutputPath);

      function tensorPill(t, cls) {
        var meta = '';
        if (t.dtype) meta += t.dtype;
        if (t.shape && t.shape.length) meta += (meta ? ' ' : '') + t.shape.join('\u00d7');
        return '<span class="pg-tensor-pill ' + cls + '" title="' + esc(meta) + '">' + esc(t.name) + '</span>';
      }
      var info = '<b>' + selected.length + ' node(s) selected</b>';
      info += '<div class="pg-boundary-sec"><span class="pg-bdot" style="background:#22c55e"></span>Inputs (' + boundary.inputs.length + ')</div>';
      if (boundary.inputs.length) {
        info += '<div>' + boundary.inputs.map(function(t){ return tensorPill(t, 'pg-tp-in'); }).join('') + '</div>';
      } else {
        info += '<div style="font-size:11px;color:#94a3b8;font-style:italic">none (uses graph inputs)</div>';
      }
      info += '<div class="pg-boundary-sec"><span class="pg-bdot" style="background:#ef4444"></span>Outputs (' + boundary.outputs.length + ')</div>';
      if (boundary.outputs.length) {
        info += '<div>' + boundary.outputs.map(function(t){ return tensorPill(t, 'pg-tp-out'); }).join('') + '</div>';
      } else {
        info += '<div style="font-size:11px;color:#94a3b8;font-style:italic">none (uses graph outputs)</div>';
      }

      if (skippedCount > 0) {
        info += '<div class="pg-extract-warn">' + skippedCount + ' node(s) inside subgraphs (Loop/If/Scan) were excluded &mdash; '
          + '<code>surgeon extract</code> operates on the top-level graph only. '
          + 'Select the parent compound node to include the entire subgraph.</div>';
      }

      if (selected.length === 0) {
        document.getElementById('pg-extract-panel').style.display = 'none';
        return;
      }

      document.getElementById('pg-extract-info').innerHTML = info;
      document.getElementById('pg-extract-cmd').textContent = cmd;
      document.getElementById('pg-extract-panel').style.display = 'block';

      // Show output path + Extract & Save button when served
      if (IS_SERVED) {
        document.getElementById('pg-extract-path').value = currentOutputPath;
        document.getElementById('pg-extract-path-row').style.display = 'flex';
        document.getElementById('pg-extract-run').style.display = 'inline-block';
        setTimeout(autoResizePath, 0);
      }
      document.getElementById('pg-extract-status').textContent = '';
    }

    cy.on('boxend', showExtractPanel);
    // Also update when selection changes via click
    cy.on('select unselect', 'node', function() {
      if (selectMode) {
        setTimeout(showExtractPanel, 50);
      }
    });

    // ── Shift+click range select ───────────────────────────────────────────
    // Selects all top-level nodes that lie on some directed path between
    // the anchor (last plain-clicked node) and the shift-clicked node.
    // Uses forward + backward BFS intersection so only nodes on the actual
    // connecting path are included — parallel siblings at the same depth
    // are NOT selected.  Works in both directions (anchor above or below
    // the shift-clicked node).

    // adj_succ / adj_pred are pre-built in Python and embedded as GRAPH_ADJ_JSON.
    function bfsReach(startId, adjMap) {
      var visited = {}; visited[startId] = true;
      var q = [startId];
      while (q.length) {
        var cur = q.shift();
        (adjMap[cur] || []).forEach(function(nb) {
          if (!visited[nb]) { visited[nb] = true; q.push(nb); }
        });
      }
      return visited;
    }

    var shiftSelectAnchor = null;

    cy.on('tap', 'node', function(e) {
      if (!selectMode) return;
      var node = e.target;
      // Only consider visible, top-level nodes.
      if (!(node.hasClass('op') || node.hasClass('graph-io') || node.hasClass('compound'))) return;
      if (node.id().indexOf('__sg_') !== -1) return;

      if (e.originalEvent && e.originalEvent.shiftKey && shiftSelectAnchor) {
        var aId = shiftSelectAnchor.id(), bId = node.id();
        // Collect nodes on A→B paths and B→A paths (handles both directions).
        var fwdA = bfsReach(aId, adj_succ), bwdB = bfsReach(bId, adj_pred);
        var fwdB = bfsReach(bId, adj_succ), bwdA = bfsReach(aId, adj_pred);
        var pathIds = {};
        Object.keys(fwdA).forEach(function(id) { if (bwdB[id]) pathIds[id] = true; });
        Object.keys(fwdB).forEach(function(id) { if (bwdA[id]) pathIds[id] = true; });
        cy.nodes().forEach(function(n) {
          if (!pathIds[n.id()]) return;
          if (!(n.hasClass('op') || n.hasClass('graph-io') || n.hasClass('compound'))) return;
          if (n.id().indexOf('__sg_') !== -1) return;
          n.select();
        });
        // Cytoscape's own shift-click will toggle the tapped node;
        // ensure it ends up selected regardless of its prior state.
        setTimeout(function() { node.select(); }, 0);
      } else if (!e.originalEvent || !e.originalEvent.shiftKey) {
        shiftSelectAnchor = node;
      }
    });

    // Reset anchor whenever the selection is fully cleared.
    cy.on('unselect', 'node', function() {
      if (selectMode && cy.nodes(':selected').length === 0) shiftSelectAnchor = null;
    });

    // Close button
    document.getElementById('pg-extract-close').addEventListener('click', function() {
      document.getElementById('pg-extract-panel').style.display = 'none';
      cy.nodes(':selected').unselect();
      cy.edges().removeClass('boundary-input boundary-output selected-internal');
      shiftSelectAnchor = null;
    });

    // Copy button (clipboard overlay icon)
    var copyBtn = document.getElementById('pg-extract-copy');
    copyBtn.addEventListener('click', function() {
      var cmd = document.getElementById('pg-extract-cmd').textContent;
      navigator.clipboard.writeText(cmd).then(function() {
        copyBtn.innerHTML = '&#x2713;';
        copyBtn.classList.add('copied');
        setTimeout(function() {
          copyBtn.innerHTML = '&#x2398;';
          copyBtn.classList.remove('copied');
        }, 1500);
      });
    });

    // Inline editable output path — regenerate command on change
    var pathInput = document.getElementById('pg-extract-path');
    function autoResizePath() {
      pathInput.style.height = 'auto';
      pathInput.style.height = pathInput.scrollHeight + 'px';
    }
    pathInput.addEventListener('input', function() {
      currentOutputPath = pathInput.value;
      autoResizePath();
      var selected = cy.nodes(':selected').filter(function(n) {
        return n.hasClass('op') || n.hasClass('graph-io') || n.hasClass('compound');
      });
      var boundary = computeBoundaryTensors(selected);
      document.getElementById('pg-extract-cmd').textContent = generateExtractCommand(boundary, currentOutputPath);
    });

    // Extract & Save button (server-side execution)
    document.getElementById('pg-extract-run').addEventListener('click', function() {
      var selected = cy.nodes(':selected').filter(function(n) {
        return n.hasClass('op') || n.hasClass('graph-io') || n.hasClass('compound');
      });
      var boundary = computeBoundaryTensors(selected);

      var st = document.getElementById('pg-extract-status');
      st.textContent = 'Extracting...';
      st.style.color = '#64748b';

      fetch('/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          inputs: boundary.inputs,
          outputs: boundary.outputs,
          output_path: currentOutputPath
        })
      })
      .then(function(r) { return r.json(); })
      .then(function(d) {
        if (d.status === 'ok') {
          st.style.color = '#22c55e';
          st.textContent = 'Saved to ' + d.path;
        } else {
          st.style.color = '#ef4444';
          st.textContent = 'Error: ' + d.message;
        }
      })
      .catch(function(e) {
        st.style.color = '#ef4444';
        st.textContent = 'Error: ' + e;
      });
    });

    // Escape key closes the extract panel and clears the selection.
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') {
        document.getElementById('pg-extract-close').click();
      }
    });
  }

  }, 0); }); // end setTimeout / requestAnimationFrame
})();
"""
)


# ── CDN script tags ────────────────────────────────────────────────────────────

_CDN_SCRIPTS = """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
<script src="https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
<script src="https://unpkg.com/cytoscape-expand-collapse@4.1.0/cytoscape-expand-collapse.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-node-html-label@1.2.2/dist/cytoscape-node-html-label.min.js"></script>
"""
