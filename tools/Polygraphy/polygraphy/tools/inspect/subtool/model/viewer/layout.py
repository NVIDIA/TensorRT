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
"""Node-size estimation and hierarchical graph layout for the model viewer."""
from polygraphy.tools.inspect.subtool.model.graph_data import (
    GraphData,
    NodeInfo,
    TensorInfo,
)
from polygraphy.tools.inspect.subtool.model.viewer.elements import (
    _has_info,
    _scalar_attrs,
    _truncate,
)


def _estimate_node_size(node: NodeInfo) -> tuple:
    """
    Return an approximate ``(width_px, height_px)`` for a ``NodeInfo``'s
    rendered HTML card (``.pg-node-card``).

    Calibrated against the CSS in ``_templates.py``:
      - header:  9 px Inter,  padding 2 px × 7 px → ~18 px tall
      - body:    7.5 px SF Mono, line-height 1.4 → ~11 px / line,
                 body padding 2 px top + 3 px bottom
      - horizontal padding: 7 px each side (14 px total)
    """
    HDR_H = 18
    LINE_H = 11
    BODY_V_PAD = 5
    H_PAD = 14
    HDR_CW = 5.5  # Inter 9 px character width
    BODY_CW = 4.5  # SF Mono 7.5 px character width
    MIN_W = 60

    w = max(MIN_W, int(len(node.op_type) * HDR_CW) + H_PAD)

    body_lines = [f"{k}: {_truncate(str(v), 14)}" for k, v in _scalar_attrs(node)]
    if _has_info(node.origin):
        body_lines.append(f"origin: {_truncate(str(node.origin), 14)}")
    if _has_info(node.tactic):
        body_lines.append(f"tactic: {_truncate(str(node.tactic), 18)}")

    if body_lines:
        max_chars = max(len(l) for l in body_lines)
        w = max(w, int(max_chars * BODY_CW) + H_PAD)

    h = HDR_H + (BODY_V_PAD + len(body_lines) * LINE_H if body_lines else 0)
    return w, h


def _estimate_io_node_size(ti: TensorInfo) -> tuple:
    """
    Return an approximate ``(width_px, height_px)`` for a graph-I/O canvas
    node (``.graph-io``).  Calibrated against 9 px SF Mono + 6 px padding.
    """
    CHAR_W = 5.5
    H_PAD = 12
    display_name = _truncate(ti.name, 22)
    w = max(50, int(len(display_name) * CHAR_W) + H_PAD)
    return w, 21


def _compute_hierarchical_positions(graph_data: GraphData, prefix: str = "") -> tuple:
    """
    Compute x,y positions for all nodes in *graph_data* using a simple
    hierarchical layout:
      1. Iterative BFS longest-path rank assignment (y-axis level).
      2. Barycenter x-ordering within each rank.
      3. Per-node size estimates for spacing (so attribute-heavy nodes get
         more room and attribute-free nodes aren't over-spaced).

    Returns a 2-tuple ``(positions, ranks)`` where ``positions`` maps
    prefixed node IDs to ``(x, y)`` pixel coordinates suitable for
    Cytoscape's ``preset`` layout, and ``ranks`` maps prefixed node IDs
    to their integer rank (depth).  Both dicts are empty when compound
    (subgraph) nodes are present so the caller can fall back to the
    client-side Dagre layout.
    """
    from collections import defaultdict, deque

    # Compound nodes require nested layout — skip and let Dagre handle it.
    if any(
        any(isinstance(v, GraphData) for v in n.attrs.values())
        for n in graph_data.nodes
    ):
        return {}, {}

    all_ids = (
        [f"{prefix}input_{i}" for i in range(len(graph_data.graph_inputs))]
        + [f"{prefix}{n.node_id}" for n in graph_data.nodes]
        + [f"{prefix}output_{i}" for i in range(len(graph_data.graph_outputs))]
    )
    id_set = set(all_ids)

    # Per-node size estimates.
    node_sizes: dict = {}
    for i, ti in enumerate(graph_data.graph_inputs):
        node_sizes[f"{prefix}input_{i}"] = _estimate_io_node_size(ti)
    for n in graph_data.nodes:
        node_sizes[f"{prefix}{n.node_id}"] = _estimate_node_size(n)
    for i, ti in enumerate(graph_data.graph_outputs):
        node_sizes[f"{prefix}output_{i}"] = _estimate_io_node_size(ti)

    predecessors: dict = defaultdict(set)
    successors: dict = defaultdict(set)
    for src, dst, _ in graph_data.edges:
        s, d = f"{prefix}{src}", f"{prefix}{dst}"
        if s in id_set and d in id_set:
            successors[s].add(d)
            predecessors[d].add(s)

    # Iterative longest-path rank assignment (handles cycles gracefully).
    in_deg = {nid: len(predecessors[nid]) for nid in all_ids}
    queue = deque(nid for nid in all_ids if in_deg[nid] == 0)
    ranks: dict = dict.fromkeys(all_ids, 0)
    while queue:
        nid = queue.popleft()
        for succ in successors[nid]:
            ranks[succ] = max(ranks[succ], ranks[nid] + 1)
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)

    # ── Adjust ranks for constant subgraphs ──────────────────────────────────
    # Nodes *not* reachable from graph inputs (Constant layers and their
    # consumers within an isolated constant subgraph) get low forward-ranks
    # because their in-degree is small.  This makes them appear at the top of
    # the graph with long edges stretching down to where they're actually used.
    #
    # Fix: compute the "latest possible rank" for each such node via a backward
    # pass from its successors.  The whole constant subgraph then slides down to
    # sit just above the first node in the main graph that consumes its output.
    reachable: set = set()
    _rq = deque(f"{prefix}input_{i}" for i in range(len(graph_data.graph_inputs)))
    while _rq:
        _nid = _rq.popleft()
        if _nid in reachable:
            continue
        reachable.add(_nid)
        for _s in successors[_nid]:
            _rq.append(_s)

    constant_nodes = {nid for nid in all_ids if nid not in reachable}
    if constant_nodes:
        # Process in descending forward-rank order so consumers within the
        # constant subgraph are updated before their predecessors.
        latest: dict = dict(ranks)
        for nid in sorted(constant_nodes, key=lambda n: -ranks[n]):
            succs = [s for s in successors[nid] if s in latest]
            if succs:
                latest[nid] = max(0, min(latest[s] for s in succs) - 1)
        for nid in constant_nodes:
            ranks[nid] = latest[nid]

    # Group by rank.
    by_rank: dict = defaultdict(list)
    for nid in all_ids:
        by_rank[ranks[nid]].append(nid)

    # ── Dummy-node insertion for long (skip) edges ────────────────────────────
    # For any edge that spans > 1 rank, insert a chain of invisible dummy nodes
    # at intermediate ranks.  This forces real nodes that share a rank with a
    # dummy to spread apart horizontally (barycenter ordering places them on
    # opposite sides), producing a V-shape instead of collinear overlap.
    # Example: Transpose(rank 0) → MatMul(rank 2) with Mul also at rank 1.
    # A dummy at rank 1 makes rank 1 = [Mul, dummy], so Mul shifts left and
    # MatMul centers between them — matching what Dagre would produce.
    DUMMY_W = 120  # virtual width; invisible but reserves x-space
    dummy_set: set = set()
    dummy_counter = 0
    for src, dst, _ in graph_data.edges:
        s, d = f"{prefix}{src}", f"{prefix}{dst}"
        if s not in id_set or d not in id_set:
            continue
        r_s, r_d = ranks.get(s, -1), ranks.get(d, -1)
        if r_d - r_s <= 1:
            continue  # adjacent-rank edge — no dummy needed
        # Replace direct long-edge connection in the layout graph with a chain.
        successors[s].discard(d)
        predecessors[d].discard(s)
        prev_id = s
        for r in range(r_s + 1, r_d):
            dummy_id = f"__dummy_{dummy_counter}_{r}__"
            dummy_counter += 1
            dummy_set.add(dummy_id)
            by_rank[r].append(dummy_id)
            ranks[dummy_id] = r
            node_sizes[dummy_id] = (DUMMY_W, 0)  # 0 height: no vertical space
            successors[prev_id].add(dummy_id)
            predecessors[dummy_id].add(prev_id)
            prev_id = dummy_id
        successors[prev_id].add(d)
        predecessors[d].add(prev_id)

    # ── Barycenter ordering (top-down pass) ───────────────────────────────────
    for rank in sorted(by_rank.keys()):
        if rank == 0:
            continue
        prev_order = {nid: i for i, nid in enumerate(by_rank[rank - 1])}

        def _bary(nid, po=prev_order):
            preds = [p for p in predecessors[nid] if p in po]
            return sum(po[p] for p in preds) / len(preds) if preds else 0.0

        by_rank[rank].sort(key=_bary)

    # ── Y-position accumulation ───────────────────────────────────────────────
    # rank_top[rank] = top edge of the rank's bounding box.
    # Node centers sit at rank_top + max_h/2 so nodes within a rank are
    # center-aligned and the gap between BOTTOM of one rank and TOP of the next
    # is exactly V_GAP (room for edge labels).  Dummy nodes have h=0 so they
    # don't inflate rank heights.
    V_GAP = 35  # vertical gap between rank edges (accommodates edge labels)
    H_GAP = 20  # horizontal gap between nodes in a rank

    rank_max_h: dict = {
        rank: max((node_sizes[nid][1] for nid in nids), default=40)
        for rank, nids in by_rank.items()
    }

    rank_top: dict = {}
    y = 0.0
    for rank in sorted(by_rank.keys()):
        rank_top[rank] = y
        y += rank_max_h[rank] + V_GAP

    # ── X-position assignment ─────────────────────────────────────────────────
    # Dummies are included in width computation (they reserve horizontal space)
    # but excluded from the returned positions dict (they are not rendered).
    positions = {}
    for rank, nids in by_rank.items():
        y_center = rank_top[rank] + rank_max_h[rank] / 2.0
        widths = [node_sizes[nid][0] for nid in nids]
        total_w = sum(widths) + H_GAP * (len(nids) - 1)
        x = -total_w / 2.0
        for nid, w in zip(nids, widths):
            if nid not in dummy_set:
                positions[nid] = (x + w / 2.0, y_center)
            x += w + H_GAP

    return positions, ranks
