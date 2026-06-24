# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write a standalone HTML visualization target for TensorRT layer-info data."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def short(value: Any, limit: int = 180) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}..."


def table_rows(items: list[tuple[str, Any]]) -> str:
    rows = []
    for key, value in items:
        rows.append(
            "<tr>"
            f"<th>{escape(key)}</th>"
            f"<td><code>{escape(value)}</code></td>"
            "</tr>"
        )
    return "\n".join(rows)


def write(processed: Any, output_path: Path) -> None:
    layer_rows = []
    for node in processed.nodes:
        attrs = node.attrs
        layer_rows.append(
            "<tr>"
            f"<td>{escape(attrs.get('trt_layer_index', ''))}</td>"
            f"<td><code>{escape(short(node.name, 140))}</code></td>"
            f"<td>{escape(attrs.get('trt_layer_type', node.op_type))}</td>"
            f"<td>{escape(node.op_type)}</td>"
            f"<td>{escape(len(node.inputs))}</td>"
            f"<td>{escape(len(node.outputs))}</td>"
            f"<td><code>{escape(short(attrs.get('trt_tactic_name', ''), 160))}</code></td>"
            "</tr>"
        )

    summary_rows = table_rows(
        [
            ("source", processed.source_path),
            ("layers", len(processed.layers)),
            ("nodes", len(processed.nodes)),
            ("graph inputs", len(processed.graph_inputs)),
            ("graph outputs", len(processed.graph_outputs)),
            ("initializers", len(processed.initializers)),
            ("layer types", compact_json(dict(processed.layer_type_counts))),
            ("node op types", compact_json(dict(processed.node_op_type_counts))),
        ]
    )

    graph_inputs = "\n".join(f"<li><code>{escape(name)}</code></li>" for name in processed.graph_inputs)
    graph_outputs = "\n".join(f"<li><code>{escape(name)}</code></li>" for name in processed.graph_outputs)

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TensorRT Layer Info: {escape(processed.source_path.name)}</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f8f9fb;
      --fg: #17202a;
      --muted: #586476;
      --line: #d7dce5;
      --panel: #ffffff;
      --accent: #0f766e;
      --code: #0b4a64;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #111417;
        --fg: #edf1f5;
        --muted: #aab4c0;
        --line: #303942;
        --panel: #181d22;
        --accent: #5eead4;
        --code: #93c5fd;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--fg);
      font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1400px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 28px 0 10px;
      font-size: 17px;
      letter-spacing: 0;
    }}
    p {{ margin: 0 0 16px; color: var(--muted); }}
    code {{
      color: var(--code);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      word-break: break-word;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      white-space: nowrap;
    }}
    thead th {{
      position: sticky;
      top: 0;
      background: var(--panel);
      z-index: 1;
    }}
    tbody tr:last-child td, tbody tr:last-child th {{ border-bottom: 0; }}
    .summary th {{ width: 160px; }}
    .split {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 12px 14px;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
    @media (max-width: 780px) {{
      main {{ width: min(100vw - 20px, 1400px); padding-top: 18px; }}
      .split {{ grid-template-columns: 1fr; }}
      th, td {{ padding: 7px 8px; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>TensorRT Layer Info</h1>
  <p>{escape(processed.source_path)}</p>

  <h2>Summary</h2>
  <table class="summary"><tbody>
{summary_rows}
  </tbody></table>

  <h2>Graph Endpoints</h2>
  <div class="split">
    <section class="panel">
      <h2>Inputs</h2>
      <ul>{graph_inputs}</ul>
    </section>
    <section class="panel">
      <h2>Outputs</h2>
      <ul>{graph_outputs}</ul>
    </section>
  </div>

  <h2>Layers</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Name</th>
        <th>Layer Type</th>
        <th>Op Type</th>
        <th>Inputs</th>
        <th>Outputs</th>
        <th>Tactic</th>
      </tr>
    </thead>
    <tbody>
{chr(10).join(layer_rows)}
    </tbody>
  </table>
</main>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
