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
"""
Cytoscape.js-based interactive model graph viewer with compound node support.

``ModelViewer`` builds a Cytoscape.js graph from a ``GraphData``, generates a
self-contained HTML page, and serves it via a local HTTP server.  ONNX subgraph
nodes (Loop, If, Scan) are rendered as compound nodes that can be expanded and
collapsed in-place.

Libraries loaded from CDN: cytoscape, dagre, cytoscape-dagre, cytoscape-expand-collapse.
"""
import http.server
import json
import os
import socketserver
import subprocess
import threading
import urllib.parse
import webbrowser

from polygraphy.logger import G_LOGGER
from polygraphy.tools.inspect.subtool.model.graph_data import GraphData
from polygraphy.tools.inspect.subtool.model.viewer.elements import (
    _build_cytoscape_elements,
)
from polygraphy.tools.inspect.subtool.model.viewer.layout import (
    _compute_hierarchical_positions,
)
from polygraphy.tools.inspect.subtool.model.viewer.node_info import _build_node_json
from polygraphy.tools.inspect.subtool.model.viewer.page import _build_html


class ModelViewer:
    """
    Interactive HTML model-graph viewer backed by Cytoscape.js.

    Subgraph nodes (Loop, If, Scan) are rendered as compound nodes that can be
    expanded in-place.  Call ``run()`` to serve the page and open it in the
    default browser; the server shuts down when the tab is closed.
    """

    def __init__(
        self,
        graph_data: GraphData,
        port: int = 8000,
        save_path: str = None,
        model_path: str = None,
    ):
        self._graph_data = graph_data
        self._port = port
        self._save_path = save_path
        self._model_path = model_path

    def run(self) -> None:
        positions, _ = _compute_hierarchical_positions(self._graph_data)
        elements = _build_cytoscape_elements(self._graph_data, positions=positions)

        if self._save_path:
            html = _build_html(
                self._graph_data, elements, model_path=self._model_path, is_served=False
            )
            with open(self._save_path, "w", encoding="utf-8") as fh:
                fh.write(html)
            G_LOGGER.info(f"Viewer HTML saved to {self._save_path}")
            return

        # Build the full node-detail dict once; served lazily via /node/<id>.
        full_node_json: dict = _build_node_json(self._graph_data)

        html = _build_html(
            self._graph_data, elements, model_path=self._model_path, is_served=True
        )
        html_bytes = html.encode("utf-8")

        model_path = self._model_path
        model_type = self._graph_data.model_type

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html_bytes)))
                    self.end_headers()
                    self.wfile.write(html_bytes)
                elif self.path == "/heartbeat":
                    self.send_response(204)
                    self.end_headers()
                elif self.path.startswith("/node/"):
                    node_id = urllib.parse.unquote(self.path[6:])
                    d = full_node_json.get(node_id)
                    if d is None:
                        self.send_response(404)
                        self.end_headers()
                    else:
                        resp_bytes = json.dumps(d).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(resp_bytes)))
                        self.end_headers()
                        self.wfile.write(resp_bytes)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == "/extract" and model_path and model_type == "onnx":
                    content_len = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(content_len))
                    output_path = body.get("output_path", "extracted.onnx")

                    cmd = [
                        "polygraphy",
                        "surgeon",
                        "extract",
                        model_path,
                        "-o",
                        output_path,
                    ]
                    for inp in body.get("inputs", []):
                        shape = inp.get("shape")
                        dtype = inp.get("dtype") or "auto"
                        if shape and len(shape):
                            shape_str = "[" + ",".join(str(d) for d in shape) + "]"
                        else:
                            shape_str = "auto"
                        cmd.extend(["--inputs", f"{inp['name']}:{shape_str}:{dtype}"])
                    for out in body.get("outputs", []):
                        dtype = out.get("dtype") or "auto"
                        cmd.extend(["--outputs", f"{out['name']}:{dtype}"])

                    try:
                        result = subprocess.run(
                            cmd, capture_output=True, text=True, timeout=120
                        )
                        resp = (
                            {"status": "ok", "path": os.path.abspath(output_path)}
                            if result.returncode == 0
                            else {
                                "status": "error",
                                "message": result.stderr.strip()
                                or result.stdout.strip(),
                            }
                        )
                    except Exception as e:
                        resp = {"status": "error", "message": str(e)}

                    resp_bytes = json.dumps(resp).encode("utf-8")
                    self.send_response(200 if resp.get("status") == "ok" else 500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp_bytes)))
                    self.end_headers()
                    self.wfile.write(resp_bytes)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *_):
                pass

        class _Server(socketserver.TCPServer):
            allow_reuse_address = True

        with _Server(("127.0.0.1", self._port), _Handler) as httpd:
            port = httpd.server_address[1]
            threading.Thread(target=httpd.serve_forever, daemon=True).start()

            url = f"http://127.0.0.1:{port}/"
            G_LOGGER.info(f"Viewer available at {url}  (press Ctrl+C to exit)")
            webbrowser.open(url)

            try:
                threading.Event().wait()  # block until Ctrl+C
            except KeyboardInterrupt:
                pass

            httpd.shutdown()
