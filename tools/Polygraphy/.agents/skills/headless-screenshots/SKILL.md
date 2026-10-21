---
name: headless-screenshots
description: Use this skill when asked to take browser screenshots of web pages or web-based tools in a headless/automated way — especially when the page uses HTML canvas (e.g. Cytoscape.js, WebGL, Chart.js) and the screenshots need to show canvas-rendered content like graph nodes, edge labels, or drawn shapes. Also use when asked to automate multi-step browser interactions (clicks, drags, form fills) before screenshotting.
---

# Headless Browser Screenshots with Playwright

## Stack

- **Playwright** (Python) — `pip install playwright && python3 -m playwright install chromium`
- **Chromium headless shell** — installed by the command above

## Critical: Canvas rendering requires software rendering

Headless Chromium GPU-composites canvas off-screen by default. `canvas.getContext('2d').getImageData()` returns all-zero pixels, and canvas-rendered content (text labels, shapes drawn via `fillText`/`strokeRect`, etc.) does **not** appear in `page.screenshot()`.

**Fix: always launch with `--use-gl=swiftshader`**

```python
browser = pw.chromium.launch(
    headless=True,
    args=["--disable-gpu", "--use-gl=swiftshader"],
)
```

This forces software rendering so canvas content is fully composited into screenshots.

### How to tell you need this

- DOM-rendered content (divs, HTML overlays) shows up fine in screenshots
- Canvas-drawn content (graph edges, IO node labels, chart axes) is missing
- `canvas.getContext('2d').getImageData(x, y, 1, 1).data` returns `[0, 0, 0, 0]` everywhere

## Serving pages

Serve HTML via a local HTTP server — snap-confined browsers and sandboxed environments block `file://` paths:

```python
srv = subprocess.Popen(
    ["python3", "-m", "http.server", "8765"],
    cwd="/tmp/my_assets",
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)
# wait for it to be ready
import socket, time
def wait_for_port(port, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try: socket.create_connection(("127.0.0.1", port), 0.5).close(); return True
        except OSError: time.sleep(0.2)
```

## Wait times for JS-heavy pages

After `wait_until="networkidle"`, add extra sleep for pages that run layout algorithms (e.g. Cytoscape dagre, D3 force-directed):

```python
page.goto(url, wait_until="networkidle")
time.sleep(3.5)   # let JS layout finish rendering
```

## Accessing Cytoscape.js instances

`window.cy` is shadowed by the `#cy` DOM element (browser auto-creates globals from element IDs). Use the internal registry instead:

```python
GET_CY = "document.getElementById('cy')._cyreg.cy"

# Get rendered node positions
positions = page.evaluate(f"""
    () => {{
        var cy = {GET_CY};
        var result = [];
        cy.nodes('.op').forEach(function(n) {{
            var pos = n.renderedPosition();
            result.push({{x: pos.x, y: pos.y}});
        }});
        return result;
    }}
""")
```

`renderedPosition()` returns pixel coordinates relative to the canvas container div (not the page).

## Box-select / drag interactions

```python
canvas_box = page.locator("#cy").bounding_box()
cx, cy_y = canvas_box["x"], canvas_box["y"]

page.mouse.move(cx + x1, cy_y + y1)
page.mouse.down()
page.mouse.move(cx + x2, cy_y + y2, steps=12)  # steps for smooth drag
page.mouse.up()
time.sleep(1.5)  # let selection + panel update
```

## Avoiding identifying paths in screenshots

If a page displays the model/file path in its UI (e.g. in a command preview), copy the file to a neutral temp location before serving:

```python
import shutil, pathlib
shutil.copy(src_model, "/tmp/polygraphy_docs/model.onnx")
# serve from /tmp/polygraphy_docs so displayed path is non-identifying
```

## Full boilerplate

```python
import subprocess, time, socket, glob, os
from pathlib import Path
from playwright.sync_api import sync_playwright

CHROMIUM_ARGS = ["--disable-gpu", "--use-gl=swiftshader"]
VIEWPORT = {"width": 1600, "height": 900}

def wait_for_server(port, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try: socket.create_connection(("127.0.0.1", port), 0.5).close(); return True
        except OSError: time.sleep(0.2)
    return False

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True, args=CHROMIUM_ARGS)
    page = browser.new_page(viewport=VIEWPORT)
    page.goto("http://127.0.0.1:PORT/", wait_until="networkidle")
    time.sleep(3.5)

    page.screenshot(path="output.png")
    browser.close()
```
