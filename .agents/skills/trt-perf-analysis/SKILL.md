---
name: trt-perf-analysis
description: Validate and analyze TensorRT performance data from paired layer-info JSON and profile/latency JSON files. Use when asked to inspect TensorRT, TRT, torch-tensorrt, or ONNX-TensorRT perf reports, verify that layer/profile JSON files are valid and from the same model, infer basic model information, find likely fusion or latency optimization opportunities, and produce a concise Markdown performance report or structured JSON data.
---

# TRT Perf Analysis

## Workflow

Use `scripts/run.sh` on Unix-like systems or `scripts\run.cmd` on Windows for Python scripts. These wrappers do best-effort Python 3.8+ discovery; set `SKILL_PYTHON` to a Python executable to override discovery. Replace placeholders with platform-native paths.

1. Choose the input scope.
   Use one folder that directly contains `layers_*.json` and/or `profile_*.json`. Do not call the packager on a parent folder that only contains component subfolders. For model suites with separate encoder, transformer, decoder, VAE, or similar components, handle each component folder separately.

2. Infer model identity and components.
   Prefer an explicit model name from the user prompt and pass it with `--model-name`. Otherwise rely on the analyzer/packager to inspect likely config files, the input directory name, and layer/profile filenames. Keep the name empty when confidence is low. Serialized JSON records the inferred model identity at the top level and in each successful backend's `model` object.

3. Run deterministic analysis.
   Run `scripts/analyze_trt_perf.py` first when you need an integrity gate before interpretation. The script uses only Python built-in modules, extracts structured analysis data, and serializes it as JSON. This JSON is the authoritative analyzer output for validation, packaging, and AI diagnosis.

4. Check validation before interpreting.
   The analyzer exits `0` when it can emit structured validation data, even if one or more backends fail validation. Read the generated JSON `validation` object to decide which backend reports are usable. If the analyzer exits nonzero, stop because it could not emit structured data. If the schema validator exits nonzero, stop because the generated JSON contract is invalid. Do not continue into performance interpretation for a backend unless its validation status is `passed`, its analysis mode is `layer_profile`, the layer graph is a DAG, and the layer/profile names match. For `layer_only` backends, graph and layer inspection are available but latency/performance interpretation is not.

5. Understand the model, then write AI diagnosis.
   Use the generated JSON as grounding material. Build a clear picture of the model or component scope, backend differences, graph structure, timing distribution, hot layers, fusion clues, and caveats before writing conclusions. The browser `Summary` panel is generated from `analyze-data.json`; do not copy that deterministic summary into `analyze.md`.

   Keep the final AI analysis in two sections:

   1. `## Summary` - short, high-confidence facts and the most important optimization leads.
   2. `## Details` - interpretation, cross-backend comparisons, hypotheses, ranked experiments, and human judgment that is not a mechanical restatement of the analyzer output.

   Before sending the final response, write the final AI analysis into `analyze.md` for each report folder you created or updated. Match each file to that report folder's scope. For a single model with multiple backend JSON pairs, one `analyze.md` may compare those backends. For multiple model components packaged as multiple report folders, write distinct component-focused `analyze.md` files; do not copy a suite-level analysis into every component report.

   `analyze.md` should not include validation checklists, complete timing tables, top-layer tables, path dumps, or other content already rendered by the report app from `analyze-data.json`. Briefly cite specific metrics or layer names only when they support a new conclusion or next experiment.

6. Package the report.
   Run `scripts/package_report.py` through the shared runner. Each invocation creates exactly one final report folder from one input folder that directly contains `layers_*.json` or `profile_*.json` files. The packager writes `analyze-data.json`, validates it against the schema, and copies the browser report template while preserving the generated JSON. Determine the output parent from the user's active workspace, not from a subdirectory of this skill. If the only known directory is the skill directory, omit `--output-parent`; the packaging script will fall back to the system temp directory.

   When producing multiple reports for one user request, prefer explicit `--report-dir` values so report folder names include the model/component identity instead of relying on auto-numbered collision suffixes. If a suite-level conclusion is useful, write it as a separate overview Markdown file in the output parent, not as every component report's `analyze.md`.

7. Preview only the final report.
   Serve the completed report folder, not the skill directory, when the user wants a browser preview. The Python `visualize_layer_info.py` script remains available for standalone HTML layer-info debugging, but it is not part of the report packaging flow or report app dependency chain.

### Common Commands

Use `<runner>` as `<skill-dir>/scripts/run.sh` on Unix-like systems or `<skill-dir>\scripts\run.cmd` on Windows.

```bash
# Validate or inspect inputs. Folder mode discovers matching backends; repeat --data for explicit backends.
<runner> <skill-dir>/scripts/analyze_trt_perf.py <model-folder> --output <analysis.json>
<runner> <skill-dir>/scripts/analyze_trt_perf.py --data <layers-a.json> <profile-a.json> --data <layers-b.json> --output <analysis.json>

# Create a report. Use --report-dir for component-specific reports and --analyze-md after writing final AI analysis.
<runner> <skill-dir>/scripts/package_report.py <model-folder> --output-parent <workspace-folder> [--model-name <model>] [--analyze-md <final-analysis.md>]
<runner> <skill-dir>/scripts/package_report.py --analyze-data <report-folder>/analyze-data.json [--analyze-md <final-analysis.md>]

# Optional preview.
<runner> -m http.server 8765 --bind 127.0.0.1 --directory <report-folder>
```

Then open `http://127.0.0.1:8765/`.

## Inputs

Expect one layer-info JSON and one profile/latency JSON per engine/report. A folder may contain multiple pairs such as:

- `layers_torch-trt-aot.json` with `profile_torch-trt-aot.json`
- `layers_onnx-tensorrt.json` with `profile_onnx-tensorrt.json`

Layer records are expected to include `Name`, `LayerType`, `Inputs`, and `Outputs`. Profile records are expected to include `name`, `timeMs`, `averageMs`, `medianMs`, and `percentage`; a first record like `{ "count": 20 }` is allowed.

Infer a per-layer `Type` for report output. Default to the raw `LayerType`, then apply narrow special-case overrides when the layer name, tactic, or metadata makes the specialization clear. Known special cases:

- `kgen_mha`: a raw `LayerType` of `kgen` that is actually an MHA/FMHA-style attention layer.
- `misc`: raw `LayerType` values `reshape`, `shape_call`, `signal`, and `wait` are grouped into this category.

## Model Categories

Use this starter category list and revise as evidence warrants:

- Transformer encoder / sentence embedding
- Transformer decoder / LLM
- Vision transformer
- CNN / convolutional vision model
- Diffusion / U-Net style model
- Recommender / ranking model
- Classical ML or feature pipeline
- Unknown DL model

Report category only when the evidence is strong. Good evidence includes Hugging Face config fields, names like `encoder.layer`, attention layers, input names such as `input_ids`, or repeated convolution/GEMM patterns. Say "unknown" instead of over-claiming.

## Analysis Heuristics

Prioritize issues by likely runtime impact and confidence:

- Hot layers: individual layers with high `percentage` or `averageMs`.
- Layer-type concentration: high aggregate time in `kgen`, shape, cast, gather, reshape, or pointwise/reduction layers may indicate fusion or dynamic-shape overhead.
- Fusion clues: for transformer encoders, check whether Q/K/V matmuls are fused per block, whether attention is an obvious fused SDPA/FMHA layer, and whether layer-norm/residual/GELU patterns are fused into larger kernels.
- Engine partitioning: one engine is usually ideal. Use adjacent `.trt`/`.engine` files, `_subgraph`, and `StreamId` as hints; state when this is only inferred.
- GEMM tactics: GEMM layers without Tensor Core or xMMA-like tactic names may deserve inspection.
- Many tiny kernels: if no single layer dominates but many small kernels add up, rank launch/fusion overhead above isolated micro-hotspots.

Keep facts and hypotheses separate. For each optimization opportunity, include evidence from the JSON and a next check or experiment.
