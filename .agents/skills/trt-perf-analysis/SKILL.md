---
name: trt-perf-analysis
description: Generate an interactive HTML performance report from existing TensorRT (TRT) layer-info and profile/latency JSON pairs, commonly named `layers_*.json` and `profile_*.json`. Use when the user has such dumps and wants to analyze or diagnose inference performance, including hot layers, per-layer-type latency breakdowns, fusion and other optimization opportunities, or comparison across engines, builds, or configurations.
---

# TRT Perf Analysis

## Workflow

Use `<runner>` below as `scripts/run.sh` on Unix-like systems or `scripts\run.cmd` on Windows. Set `SKILL_PYTHON` to override Python 3.8+ discovery.

Treat each step as a gate. On failure, stop, report the error, clean up temporary outputs, and do not create or preview a report.

1. Resolve inputs in this order:

   1. Use JSON pairs explicitly provided by the user; preserve their pairing. Filenames may be arbitrary.
   2. For a provided folder, find complete pairs there. Match conventional `layers_<label>.json` and `profile_<label>.json` names first; use JSON structure and layer names for nonstandard names. Handle component subfolders separately.
   3. With no explicit path, infer only from clear context: attachments, discussed paths, or the active workspace. Fail if no unambiguous pair is found.

   Require at least one readable layer-info and profile/latency pair. Accept raw detailed TensorRT inspector output as either a layer array or a TensorRT 11 object containing `Layers`. Do not use `analyze-data.json` as layer-info input. Reject missing, ambiguous, layer-only, or profile-only inputs.

2. Analyze in staging.
   Run `scripts/analyze_trt_perf.py` before interpretation. Use repeated `--data <layer.json> <profile.json>` for explicit or nonstandard pairs; use folder mode only for conventionally named pairs. Pass a user-provided model name with `--model-name`; otherwise use analyzer inference. Write `analyze-data.json` outside the final report folder and use it as the source for all conclusions.

3. Validate.
   The analyzer may exit `0` for invalid inputs. Require its output and run `scripts/validate_analyze_data.py --require-passed`. Continue only when it reports both `SCHEMA PASS` and `SEMANTIC PASS`; otherwise surface the validation messages and stop.

   Treat TensorRT inspector names as authoritative. Match repeated layer/profile names by occurrence, and treat every repeated tensor output name as one logical tensor. For topology, fail only when the resulting graph is not a DAG or a declared graph input is also produced by a layer; a declared graph output may still feed another layer.

4. Write the diagnosis.
   First understand the report scope, graph, timing distribution, backend differences, hot layers, fusion clues, and caveats. Ground `analyze.md` in the generated JSON, and treat the analyzer's model category as a hypothesis no stronger than its evidence. Use exactly:

   1. `## Summary` — high-confidence findings and top optimization leads.
   2. `## Details` — interpretation, comparisons, hypotheses, and ranked experiments.

   Focus on conclusions not already rendered from `analyze-data.json`; omit validation checklists, full tables, and path dumps. Rank recommendations by impact and confidence, separate facts from hypotheses, and support each with evidence and a next experiment. Use one analysis per report scope; compare multiple pairs only when they belong to that scope.

5. Package.
   Create one report per model or component with `scripts/package_report.py`. For explicit or nonstandard pairs, keep validated analyzer JSON in staging and pass it with `--analyze-data` plus the final `--report-dir`.

   Honor explicit output locations: use `--report-dir` for an exact folder and `--output-parent` for an auto-named child. Otherwise use the active workspace; when running inside this skill, let the packager fall back to the system temp directory. Remove incomplete report artifacts if packaging fails.

6. Preview.
   Serve only the completed report folder when requested.

### Common Commands

```bash
<runner> <skill-dir>/scripts/analyze_trt_perf.py --data <layer-info-a.json> <profile-a.json> --data <layer-info-b.json> <profile-b.json> --output <staging-folder>/analyze-data.json
<runner> <skill-dir>/scripts/analyze_trt_perf.py <model-folder> --output <analysis.json>
<runner> <skill-dir>/scripts/validate_analyze_data.py <staging-folder>/analyze-data.json --require-passed
<runner> <skill-dir>/scripts/package_report.py <model-folder> --report-dir <exact-report-folder> [--model-name <model>] [--analyze-md <final-analysis.md>]
<runner> <skill-dir>/scripts/package_report.py --analyze-data <staging-folder>/analyze-data.json --report-dir <exact-report-folder> --analyze-md <final-analysis.md>
<runner> -m http.server 8765 --bind 127.0.0.1 --directory <report-folder>
```

Then open `http://127.0.0.1:8765/`.
