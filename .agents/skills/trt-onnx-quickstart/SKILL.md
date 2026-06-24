---
name: trt-onnx-quickstart
description: >
  Build and verify a TensorRT engine from a Hugging Face model ID or ONNX
  file, with numerical parity checked against ONNX Runtime. Use when the
  user imports a non-LLM model to TensorRT,
  needs a verified engine from ONNX, hits trtexec "unsupported operator",
  must verify the engine matches ONNX numerically, debugs a polygraphy
  parity failure (large max abs diff at FP16), or configures multi-input
  dynamic shapes. Triggers: convert ONNX to TensorRT, Hugging Face to
  TensorRT, trtexec onnx, trtexec unsupported operator, optimum-cli
  export, polygraphy parity check, polygraphy run --trt --onnxrt, parity
  check failed, max abs diff, verify engine matches ONNX, --minShapes,
  dynamic shapes trtexec, multi-input shape profile, FP16 engine, INT64
  warning. Adjacent skills: `trt-torch-quickstart` (PyTorch frontend),
  `trt-cpp-runtime-quickstart` (C++ engine load). LLM token generation
  belongs in TensorRT-LLM, not here.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
  version: "1.0"
  tags:
    - onnx
    - import
    - huggingface
    - quickstart
    - fp16
---

# TensorRT ONNX Quickstart

Take a developer from "I have a Hugging Face model ID" or "an ONNX file" to "a TensorRT engine whose outputs match the source model within tolerance." Follows Path 1 (ONNX → TensorRT) of the [Import Workflows Guide](https://github.com/NVIDIA/TensorRT/blob/main/documents/import_workflows.md), specialized for the most common starting point: a Hugging Face Hub model.

## When to Use

| Scenario | Use this skill? |
|----------|-----------------|
| Has a Hugging Face model ID (`google-bert/bert-base-uncased`) and wants TRT-accelerated inference | Yes |
| Has an `.onnx` file and wants a `.plan` engine with verified parity | Yes |
| Ran `trtexec --onnx=...` and hit a warning/error they don't recognize | Yes |
| Wants LLM token generation (Llama, Mistral, Qwen text generation) | **No** — route to [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |
| Has a PyTorch model and wants to stay in PyTorch | No — use `trt-torch-quickstart` |
| Already has a `.plan` engine and wants to run inference from C++ | No — use `trt-cpp-runtime-quickstart` |
| Is migrating a weakly-typed network to strongly-typed | No — use `trt-strong-typing-migration` |
| Has a full diffusion pipeline (SD, FLUX) | Partial — must be component-split first; this skill imports one component at a time |

## Prerequisites

Verify each before continuing. Failures here surface as confusing errors later, so fix them before running the import.

1. **NVIDIA GPU + driver matching TensorRT 11.x.** TRT 11 requires CUDA 13.x. Confirm:
   ```bash
   nvidia-smi  # driver + CUDA runtime version
   ```
2. **Python ≥ 3.10.** TRT 11 dropped 3.9 and earlier.
3. **TensorRT 11 installed — including the `trtexec` CLI**, this skill's primary build tool. **The pip wheel does not ship `trtexec`.** Pick one:

   - **NGC container (recommended for first-time users):** `nvcr.io/nvidia/tensorrt:<tag>` — bundles `trtexec`, Python bindings, libraries, and most common dependencies. Run with `--gpus all`.
   - **`.tar.gz` or `.deb` from [TensorRT downloads](https://developer.nvidia.com/tensorrt-download):** installs `trtexec` to a system path.

   The pip path (`pip install --extra-index-url https://pypi.nvidia.com tensorrt-cu13`) provides Python bindings but not `trtexec`, so it is **insufficient on its own**. For pip-only, route the developer to the alternative Python-builder flow in `import_workflows.md` Path 1 Option B; this skill assumes `trtexec` is available.

   Verification:
   ```bash
   python3 -c "import tensorrt; print(tensorrt.__version__)"   # expect 11.x.y.z
   trtexec --help | head -5                                     # MUST work; if not, install is incomplete
   ```
4. **Polygraphy and `optimum-onnx` installed.**
   ```bash
   pip install polygraphy onnx onnxruntime onnx_graphsurgeon onnxsim 'optimum-onnx'
   # If the model ID starts with `sentence-transformers/`, also:
   pip install sentence-transformers
   ```
   `onnx_graphsurgeon` is required by `polygraphy surgeon sanitize` (Step 3) — Polygraphy does not bundle it. `onnxruntime` is required by `optimum-cli export` (Step 2, to fix dynamic axes) and Polygraphy's `--onnxrt` parity check (Step 5); stock NGC containers do not always ship it.
   Note: `optimum-onnx` is the active maintained package — the older `optimum` no longer includes ONNX integration. Do **not** use `optimum-nvidia` (last release 2025-01, unmaintained).

## Step 1 — Choose a model and confirm it's supported

Consult the [Supported Models](https://github.com/NVIDIA/TensorRT/blob/main/documents/supported_models.md) matrix. If listed, the dtype column gives a known-good starting precision. If not, it is still expected to work — file an issue if it doesn't.

If the model is a diffusion pipeline, **stop here** — TRT cannot ingest the whole pipeline object. Split it into components (text encoder, UNet/DiT, VAE) and import each separately. See `supported_models.md` for validated splits, and consider whether `trt-export-rewrite` is a better fit.

This skill's worked example uses `google-bert/bert-base-uncased` from the encoder-NLP section.

## Step 2 — Export to ONNX

The `optimum-cli export onnx` command's `--task` flag selects which head of the model to export. The wrong task yields a valid graph but the wrong output — and Step 5 parity validation won't catch it, because reference and engine are both wrong.

Pick the task from this table based on what the developer wants to do with the model:

| Developer's intent | `--task` | Typical model families |
|--------------------|----------|------------------------|
| Get hidden-state embeddings or pooled output | `feature-extraction` | BERT, RoBERTa, sentence-transformers, CLIP |
| Classify images into N classes | `image-classification` | ResNet, ViT, MobileNet, EfficientNet |
| Classify text into N labels | `text-classification` | BERT for sentiment, DistilBERT, etc. |
| Detect / segment objects | `object-detection`, `image-segmentation` | DETR, Mask R-CNN |
| Transcribe speech | `automatic-speech-recognition` | Whisper, Wav2Vec2 |
| Generate captions / VQA | `image-to-text`, `visual-question-answering` | BLIP, LLaVA |
| Translation / summarization | `text2text-generation` | T5, BART |

For an unknown model, run `optimum-cli export onnx --help` or check the [Optimum supported tasks table](https://huggingface.co/docs/optimum/main/en/exporters/onnx/overview). If unsure, surface the choice to the developer rather than guessing.

### Sentence-transformers models need an extra install

If the model ID starts with `sentence-transformers/` (e.g. `all-MiniLM-L6-v2`, `bge-base-en-v1.5`), install `sentence-transformers` *before* exporting:

```bash
pip install sentence-transformers
```

Without it, `optimum-cli` falls back to plain `transformers` and emits raw token-level `last_hidden_state` instead of the pooled sentence embedding the model is designed to produce. The exporter prints `library name was inferred as sentence_transformers, which is not installed. Falling back to transformers.` — treat that warning as a hard error and install before retrying.

```bash
# Worked example: BERT for embedding extraction
optimum-cli export onnx \
  --model google-bert/bert-base-uncased \
  --task feature-extraction \
  bert_onnx/
```

### Expected warnings (not errors)

The exporter often prints a "max diff between reference and ONNX exported model is not within the set tolerance 1e-05" line (with `max diff` around `1e-5` to `1e-4`). This is its own self-check at a strict tolerance — **the export still succeeded.** Step 5's Polygraphy parity check uses `--atol 1e-2 --rtol 1e-2 --check-error-stat mean`, the correct tolerance for FP16 inference. Don't surface this warning as a failure.

### Authentication for gated models

LLaMA, FLUX, and some Mistral and Qwen variants are gated behind HF agreements and require `huggingface-cli login` before download. If `optimum-cli` exits with a 401, run `huggingface-cli login` with a token from `huggingface.co/settings/tokens` and retry.

### If the export fails

The most common cause is a HF model using patterns `torch.export` / `torch.onnx.export` cannot trace — complex-number arithmetic, data-dependent control flow, non-tensor forward arguments, output dataclasses. **Do not patch the upstream library on disk.** The correct workflow is agentic monkey-patching at runtime: route the developer to the `trt-export-rewrite` skill, which drives it with five concrete patterns from the Qwen-Image case study.

## Step 3 — Sanitize the ONNX

ONNX exporters often produce graphs with constant-folding opportunities, spurious dynamic axes, or shape-inference gaps. Sanitizing cleans these up, but it is a **quality pass, not a correctness gate** — TRT parses many un-sanitized exports fine. Both stages below are therefore optional and **allowed to fail**: each falls through to its input, and Step 3 always leaves a `model.clean.onnx` for the rest of the workflow (worst case, a copy of the exported model). Run:

```bash
# Seed model.clean.onnx with the raw export, then update it in-place
cp bert_onnx/model.onnx bert_onnx/model.clean.onnx

# (1) onnxsim — graph simplification / constant folding.
python3 -m onnxsim bert_onnx/model.clean.onnx bert_onnx/model.simplified.onnx \
  && cp bert_onnx/model.simplified.onnx bert_onnx/model.clean.onnx

# (2) polygraphy surgeon — fold constants, prune dangling nodes.
polygraphy surgeon sanitize bert_onnx/model.clean.onnx \
  -o bert_onnx/model.surgeon.onnx --fold-constants \
  && cp bert_onnx/model.surgeon.onnx bert_onnx/model.clean.onnx
```

## Step 3.5 — Inspect the ONNX inputs (parameterize the rest of the workflow)

**Every step from here uses input names and ranks that depend on the model.** Read them out of the sanitized ONNX before building. Do not assume `input_ids` / `attention_mask` — those are BERT-family-specific.

```bash
polygraphy inspect model bert_onnx/model.clean.onnx
```

Example output for `bert-base-uncased` (verified against `optimum-onnx` 0.1.0, opset 18):

```
---- 3 Graph Input(s) ----
{input_ids [dtype=int64, shape=('batch_size', 'sequence_length')],
 attention_mask [dtype=int64, shape=('batch_size', 'sequence_length')],
 token_type_ids [dtype=int64, shape=('batch_size', 'sequence_length')]}

---- 1 Graph Output(s) ----
{last_hidden_state [dtype=float32, shape=('batch_size', 'sequence_length', 768)]}
```

Inputs and outputs vary per model and per `--task`. Inspect, don't assume.

**Build the shape strings from this output.** Every input with a dynamic (named) axis needs a `--minShapes / --optShapes / --maxShapes` entry; fixed axes are reproduced literally. Use representative values — `--opt` is what the engine optimizes for; `--min` and `--max` define the supported range.

If any input reports an unknown-rank axis (printed as `?`), shape inference failed; reroute through `trt-export-rewrite` to fix the export.

## Step 4 — Bake FP16 into the model (TRT 11) or set the build flag (TRT 10)

Start with FP16 — the safe default for encoder and vision models on modern GPUs. (Use BF16 on Blackwell/Hopper if the developer reports persistent FP16 accuracy issues. Sentence-embedding models sometimes prefer FP32 over lossy FP16 — surface the option.)

**TRT 11 removed the weak-typing builder flags** (`--fp16`, `--int8`, `--bf16`, `BuilderFlag::kFP16`, etc. — see [the migration guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x.html) and the sibling `trt-strong-typing-migration` SKILL). In TRT 11 you bake precision **into the model** before building, and `trtexec` honors the dtypes it finds via **ModelOpt AutoCast**.

Detect the version first:

```bash
TRT_MAJOR=$(python3 -c "import tensorrt as t; print(t.__version__.split('.')[0])")
echo "TensorRT major version: $TRT_MAJOR"
```

### TRT 11: AutoCast the model, then build

```bash
pip install nvidia-modelopt    # one-time

# Convert FP32 ONNX to mixed FP16 (ModelOpt picks accuracy-sensitive ops to keep FP32):
python3 -m modelopt.onnx.autocast \
  --onnx_path=<MODEL>.clean.onnx \
  --output_path=<MODEL>.fp16.onnx

# Build the engine. NO --fp16 flag — strong typing is the default and the model already carries FP16 dtypes.
trtexec \
  --onnx=<MODEL>.fp16.onnx \
  --saveEngine=<MODEL>.fp16.plan \
  --memPoolSize=workspace:4096 \
  --minShapes=<INPUT1>:<MINSHAPE>,... \
  --optShapes=<INPUT1>:<OPTSHAPE>,... \
  --maxShapes=<INPUT1>:<MAXSHAPE>,...
```

### TRT 10: pass the legacy --fp16 flag

```bash
trtexec \
  --onnx=<MODEL>.clean.onnx \
  --saveEngine=<MODEL>.fp16.plan \
  --memPoolSize=workspace:4096 \
  --fp16 \
  --minShapes=<INPUT1>:<MINSHAPE>,... \
  --optShapes=<INPUT1>:<OPTSHAPE>,... \
  --maxShapes=<INPUT1>:<MAXSHAPE>,...
```

Worked example for BERT (three inputs, all dynamic on batch and sequence axes), TRT 11 path:

```bash
python3 -m modelopt.onnx.autocast \
  --onnx_path=bert_onnx/model.clean.onnx \
  --output_path=bert_onnx/model.fp16.onnx
trtexec \
  --onnx=bert_onnx/model.fp16.onnx \
  --saveEngine=bert.fp16.plan \
  --memPoolSize=workspace:4096 \
  --minShapes=input_ids:1x32,attention_mask:1x32,token_type_ids:1x32 \
  --optShapes=input_ids:8x128,attention_mask:8x128,token_type_ids:8x128 \
  --maxShapes=input_ids:16x512,attention_mask:16x512,token_type_ids:16x512
```

Worked example for ResNet-50 (one input, dynamic on batch axis only), TRT 11 path:

```bash
python3 -m modelopt.onnx.autocast \
  --onnx_path=resnet50_onnx/model.clean.onnx \
  --output_path=resnet50_onnx/model.fp16.onnx
trtexec \
  --onnx=resnet50_onnx/model.fp16.onnx \
  --saveEngine=resnet50.fp16.plan \
  --memPoolSize=workspace:4096 \
  --minShapes=pixel_values:1x3x224x224 \
  --optShapes=pixel_values:16x3x224x224 \
  --maxShapes=pixel_values:32x3x224x224
```

For purely static-shape models, omit all three shape flags.

### Common errors at this step

- **"Engine plan file is generated on an incompatible device"** — plans are not portable across compute capabilities. Rebuild on the deployment GPU.
- **OOM during build** — lower `--memPoolSize=workspace:N`. If still failing, try `--tacticSources=-CUBLAS_LT` to disable expensive tactic sources.
- **Unsupported operator** — `trtexec` names the op. Route to `trt-unsupported-op` for triage (decompose / plugin / switch frontend / file bug).
- **INT64 warnings** — TRT casts to INT32. Usually safe; if values exceed INT32 range, rerun Step 3's `polygraphy surgeon sanitize --fold-constants`.
- **"Input <name> has missing shape information"** — you missed an input in the `--minShapes / --optShapes / --maxShapes` flags. Compare against Step 3.5's output and add the missing entry.

## Step 5 — Verify numerical parity

This is the step most developers skip — and the one that most often surfaces a silent correctness bug.

Construct the parity command from the same input names used in Step 4. Beyond input shapes, three knobs matter; get them right or the result is misleading:

1. **`--val-range`** for any input with constrained semantics. Polygraphy fills with `[0, 1)` floats by default. For BERT-family `input_ids` you must constrain to vocabulary range, set `attention_mask` to all-ones, and `token_type_ids` to all-zeros — otherwise the model sees garbage and FP16 noise amplifies through the attention softmax.
2. **`--check-error-stat mean`** for transformer outputs. FP16 commonly has a few outlier positions with large abs diff; the mean across all positions is the meaningful metric for downstream use (embedding similarity, classification). Default `max` is too pessimistic for FP16 transformers and produces false failures.
3. **`--seed`** for repeatability across runs and CI environments.

Worked example for BERT — run polygraphy against the **same ONNX file you fed to `trtexec`** (the AutoCast-converted `.fp16.onnx` on TRT 11; the `.clean.onnx` on TRT 10):

```bash
# TRT 11: polygraphy reads the model's dtypes (already FP16 from AutoCast)
polygraphy run bert_onnx/model.fp16.onnx \
  --trt --onnxrt \
  --atol 1e-2 --rtol 1e-2 \
  --check-error-stat mean \
  --input-shapes input_ids:[8,128] attention_mask:[8,128] token_type_ids:[8,128] \
  --val-range input_ids:[0,30000] attention_mask:[1,1] token_type_ids:[0,0] \
  --seed 42

# TRT 10: pass --fp16 explicitly (the polygraphy flag maps to BuilderFlag::kFP16)
polygraphy run bert_onnx/model.clean.onnx \
  --trt --onnxrt \
  --atol 1e-2 --rtol 1e-2 \
  --check-error-stat mean \
  --input-shapes input_ids:[8,128] attention_mask:[8,128] token_type_ids:[8,128] \
  --val-range input_ids:[0,30000] attention_mask:[1,1] token_type_ids:[0,0] \
  --seed 42 \
  --fp16
```

Expected output: `Pass Rate: 100.0%` with `mean_absdiff ≤ 1e-3` per output. `Pass Rate: 0.0%` with `max_absdiff` near 1e-2 but `mean_absdiff` well below tolerance is the outlier pattern above — use `--check-error-stat mean`.

Worked example for ResNet-50 (same FP16 recipe; TRT-11 path shown — for TRT 10 add `--fp16` and read from `model.clean.onnx`):

```bash
polygraphy run resnet50_onnx/model.fp16.onnx \
  --trt --onnxrt \
  --atol 1e-2 --rtol 1e-2 \
  --check-error-stat mean \
  --input-shapes pixel_values:[16,3,224,224] \
  --val-range pixel_values:[0,1] \
  --seed 42
```

Polygraphy runs the model through ONNX Runtime (reference) and TensorRT (engine) on the seeded inputs and compares outputs per the chosen error stat.

### If parity fails

- **Diverges only at FP16, not FP32**: a few layers are losing precision. Try `--strongly-typed` with an explicit FP32 cast on the offending subgraph (Polygraphy names the layer). For LLM-style models, prefer BF16 over FP16 on Hopper/Blackwell.
- **Diverges at all precisions**: the export is wrong. Re-run Steps 2–3 fresh and check `polygraphy inspect model` for shape-inference issues. If the model uses patterns covered by `trt-export-rewrite`, reroute.
- **Diverges only for specific input shapes**: dynamic-shape profile is too narrow. Widen `--minShapes` / `--maxShapes` and rebuild.

## Step 6 — Run the engine

```bash
trtexec --loadEngine=bert.fp16.plan \
  --shapes=input_ids:8x128,attention_mask:8x128,token_type_ids:8x128 \
  --verbose
```

This confirms the saved `.plan` deserializes and runs (Step 5's parity used its own engine, not this file) and reports throughput and per-iteration latency. For production, see the developer guide on `IExecutionContext` and stream-aware execution; the `.plan` is portable to any compatible GPU + TRT runtime.

## Success criteria

The skill is complete when all of the following are true:

- `bert.fp16.plan` exists and `trtexec --loadEngine=...` runs without error.
- `polygraphy run --trt --onnxrt --atol 1e-2 --rtol 1e-2 --check-error-stat mean` reports `PASSED` across all configured input shapes.
- The developer can articulate which precision was used and why.

If any are not true, hand control back with the specific failing diagnostic — do not declare success.

## References

- [Import Workflows Guide (Path 1: ONNX → TensorRT)](https://github.com/NVIDIA/TensorRT/blob/main/documents/import_workflows.md#path-1-onnx--tensorrt)
- [Supported Models matrix](https://github.com/NVIDIA/TensorRT/blob/main/documents/supported_models.md)
- [Polygraphy documentation](https://docs.nvidia.com/deeplearning/tensorrt/polygraphy/index.html)
- [`samples/sampleOnnxMNIST/`](https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleOnnxMNIST) for the C++ equivalent of the build flow
