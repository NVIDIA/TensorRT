---
name: trt-torch-quickstart
description: >
  Compile a PyTorch model to a TensorRT engine via Torch-TensorRT — AOT or
  JIT — under the new strong-typing default. Use when the user
  compiles PyTorch to TensorRT without ONNX, hits "enabled_precisions
  should not be used when use_explicit_typing=True", sees Dynamo graph
  breaks or PyTorch fallback, debugs ABI errors at import torch_tensorrt, or
  needs the compatible torch / torch_tensorrt / tensorrt-cu13 version pins for
  TensorRT 11. Triggers: torch_tensorrt, torch_tensorrt.dynamo.compile,
  torch.compile backend torch_tensorrt, pytorch to tensorrt, ExportedProgram,
  Dynamo graph break, use_explicit_typing, enabled_precisions,
  torch_tensorrt.Input, min_block_size, truncate_double, tensorrt-cu13,
  version pinning, version compatibility. Adjacent skills: `trt-onnx-quickstart`,
  `trt-cpp-runtime-quickstart`. LLM token generation belongs in TensorRT-LLM.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
  version: "1.0"
  tags:
    - torch-tensorrt
    - dynamo
    - pytorch
    - quickstart
    - fp16
---

# Torch-TensorRT Quickstart

Convert a PyTorch `nn.Module` to a TensorRT engine using `torch_tensorrt` (the [Torch-TensorRT](https://github.com/pytorch/TensorRT) frontend). Covers the AOT path (for production and C++ deploy) and the JIT path (for Python-only inference).

## When to Use

| Scenario | Use this skill? |
|----------|-----------------|
| PyTorch `nn.Module`; want a TensorRT engine without writing an ONNX intermediate | Yes |
| Model uses ops that don't export cleanly to ONNX (custom autograd, dynamic control flow) | Yes |
| Want Torch-TensorRT's automatic fallback of unsupported subgraphs to PyTorch | Yes |
| Need a serialized engine for C++ deploy from a PyTorch source | Yes — use AOT path here, then `trt-cpp-runtime-quickstart` for the C++ load |
| Have a working ONNX file already | No — use `trt-onnx-quickstart` |
| LLM token generation (Llama, Mistral, Qwen text gen) | **No** — route to [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |
| Have a `.plan` file already and want to run it from C++ | No — use `trt-cpp-runtime-quickstart` |
| Migrating an existing weakly-typed TRT network to strongly-typed | No — use `trt-strong-typing-migration` |

## Prerequisites

Torch-TensorRT versions are tightly coupled to a (torch, tensorrt-cu13, CUDA) triple; mismatches produce ABI errors at `import torch_tensorrt`. **Pin matrix source of truth:** the [pytorch/TensorRT releases page](https://github.com/pytorch/TensorRT/releases) — each release lists the exact `torch` / `tensorrt` / CUDA versions it was built against. Do not invent pins from memory.

For TensorRT 11.0:

| Component | Required | Notes |
|-----------|----------|-------|
| TensorRT | 11.x (`tensorrt-cu13` wheel) | Mixing `-cu12` and `-cu13` wheels breaks imports. |
| CUDA toolkit | 13.x | Driver R590+. |
| Python | ≥ 3.10 | TRT 11 dropped 3.9 and earlier. |
| `torch` | Match the `torch_tensorrt` release notes for the chosen `torch_tensorrt` version. | |
| `torch_tensorrt` | The release matching the TRT 11.0 RC — see [pytorch/TensorRT releases](https://github.com/pytorch/TensorRT/releases). | |

**Recommended environment**: NGC TensorRT container with PyTorch (`nvcr.io/nvidia/pytorch:<tag>` or `nvcr.io/nvidia/tensorrt:<tag>` with `torch_tensorrt` pip-installed on top). Run with `--gpus all`.

Verification:
```bash
python3 -c "import torch, torch_tensorrt, tensorrt; print(torch.__version__, torch_tensorrt.__version__, tensorrt.__version__)"
```

## Step 1 — Load and prepare the model

Use a `torch.nn.Module` in eval mode. Torch-TensorRT will trace through it.

```python
import torch
import torchvision.models as models

model = models.resnet50(weights=None).eval().cuda()
example = torch.randn(1, 3, 224, 224, device="cuda")
```

Notes:
- `weights=None` skips the download — fine for shape/perf testing. For accuracy work, load real weights.
- Model must be on CUDA before compile. Torch-TensorRT does not move it for you.
- `.eval()` matters: BatchNorm and Dropout behave differently in train mode and can produce different engines.

## Step 2 — Compile to a TensorRT engine (AOT)

Two compile paths — pick by deployment target:

| Goal | Path | API |
|------|------|-----|
| Serialized engine (production, C++ deploy, reuse) | **AOT** — this section | `torch_tensorrt.dynamo.compile` → `torch_tensorrt.save` |
| In-process Python callable only (no serialized engine) | **JIT** — end of this section | `torch.compile(backend="torch_tensorrt")` |

The AOT path uses `torch_tensorrt.dynamo.compile` and returns a serializable `ExportedProgram`.

**Important — strong typing is the default in torch_tensorrt ≥ 2.12.** With `use_explicit_typing=True` (the default), engine precision is **inferred from the exported model's dtype** — passing `enabled_precisions={torch.float16}` raises `AssertionError`. To compile in FP16, cast the model and example tensor to FP16 before exporting:

```python
import torch
import torch_tensorrt

model = MyModel().eval().cuda().half()                          # cast to FP16
example = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float16)

trt_gm = torch_tensorrt.dynamo.compile(
    torch.export.export(model, (example,)),
    inputs=[example],
    # No enabled_precisions — precision comes from the model dtype.
    truncate_double=True,
    min_block_size=1,
)

# Serialize for later / C++ loading — use torch_tensorrt.save, NOT raw bytes.
torch_tensorrt.save(trt_gm, "resnet50_trt.ep", inputs=[example])
```

**Do not extract raw engine bytes via `submod.engine` and write them with `open(...).write(...)`.** That bypasses Torch-TensorRT's metadata wrapper: the blob will not round-trip through `torch_tensorrt.load(...)` and silently drops the dispatch graph that handles partial-fallback subgraphs. `torch_tensorrt.save(trt_gm, path, inputs=...)` is the only supported serialization path.

For mixed precision or to override the model dtype, set `use_explicit_typing=False` and `enabled_precisions` applies as before (the weakly-typed path, deprecated in TRT 11):

```python
trt_gm = torch_tensorrt.dynamo.compile(
    torch.export.export(model, (example,)),
    inputs=[example],
    use_explicit_typing=False,                # weakly-typed (deprecated)
    enabled_precisions={torch.float16, torch.float32},
    truncate_double=True,
)
```

Key arguments:
- `truncate_double=True`: silently downcasts FP64 constants. Without it, any FP64 op forces a partition boundary.
- `min_block_size`: smallest subgraph (in node count) worth handing to TRT. `1` is aggressive; default `5` avoids tiny TRT subgraphs that don't pay back their launch overhead.
- `workspace_size`: bytes of scratch memory the TRT builder can use. Leave unset to let TRT decide.

**JIT alternative (Python-only inference):** if you just want a callable and not a serialized engine, use `torch.compile` with the Torch-TensorRT backend. Under the strong-typing default, precision comes from the model dtype — `.half()` the model for FP16 rather than passing `enabled_precisions`:
```python
model = model.half()  # FP16 inferred from model dtype (strong typing default)
example = example.half()
trt_model = torch.compile(model, backend="torch_tensorrt")
out = trt_model(example)  # compiles lazily on the first call, then runs through TRT in-process
```
This skips `torch.export` and gives a `torch.compile`-wrapped callable. It cannot be loaded from C++ — use the AOT path above if you need that.

## Step 3 — Dynamic shapes

Wrap inputs in `torch_tensorrt.Input` with `min_shape`/`opt_shape`/`max_shape`:

```python
dynamic_input = torch_tensorrt.Input(
    min_shape=(1, 3, 224, 224),
    opt_shape=(8, 3, 224, 224),
    max_shape=(32, 3, 224, 224),
    dtype=torch.float16,
)
trt_gm = torch_tensorrt.dynamo.compile(
    torch.export.export(model, (example,), dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=32)}}),
    inputs=[dynamic_input],
    # Strong typing: FP16 comes from the input/model dtype, not enabled_precisions.
)
```

Pitfalls:
- The `dynamic_shapes=` argument to `torch.export.export` and the `Input` ranges must agree. A mismatch produces a builder error 2–3 minutes into compilation, not at export time.
- `opt_shape` is what TRT tunes kernels for. Set it to the most common runtime shape, not the midpoint.
- Only mark dimensions dynamic that actually vary at runtime; unnecessary dynamism degrades performance.

## Step 4 — Run inference and verify

Raw-tensor comparison (`assert_close`) is fine for FP32 but **misleading for FP16** — a few outlier positions exceed the tolerance even when downstream behavior is identical. Use a **semantic check** appropriate to the model type:

```python
with torch.inference_mode():
    torch_out = model(example)
    trt_out = trt_gm(example)
```

**Image classification** — top-1 (or top-5) class match, plus mean-softmax-distance:

```python
assert torch.equal(torch_out.argmax(-1), trt_out.argmax(-1)), "top-1 mismatch"
mean_prob_diff = (torch.softmax(torch_out.float(), -1) - torch.softmax(trt_out.float(), -1)).abs().mean()
assert mean_prob_diff < 1e-3
```

**Sentence / token embeddings** — cosine similarity per token, mean across the batch:

```python
cos = torch.nn.functional.cosine_similarity(torch_out.flatten(0, -2), trt_out.flatten(0, -2), dim=-1)
assert cos.mean() > 0.999
```

**Detection / regression heads** — bounded raw-tensor closeness on the meaningful sub-tensor (e.g., bbox coordinates), softmax check on the class logits.

**FP32 sanity** — for any model, an FP32 (no `.half()`) run should pass `torch.testing.assert_close(trt_out, torch_out, rtol=1e-4, atol=1e-4)`. If that fails, the issue is in the compilation, not numeric precision.

## Loading the serialized engine

**From Python**:
```python
import torch_tensorrt
loaded = torch_tensorrt.load("resnet50_trt.ep").module()
out = loaded(example)
```

**From C++**: this skill does NOT cover C++ loading. The `.ep` file `torch_tensorrt.save` produces by default is a `torch.export` archive for the **Python** `torch_tensorrt.load(...)` path — it is *not* directly consumable by the plain TensorRT C++ runtime (`IRuntime` / `deserializeCudaEngine`), which expects a serialized TensorRT engine (`.plan`). Two supported C++ routes:

- Save with `output_format="torchscript"` and deploy the TorchScript module with libtorch + the Torch-TensorRT C++ runtime.
- If you only need the raw TensorRT engine in C++, build it through the ONNX path (`trt-onnx-quickstart`) and load the resulting `.plan` with `trt-cpp-runtime-quickstart` (modern `IRuntime` + `enqueueV3` + `setTensorAddress`).

Do not inline a C++ snippet here — it will diverge from the canonical pattern.

## Common issues

**`RuntimeError: Trying to create tensor with negative dimension`** during export — the model has a shape-dependent control flow path that `torch.export` can't trace. Either rewrite with `torch.cond` / `torch.where`, or fall back to `ir="dynamo"` + `min_block_size=1` so the unsupported region runs in PyTorch.

**`Unsupported operator` warnings** — Torch-TensorRT partitions around them and runs those nodes in PyTorch. Set `require_full_compilation=True` to turn this into an error; use during development to find what's actually falling back.

## Diagnosing excessive PyTorch fallback

If the compile completes but most of the model fell back to PyTorch (only a few subgraphs run on TRT), three knobs in order:

1. **Identify what fell back.** Run with `TORCH_LOGS="graph_breaks"` and `require_full_compilation=True` to escalate fallbacks to errors that name the offending op. Without this you don't know what the autotuner rejected.
   ```bash
   TORCH_LOGS="graph_breaks" python3 your_compile.py
   ```
2. **Reconsider `min_block_size`.** This is a real trade-off — not just a knob.
   - Default `min_block_size=5`: Torch-TensorRT only hands a subgraph to TRT if it has ≥5 nodes. Tiny TRT subgraphs cost more in kernel-launch overhead than they save.
   - `min_block_size=1`: aggressive — every supported op goes to TRT, even single-node subgraphs. Useful to see what's *theoretically* supported, but commonly slower at inference due to the per-subgraph launch tax.
   - Recommended: leave at default `5` for production; drop to `1` only during diagnosis to see maximum TRT coverage.
3. **Look for shape-dependent control flow.** `torch.export` traces only one branch of `if x.shape[0] > 0:` style code; rewrite with `torch.where`, `torch.cond`, or lift the condition out of the model. The trace error names the offending op.

If fallback persists after all three, you have a genuine unsupported op — write a converter (see Torch-TensorRT upstream docs) or switch to the ONNX path (`trt-onnx-quickstart`).

**Engine builds but output is garbage** — almost always a dtype issue. Check:
1. Model/input dtype matches your pipeline — under strong typing precision follows the model dtype (`.half()` for FP16); `enabled_precisions` applies only under weak typing (`use_explicit_typing=False`).
2. `truncate_double=True` if the model has any FP64 constants (common in positional encodings).
3. The model was `.eval()` before export.

**Compile takes >10 minutes** — turn on the builder log to see what's going on:
```python
import torch_tensorrt.logging as ttlog
ttlog.set_reportable_log_level(ttlog.Level.Info)
```
Usually it's the autotuner exploring kernel variants for a heavy GEMM/conv. To cap exploration, set `workspace_size` smaller (less scratch → fewer candidates).

**`ImportError: cannot import name 'XYZ' from 'torch_tensorrt'`** — version mismatch. Confirm the pin matrix; the `torch_tensorrt` Python API moves between minor versions.

## Numerical debugging

If `assert_close` fails by more than ~5× tolerance:
1. Re-run in FP32 — drop the model's `.half()` (or under weak typing, `enabled_precisions={torch.float32}`). If that matches PyTorch, the issue is FP16 accumulator drift in a specific op — usually a softmax or LayerNorm. The pragmatic fix is to wrap the offending submodule in a module excluded from the compile (`min_block_size` won't help; explicit exclusion is needed). See the [Torch-TensorRT lowering guide](https://pytorch.org/TensorRT/) for the version-specific exclusion API.
2. Sanity-check at FP32 first, then re-introduce FP16 and bisect — almost always reveals one offending op.
3. For FP8: check that the calibration distribution actually covers the runtime distribution. FP8 has no headroom for outliers.

See `trt-strong-typing-migration` for the broader weak-vs-strong typing discussion. By default Torch-TensorRT produces strongly-typed engines in TRT 11.0 — types inferred from the exported program — but you can override.

## What this skill is not

- Not a guide to writing custom Torch-TensorRT converters. Use the upstream Torch-TensorRT docs for that.
- Not for TensorRT-LLM / LLM inference — use the TRT-LLM `examples/` flow instead.
- Not for QAT (quantization-aware training) — Torch-TensorRT consumes the QAT model but the training loop is upstream.

## References

- Upstream Torch-TensorRT: https://pytorch.org/TensorRT/
- Version pin matrix: [pytorch/TensorRT releases](https://github.com/pytorch/TensorRT/releases)
- Related skills: `trt-onnx-quickstart`, `trt-strong-typing-migration`, `trt-cpp-runtime-quickstart`
