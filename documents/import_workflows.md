# TensorRT Import Workflows — Step-by-Step Guide

This guide, together with [`supported_models.md`](./supported_models.md), centralizes TensorRT import-path guidance that was previously fragmented across release notes, samples, blogs, and forum posts.

TensorRT supports several paths for bringing a trained model into an optimized inference engine. This guide walks through each path end to end — install, export, build, verify — with runnable commands and the most common pitfalls called out inline.

## Table of Contents

- [Choosing a Path](#choosing-a-path)
- [Common Prerequisites](#common-prerequisites)
- [Path 1: ONNX → TensorRT](#path-1-onnx--tensorrt)
- [Path 2: Torch-TensorRT (PyTorch native)](#path-2-torch-tensorrt-pytorch-native)
- [Path 3: Hugging Face Hub Models → TensorRT](#path-3-hugging-face-hub-models--tensorrt)
- [Path 4: Direct Network Definition API (C++/Python)](#path-4-direct-network-definition-api-cpython)
- [Adding a Custom Operator / Plugin](#adding-a-custom-operator--plugin)
- [AI-Assisted Model Rewriting for Export](#ai-assisted-model-rewriting-for-export)
- [Verifying an Engine](#verifying-an-engine)
- [Tools Reference](#tools-reference)
- [Troubleshooting & Insights](#troubleshooting--insights)

---

## Choosing a Path

| You have…                                   | Recommended path                   | Notes |
|---------------------------------------------|------------------------------------|-------|
| An ONNX file from any framework             | [ONNX → TensorRT](#path-1-onnx--tensorrt) | Most portable path. Build via `trtexec`, Python API, or Polygraphy. |
| A trained PyTorch model, want fastest onboarding | [Torch-TensorRT](#path-2-torch-tensorrt-pytorch-native) | Python-first, stays in PyTorch. Best for iterative development. |
| A Hugging Face Hub model (LLM, diffusion, etc.) | [Hugging Face Hub Models](#path-3-hugging-face-hub-models--tensorrt) | Export → ONNX → TRT for most models; use TensorRT-LLM directly for LLM generation. |
| A model architecture authored in C++ or a custom research stack | [Network Definition API](#path-4-direct-network-definition-api-cpython) | Maximum control, maximum effort. |
| An existing TRT plan that just needs to be run | See [Verifying an Engine](#verifying-an-engine) | Not covered by this guide — see the Developer Guide on deserialization. |

---

## Common Prerequisites

All paths below assume:

1. A supported NVIDIA GPU (see the [Support Matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)).
2. NVIDIA driver + CUDA matching your TensorRT release. For TRT 11.x: **CUDA 13.x**.
3. Python 3.10+ if using Python APIs. C++ paths need a C++17 compiler.

### Install TensorRT (Python, pip)

```bash
# Python TRT runtime + Python bindings
pip install --extra-index-url https://pypi.nvidia.com tensorrt-cu13
```

> **Note:** Always use `-cu13` packages with TRT 11.x. Do not mix `-cu12` wheels.

### Install TensorRT (system packages)

Follow the [Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/overview.html) for `.deb` / `.tar` / container options. The NGC container `nvcr.io/nvidia/tensorrt:<tag>` is the fastest way to get a known-good environment.

### Verify the install

```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
trtexec --help | head -5
```

---

## Path 1: ONNX → TensorRT

The ONNX path is the most portable way to bring models from PyTorch, TensorFlow, JAX, or any framework with an ONNX exporter.

> **ONNX compatibility note:** TensorRT does not support every ONNX operator or every ONNX opset version. Operator and opset coverage depends on the TensorRT release, so validate exported models with `trtexec` or Polygraphy and be prepared to update the exporter/opset, rewrite unsupported subgraphs, or provide custom plugins.

### 1. Export to ONNX

**PyTorch (dynamo exporter, preferred for TRT 11+):**

```python
import torch

model = MyModel().eval().cuda()
example = torch.randn(1, 3, 224, 224, device="cuda")

onnx_program = torch.onnx.export(
    model,
    (example,),
    "model.onnx",
    dynamo=True,            # Use the dynamo exporter
    dynamic_shapes=None,    # Or specify with torch.export.Dim
)
```

**TensorFlow / Keras:** use `tf2onnx`:

```bash
python -m tf2onnx.convert --saved-model ./saved_model --output model.onnx --opset 20
```

### 2. (Optional) Simplify & sanitize

```bash
pip install onnx onnxsim polygraphy
python -m onnxsim model.onnx model.sim.onnx
polygraphy surgeon sanitize model.sim.onnx -o model.clean.onnx --fold-constants
```

### 3. Build a TensorRT engine

An ONNX file must be converted to a serialized TensorRT plan (`.plan` / `.engine`) before the runtime can execute it — the TensorRT runtime deserializes plans, it does **not** parse ONNX. The three options below are interchangeable front-ends that call the same `IBuilder` + `nvonnxparser::IParser` underneath; pick whichever fits your workflow.

> **Note on ONNX Runtime:** If you've seen "TensorRT executes ONNX directly," that refers to ONNX Runtime's `TensorrtExecutionProvider`, which lazy-builds a TRT engine internally on first call. That's ORT integrating TRT, not the TRT runtime itself.

**Option A — `trtexec` (CLI, fastest to try):**

```bash
trtexec \
  --onnx=model.clean.onnx \
  --saveEngine=model.plan \
  --memPoolSize=workspace:4096 \
  --fp16                        # or --bf16, --int8, --fp8 (platform-dependent)
```

For dynamic shapes, add:

```bash
  --minShapes=input:1x3x224x224 \
  --optShapes=input:8x3x224x224 \
  --maxShapes=input:16x3x224x224
```

**Option B — Python (`tensorrt.Builder` + `OnnxParser`):**

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
network = builder.create_network(flags)

parser = trt.OnnxParser(network, logger)
with open("model.clean.onnx", "rb") as f:
    assert parser.parse(f.read()), [parser.get_error(i) for i in range(parser.num_errors)]

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
serialized = builder.build_serialized_network(network, config)
open("model.plan", "wb").write(serialized)
```

**Option C — Polygraphy (scriptable, good for pipelines):**

```bash
polygraphy convert model.clean.onnx \
  --convert-to trt \
  --fp16 \
  --workspace 4G \
  --trt-min-shapes input:[1,3,224,224] \
  --trt-opt-shapes input:[8,3,224,224] \
  --trt-max-shapes input:[16,3,224,224] \
  -o model.plan
```

C++ users: see `samples/sampleOnnxMNIST/` for the equivalent `IBuilder` + `IParser` flow.

### 4. Run the engine

See [Verifying an Engine](#verifying-an-engine).

### Common pitfalls

- **Unsupported op.** `trtexec` will name the op. Options: update your exporter/opset, rewrite the subgraph, or write a [custom plugin](#adding-a-custom-operator--plugin).
- **Shape inference failures.** Run `polygraphy inspect model model.onnx --show attrs` to confirm every tensor has a known rank.
- **Constant-folding surprises.** `polygraphy surgeon sanitize --fold-constants` often removes spurious dynamic axes introduced during export.
- **`INT64` tensors.** TRT will warn and cast to `INT32`; if values exceed `INT32` range, sanitize first.

---

## Path 2: Torch-TensorRT (PyTorch native)

The Dynamo frontend (`torch.compile(backend="tensorrt")`) is the **active, preferred** path. JIT/tracing-based `torch_tensorrt.compile` still works but receives minimal new investment.

### 1. Install

```bash
pip install --extra-index-url https://pypi.nvidia.com torch-tensorrt tensorrt-cu13
```

### 2. Compile (AOT — produces a standalone artifact)

```python
import torch
import torch_tensorrt as torch_trt

model = MyModel().eval().cuda().to(torch.float16)
example = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float16)

trt_gm = torch_trt.dynamo.compile(
    torch.export.export(model, (example,)),
    inputs=[example],
    enabled_precisions={torch.float16},
    workspace_size=4 << 30,
)

# Save and reload
torch_trt.save(trt_gm, "model.ep", inputs=[example])
loaded = torch.export.load("model.ep").module()
```

### 3. Compile (JIT — first call triggers compilation)

```python
import torch

compiled = torch.compile(model, backend="tensorrt", options={"enabled_precisions": {torch.float16}})
out = compiled(example)   # Triggers TRT compilation + cache on first call
```

### 4. Run

```python
with torch.no_grad():
    y = trt_gm(example)   # or compiled(example)
```

### Common pitfalls

- **Graph breaks** fall back to eager. Inspect with `TORCH_LOGS="graph_breaks"`. Eliminate them by lifting Python conditionals, avoiding `.item()` calls, and using `torch.cond` where possible.
- **Dynamic shapes** need explicit `torch.export.Dim(...)` annotations for AOT. JIT handles them but may recompile per shape.
- **Custom ops / plugins.** Torch-TRT converters live at `core/conversion/converters/` in the Torch-TensorRT repo. To add one, see the [Torch-TensorRT converter guide](https://docs.pytorch.org/TensorRT/contributors/writing_converters.html).

### Insights from historical issues

- `torch.export` in PyTorch 2.4+ is required for stable Dynamo AOT. Earlier versions fall back to torchscript tracing, which has been deprecated.
- Mixed precision: prefer `enabled_precisions={torch.float16}` over `torch.float32` unless a specific layer loses accuracy. For BF16 targets (Blackwell, Hopper), use `{torch.bfloat16}` and cast inputs accordingly.

---

## Path 3: Hugging Face Hub Models → TensorRT

> **For LLM generation, use [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) directly.** It is NVIDIA's active, production-grade path for Hugging Face LLMs — handles KV-cache, batching, paged attention, FP8/INT4 quantization, speculative decoding, and multi-GPU tensor/pipeline parallelism. The `optimum-nvidia` wrapper described in Option C below has not seen a release in over a year as of 2026-Q2 and is not recommended for new work.

For non-LLM Hugging Face Hub models (encoders, vision, diffusion components, speech), prefer Option A.

### Option A — Export to ONNX, then Path 1 (recommended default)

Most HF models export cleanly through `optimum`'s ONNX exporter:

```bash
pip install optimum-onnx    # ONNX integration moved out of the `optimum` package in v2
optimum-cli export onnx \
  --model google-bert/bert-base-uncased \
  --task feature-extraction \
  bert_onnx/

trtexec --onnx=bert_onnx/model.onnx --saveEngine=bert.plan --fp16
```

Then run through the standard [Path 1 build](#3-build-a-tensorrt-engine). This is the most durable HF → TRT path because it depends only on actively-maintained pieces (`optimum-onnx`, `trtexec`/Python builder).

### Option B — Torch-TensorRT

Covered in [Path 2](#path-2-torch-tensorrt-pytorch-native). Load with `transformers`, move to CUDA, and compile via `torch_tensorrt.dynamo.compile` or `torch.compile(backend="tensorrt")`. Good fit when you want to stay inside PyTorch and iterate quickly.

### Option C — `optimum-nvidia` (convenience wrapper; upstream is stale)

```bash
pip install optimum-nvidia
```

```python
from optimum.nvidia import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    use_fp8=False,  # Or True on Hopper/Blackwell
)
out = model.generate(input_ids, max_new_tokens=128)
```

> **Status warning:** The last `optimum-nvidia` release (`v0.1.0b9`) shipped on 2025-01-21 and there have been no releases since. It still pins an older `tensorrt-llm` via `third-party/`. Verify the pinned versions match your TRT/CUDA stack before adopting, and prefer TensorRT-LLM directly (see callout at the top of this section) for anything production-facing.

### Common pitfalls

- **Tokenizer padding.** Hugging Face defaults to right-padding; some decoder models expect left-padding for generation. Mismatch produces silently-wrong logits.
- **KV-cache shapes.** For generative models, dynamic shapes along the sequence axis are mandatory. Use `optimum-nvidia` or hand-author shape profiles.
- **Diffusion pipelines** must be split by component (text encoder, UNet/DiT, VAE) — TRT cannot ingest the whole pipeline. See [supported_models.md](./supported_models.md) for component-by-component support.

---

## Path 4: Direct Network Definition API (C++/Python)

Use this only when neither ONNX nor PyTorch can express what you need (e.g., custom research architectures, ultra-tight control over layer choice).

```python
import numpy as np
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

x = network.add_input("x", trt.float16, (-1, 3, 224, 224))
w = trt.Weights(np.random.randn(64, 3, 7, 7).astype(np.float16))
conv = network.add_convolution_nd(x, 64, (7, 7), w, trt.Weights())
conv.stride_nd = (2, 2)
network.mark_output(conv.get_output(0))

config = builder.create_builder_config()
profile = builder.create_optimization_profile()
profile.set_shape("x", (1, 3, 224, 224), (8, 3, 224, 224), (16, 3, 224, 224))
config.add_optimization_profile(profile)

plan = builder.build_serialized_network(network, config)
open("model.plan", "wb").write(plan)
```

The C++ equivalent follows the same structure; see `samples/sampleINT8API` and `samples/python/refactored/2_construct_network_with_layer_apis/` for a runnable reference.

---

## Adding a Custom Operator / Plugin

When the importer reports an unsupported op:

1. **Check `tensorrt.IPluginRegistry`** — the op may already have a plugin you haven't loaded.
2. **Write a plugin** implementing `IPluginV3` (preferred for TRT 10+).
3. **Register it** via `REGISTER_TENSORRT_PLUGIN` (C++) or `trt.get_plugin_registry().register_creator(...)` (Python).
4. **Wire into ONNX** by naming the op `mydomain::MyPlugin` during export and supplying a matching plugin name.
5. **Torch-TensorRT custom converters** live in `core/conversion/converters/` — see the Torch-TRT docs.

A runnable reference: `samples/python/aliased_io_plugin/` in this repo.

For migration details, see the TensorRT Developer Guide section on [migrating V2 plugins to IPluginV3](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html#migrating-v2-plugins-to-ipluginv3).

---

## AI-Assisted Model Rewriting for Export

Hugging Face Hub models often don't export cleanly on the first try. Modern library code uses patterns — complex-number arithmetic, data-dependent control flow, non-tensor forward arguments, variable-length outputs — that `torch.export` / `torch.onnx.export` / Torch-TensorRT cannot trace directly. The usual workaround is *not* to modify the upstream library on disk, but to **monkey-patch equivalent, export-friendly variants at runtime** before export, then transparently swap the compiled module back into the pipeline.

This is repetitive, carefully-scoped work: read an upstream implementation, identify the one pattern that breaks the exporter, write a behaviorally-equivalent replacement, preserve everything the rest of the library expects. **It is exactly the kind of task where an AI coding agent pays off** — the agent reads hundreds of lines of upstream source (diffusers, transformers), proposes an equivalent formulation, and iterates against tracer errors without losing the thread. The worked example below is the output of this process for a non-trivial diffusion pipeline.

### Worked example: Qwen-Image (`diffusers`) → Torch-TensorRT AOT

The condensed pattern below comes from a Qwen-Image Torch-TensorRT AOT validation script and is written to be self-contained for OSS users.

The script compiles all three heavy components of `QwenImagePipeline` — text encoder, MMDiT transformer, VAE decoder — via `torch_tensorrt.dynamo.compile` and re-injects the compiled modules back into the pipeline. Five distinct export blockers had to be fixed, each representative of a broader class:

#### 1. Complex-number RoPE math → pre-compute real-valued cos/sin

Diffusers' `QwenEmbedRope` stores rotary-embedding frequencies as `torch.complex64` buffers and calls `torch.view_as_real(...)` inside the forward path. Torch-TensorRT's complex-graph detection cannot handle residual complex ops and segfaults.

**Fix:** pre-materialize real-valued `cos`/`sin` buffers on the module, patch the forward to read from them, **but keep the original complex buffers intact** (the exporter still probes them on entry):

```python
pos = torch.view_as_real(module.pos_freqs)
module._real_pos_cos = pos[..., 0].repeat_interleave(2, dim=-1).contiguous()
module._real_pos_sin = pos[..., 1].repeat_interleave(2, dim=-1).contiguous()
# Do NOT overwrite pos_freqs/neg_freqs — complex_graph_detection still reads them.
```

Where an agent helps: reading `diffusers/models/transformers/transformer_qwenimage.py` (≈1.5k lines), pinpointing the two `forward` methods that touch complex tensors, and deriving the real-valued equivalent without changing numerics.

#### 2. Non-tensor forward arguments → bake them into a wrapper

The transformer takes `img_shapes: list[list[tuple[int, int, int]]]`, which `torch.export` refuses to trace.

**Fix:** a thin wrapper stores the shape list as a constructor arg so the exported forward signature is pure tensors:

```python
class QwenImageTransformerAOTWrapper(nn.Module):
    def __init__(self, transformer, img_shapes):
        super().__init__()
        self.transformer = transformer
        self.img_shapes = img_shapes
    def forward(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, timestep):
        return self.transformer(
            hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask, timestep=timestep,
            img_shapes=self.img_shapes, return_dict=False)[0]
```

#### 3. HF-style output dataclasses → unwrap before export, re-wrap on the way back

Pipelines expect outputs like `Transformer2DModelOutput(sample=...)` or objects with `.hidden_states`. Torch-TRT needs plain tensors. Solve in two layers:

- **Export wrappers** (`TextEncoderAOTWrapper`, `VaeDecoderAOTWrapper`) return a bare tensor.
- **Reinjection proxies** (`CompiledTextEncoderProxy`, `CompiledTransformerProxy`, `CompiledVAEProxy`) expose every attribute the pipeline touches on the original module (`config`, `dtype`, `device`, `cache_context()`, etc.) and re-wrap outputs into the expected dataclass so the surrounding pipeline code does not notice the swap.

#### 4. Variable-length tokenization → force a static prompt shape

The default `encode_prompt` path slices per-sample by attention mask, producing variable-length hidden states. Torch-TRT requires static shapes for AOT.

**Fix:** monkey-patch `_get_qwen_prompt_embeds` onto the pipeline so tokenization always produces the same `[B, S]` shape the TRT text encoder was compiled with:

```python
pipe._get_qwen_prompt_embeds = MethodType(_get_qwen_prompt_embeds_fixed, pipe)
```

Where an agent helps: locating the (undocumented) method the pipeline dispatches to, reproducing the trimming/padding logic with a fixed `max_seq_len`, and ensuring dtype/device alignment with the compiled module.

#### 5. Memory-bound compilation → hint the resource partitioner

The full pipeline exceeds a single GPU's working set if everything is compiled greedily:

```python
import torch_tensorrt as torch_trt

# cpu_memory_budget is in bytes — tune to your host RAM headroom.
CPU_MEMORY_BUDGET_BYTES = 32 * 1024**3  # 32 GiB

torch_trt.compile(module, ir="dynamo", arg_inputs=inputs,
                  require_full_compilation=False,
                  enable_resource_partitioning=True,
                  cpu_memory_budget=CPU_MEMORY_BUDGET_BYTES,
                  truncate_double=True, optimization_level=1)
```

### Pattern takeaways

Apply this recipe to any non-trivial HF export:

1. **Run the naive export first.** Let the tracer/exporter fail and read the error carefully — the failing op/pattern tells you what to patch.
2. **Patch at runtime, not on disk.** Monkey-patch upstream modules from your export script so you never fork the library.
3. **Wrap for the exporter; proxy for the pipeline.** A `*AOTWrapper` unwraps HF outputs for export; a `Compiled*Proxy` re-wraps them on the way back and carries every attribute the downstream code reads.
4. **Preserve unobserved invariants.** When the exporter probes a buffer (e.g., `complex_graph_detection` reading `pos_freqs`), don't overwrite that buffer — add a parallel real-valued one.
5. **Iterate with an agent in the loop.** Each of the fixes above took one or two read-diagnose-patch cycles against upstream source; an agent can execute those cycles faster than a human skimming unfamiliar library code, while you review the diffs.

The end result for Qwen-Image: a pipeline whose heavy components all run on Torch-TensorRT, with zero changes to installed `diffusers` / `transformers` / `torch_tensorrt`, and a generated image that matches the eager pipeline's output on a fixed seed.

---

## Verifying an Engine

```bash
# Sanity-check performance and numerics
trtexec --loadEngine=model.plan --shapes=input:1x3x224x224 --verbose

# Side-by-side accuracy against the ONNX source
polygraphy run model.onnx --trt --onnxrt \
  --atol 1e-3 --rtol 1e-3 --input-shapes input:[1,3,224,224]
```

For LLM-style generation, compare token-by-token against the reference implementation on a deterministic seed before trusting a new engine.

---

## Tools Reference

| Tool        | What it does                                                  | Install |
|-------------|---------------------------------------------------------------|---------|
| `trtexec`   | Build + run + profile engines from ONNX or serialized plans  | Bundled with TRT |
| `polygraphy`| Inspect, sanitize, compare, and debug models at every stage  | `pip install polygraphy` |
| `onnxsim`   | Fold constants and simplify ONNX graphs                      | `pip install onnxsim` |
| `onnx-graphsurgeon` | Programmatic ONNX graph edits                         | `pip install onnx-graphsurgeon` |
| `nsys` / `ncu`      | Runtime profiling and kernel analysis                 | NVIDIA CUDA Toolkit |

---

## Troubleshooting & Insights

Shared learnings from customer-reported issues. Extend this list liberally — it is the single most valuable section of this guide.

- **"Engine plan file is generated on an incompatible device"** — plans are not portable across compute capabilities. Rebuild on the deployment GPU, or target multiple SMs at build time.
- **Accuracy gap vs. framework** — start with `polygraphy run ... --onnxrt --trt --atol ...` to localize. If an FP16 engine diverges, try `--stronglyTyped` + explicit FP32 cast on the offending subgraph.
- **OOM during build** — lower `--memPoolSize=workspace:N` or disable tactic sources you don't need (`--tacticSources=-CUBLAS_LT`).
- **Slow first inference** — CUDA kernel JIT + plan deserialization cost is one-time. Warm up with ≥3 iterations before timing.
- **`IShapeLayer` / data-dependent shapes** — some patterns (e.g. `where(cond, x, y)` with dynamic output shape) require `IShapeLayer` + profile-shaped tensors. See the Developer Guide chapter on dynamic shapes.
