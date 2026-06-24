---
name: trt-strong-typing-migration
description: >-
  Migrate a TensorRT build from weak typing (deprecated 10.12, removed 11.0) to
  strong typing — across Python INetworkDefinition builders, the trtexec CLI, and
  C++ builder code. Use when a TRT 11 upgrade breaks a weakly-typed build. Triggers: weakly typed to strongly typed,
  kSTRONGLY_TYPED, weak typing deprecated, kFP16/kINT8 removed, setPrecision rejected,
  setComputePrecision deprecated, do I still need --stronglyTyped, how to add the
  kSTRONGLY_TYPED flag, ModelOpt autocast, INT8 on TRT 11. NOT for ONNX import
  (`trt-onnx-quickstart`), Torch-TRT (`trt-torch-quickstart`), or C++ deploy
  (`trt-cpp-runtime-quickstart`).
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
  version: "1.0"
  tags:
    - migration
    - strong-typing
    - autocast
    - modelopt
    - builder
---

# Migrate Weakly-Typed to Strongly-Typed TensorRT Build

Strong typing is the recommended build mode for TensorRT 11.x; weak typing was
deprecated in 10.12 and is scheduled for removal. This skill migrates a weakly-typed
build flow to a strongly-typed equivalent, one path per entry point (Python network,
`trtexec`, C++ builder), plus the ModelOpt AutoCast pre-step needed when the source
is an ONNX file that must run in mixed precision.

**Version behavior.** On **TensorRT 11.x strong typing is unconditional** — every
network is strongly typed, the `kSTRONGLY_TYPED` creation flag is accepted but
ignored, and `trtexec --stronglyTyped` is a no-op (default-on). On **10.12–10.x**
you opt in explicitly via that flag / `--stronglyTyped`. Setting it is therefore
always safe: required on 10.x, harmless on 11.x. Either way the migration is the
same — remove the precision *hints* (gone in 11.x) and move precision into the graph.

## When to Use

| Scenario | Use this skill? |
|----------|-----------------|
| Existing Python build calls `config.set_flag(trt.BuilderFlag.FP16 / INT8 / TF32)` | Yes |
| Existing C++ build calls `config->setFlag(BuilderFlag::kFP16 / kINT8 / kTF32)` | Yes |
| Existing `trtexec` line uses `--fp16`, `--int8`, `--best`, `--bf16` and the developer wants strong typing | Yes |
| Build emits "weak typing is deprecated, use STRONGLY_TYPED" warning at runtime | Yes |
| Build fails with "flag X is incompatible with strongly typed networks" | Yes |
| Developer has a FP32 ONNX file and wants a strongly-typed mixed FP16 engine | Yes — start at Step 1 (AutoCast) |
| Developer is importing a brand-new ONNX file with no existing build flow | No — use `trt-onnx-quickstart` first; come back here to switch the flag |
| Developer's engine produces wrong outputs *after* migration | No — use `trt-quantization-accuracy` |
| Developer is on TensorRT 10.11 or earlier | No — strong typing is fully available from 10.12 onward; upgrade before migrating |

## Prerequisites

1. **TensorRT 10.12 or later (11.x recommended).** Check:
   ```bash
   python3 -c "import tensorrt; print(tensorrt.__version__)"
   trtexec --help 2>&1 | grep -i stronglyTyped
   ```
   The version check is the determinant — strong typing is available from 10.12
   on. The `trtexec` probe just confirms the CLI flag for the trtexec path; a
   missing `--stronglyTyped` means a pre-10.12 install.

2. **ModelOpt installed when the source is an ONNX file requiring mixed precision.**
   ```bash
   pip install nvidia-modelopt onnx onnxruntime
   ```
   If the ONNX file is already strongly typed (contains explicit `Cast` nodes
   and FP16 initializers), AutoCast is not needed — skip directly to Step 2.

3. **A working baseline.** Capture the weakly-typed engine's outputs on a known
   input set before migrating. Strong typing is stricter, and any silent precision
   substitutions weak typing was making will become observable.

4. **The verifier script.** `scripts/verify.sh` runs `migrate.py` against a
   committed sample and asserts the rewrite is well-formed. Run it before editing
   anything to confirm the helper works.

## Migration paths

There are three migration paths — Python builder, `trtexec`, C++ builder.
**Infer the path from the query when signals are clear; ask only if genuinely
uncertain.** Match the strongest signal:

| Signal in the request | Path | Entry point |
|------------------------|------|-------------|
| Python tokens: `trt.BuilderFlag`, `set_flag(`, `create_network(`, `.py` | **A** | Python `INetworkDefinition` builder |
| Standalone `trtexec --fp16` / `--int8` / `--best` / `--bf16` | **B** | `trtexec` CLI |
| C++ tokens: `BuilderFlag::`, `createNetworkV2`, `->setFlag`, `.cpp` / `.h` | **C** | C++ builder |
| Multiple signals | — | do each, most-specific first |
| No usable signal (generic "how do I migrate") | — | ask which entry point, or proceed with Python and say you assumed it |

The migration preserves existing precisions (kFP16 → FP16 cast, kINT8 → Q/DQ) and is
identical whether the network came from a parser or was hand-built — only Step 1
(AutoCast) is ONNX-specific, so source and precision goal rarely need a question.

Inferring the path is a low-stakes default — it does **not** extend to destructive
operations, which always need confirmation. `migrate.py` defaults to dry-run
(prints a diff); the `--write` flag is the human-in-the-loop confirmation. Treat
any other destructive step (overwriting a `.plan`, replacing an ONNX) the same way:
show, confirm, then act.

For a strongly-typed INT8 model straight from an ONNX file (not a migration of
existing builder code), jump to Step 1's ModelOpt quantization recipe.

## Step 1 — (ONNX source only) AutoCast the model to mixed precision

Skip if the source is not ONNX, or if the ONNX file is already strongly typed.

A FP32 ONNX file built for a weakly-typed `--fp16` flow carries no precision
information — TensorRT was free to pick FP16 or FP32 per layer. Strong typing
requires the graph itself to declare precision via `Cast` operators and typed
initializers. ModelOpt's AutoCast inserts these.

Use the ModelOpt CLI — it's the supported entry point and takes `.npz` calibration
directly. Use `--calibration_data` for both tools (autocast's canonical flag;
quantization's abbreviation of `--calibration_data_path`):

```bash
# FP16 / BF16 mixed-precision AutoCast
python -m modelopt.onnx.autocast \
    --onnx_path model.fp32.onnx \
    --output_path model.mixed.onnx \
    --low_precision_type fp16 \
    --calibration_data calib.npz

# INT8 / FP8 strongly-typed quantization (produces ONNX with explicit Q/DQ)
python -m modelopt.onnx.quantization \
    --onnx_path model.fp32.onnx \
    --output_path model.int8.onnx \
    --quantize_mode int8 \
    --high_precision_dtype fp16 \
    --calibration_data calib.npz
```

Keep accuracy-sensitive ops in FP32 via AutoCast's node/op-type exclusion lists
(e.g. exclude `MatMul`/`LayerNormalization` on transformer-heavy networks), and keep
FP32 graph I/O when downstream stages expect it. See the
[Model Optimizer docs](https://nvidia.github.io/Model-Optimizer) for the exact
module and flags for your ModelOpt version.

Verify the converted model in ONNX Runtime before passing it to TRT:

```python
import onnxruntime, numpy as np
sess = onnxruntime.InferenceSession("model.mixed.onnx", providers=["CPUExecutionProvider"])
# Compare a few outputs against the FP32 model with np.allclose(..., rtol=5e-3, atol=5e-3).
```

If parity fails here, **stop** — the conversion is wrong and no amount of TRT
engine work will recover it. Re-run with tighter exclusion lists.

**Next step — build the strongly-typed plan from the converted ONNX:**

```bash
# For the INT8 case (modelopt.onnx.quantization output)
trtexec --onnx=model.int8.onnx --stronglyTyped --saveEngine=model.int8.plan

# For the mixed-precision case (modelopt.onnx.autocast output)
trtexec --onnx=model.mixed.onnx --stronglyTyped --saveEngine=model.fp16.plan
```

**Critical**: do NOT pass `--int8` or `--fp16` alongside `--stronglyTyped`. Those flags are removed in TRT 11, and on TRT 10.12+ they conflict with strong typing. The precision is already encoded in the ONNX (via explicit Cast / Q / DQ nodes); the builder honors it.

The reference recipe for the end-to-end INT8 flow is `samples/python/strongly_type_autocast/sample.py` in `tensorrt-oss`.

## Step 2A — Migrate a Python `INetworkDefinition` builder

Mechanical replacements:

| Weakly typed | Strongly typed |
|--------------|----------------|
| `builder.create_network(0)` or `builder.create_network(1 << int(NetworkDefinitionCreationFlag.EXPLICIT_BATCH))` | `builder.create_network(1 << int(NetworkDefinitionCreationFlag.STRONGLY_TYPED))` |
| `config.set_flag(trt.BuilderFlag.FP16)` | **Remove.** Types come from the network. |
| `config.set_flag(trt.BuilderFlag.BF16)` | **Remove.** |
| `config.set_flag(trt.BuilderFlag.INT8)` | **Remove.** Quantization expressed via Q/DQ nodes in the graph. |
| `config.set_flag(trt.BuilderFlag.TF32)` | **Keep.** `kTF32` is retained in 11.x — it allows (not requires) TF32 for FP32 tensors and is orthogonal to typing. |
| `layer.precision = trt.float16` | **Remove.** Cast the input tensor instead (`network.add_cast`). |
| `layer.set_output_type(0, trt.float16)` | **Remove.** Output type follows the network; for cast/quantize/dequantize/fill layers use `set_to_type` instead. |
| `builder.platform_has_fast_int8 / has_fast_fp16` checks gating the above | **Remove the gating logic** along with the flag. |

Worked rewrite (the MNIST builder from `samples/python/network_api_pytorch_mnist/sample.py`
already uses the strongly-typed form — note `NetworkDefinitionCreationFlag.STRONGLY_TYPED`
and the absence of any `BuilderFlag.FP16` call):

```python
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))
# ... populate network ...
plan = builder.build_serialized_network(network, config)
```

If the network is built via the ONNX parser, the parser respects strong typing
automatically — no code change is needed beyond the flag.

Where a removed `precision` / `set_output_type` hint actually mattered, encode it
in the graph: cast a tensor's dtype with `add_cast`, or insert an explicit
Quantize/Dequantize pair for INT8 (the scale is a build-time constant; ModelOpt
derives it from calibration, or supply it by hand):

```python
# kFP16 hint -> explicit cast on the tensor:
cast = network.add_cast(some_tensor, trt.float16)

# kINT8 hint -> explicit Q/DQ pair (per-tensor scalar scale). NOTE: trt.Weights does
# not copy the numpy buffer, so q_scale must stay alive until the engine is built.
q_scale = np.array(1.0 / 127.0, dtype=np.float32).reshape(())          # rank-0 scalar
scale = network.add_constant(shape=(), weights=trt.Weights(q_scale)).get_output(0)
q  = network.add_quantize(some_tensor, scale, trt.int8).get_output(0)
dq = network.add_dequantize(q, scale, trt.float32).get_output(0)
# Feed dq into the consumer; TensorRT fuses the Q/DQ into it so it runs in INT8.
```

Drive the rewrite from `scripts/migrate.py`:

```bash
python3 scripts/migrate.py path/to/build.py            # dry-run diff
python3 scripts/migrate.py path/to/build.py --write    # rewrite in place
```

The script edits AST nodes (not regexes) — it understands `create_network` and
`set_flag` call shapes, so it is safe on files that mix conventions.

## Step 2B — Migrate a `trtexec` command line

Replacements:

| Weakly typed flag | Strongly typed equivalent |
|-------------------|---------------------------|
| `--fp16` | **Remove.** Precision comes from the ONNX. See the `--stronglyTyped` note below. |
| `--bf16` | **Remove.** Same as `--fp16`. |
| `--int8` | **Remove.** Q/DQ must be present in the ONNX. |
| `--best` | **Remove.** Not allowed with `--stronglyTyped` on 10.12–10.x (see Common Errors). |
| `--stronglyTyped` | **Version-gated.** Required on 10.12–10.x to opt into strong typing; a no-op (default-on) on 11.x — drop it or leave it, your choice. |
| `--noTF32` | **Keep.** `kTF32` is retained in 11.x; `--noTF32` still disables TF32 and is orthogonal to typing. |
| `--precisionConstraints=obey` / `prefer` | **Remove.** Constraints are expressed in the graph. |
| `--layerPrecisions=...` | **Remove.** Use ONNX `Cast` nodes / AutoCast. |
| `--layerOutputTypes=...` | **Remove.** |

Worked rewrite:

```bash
# Before (weakly typed)
trtexec --onnx=model.onnx --fp16 --saveEngine=model.fp16.plan \
        --minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 \
        --maxShapes=input:16x3x224x224

# After (strongly typed; ONNX has been AutoCast'd to mixed FP16)
# On 11.x --stronglyTyped is the default and a no-op — the line below is equivalent
# with or without it. On 10.12–10.x it is REQUIRED, or the build stays weakly typed.
trtexec --onnx=model.mixed.onnx --stronglyTyped --saveEngine=model.fp16.plan \
        --minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 \
        --maxShapes=input:16x3x224x224
```

The `trtexec` README's [Example 6](https://github.com/NVIDIA/TensorRT/blob/main/samples/trtexec/README.md#example-6-create-a-strongly-typed-plan-file)
is the canonical reference.

## Step 2C — Migrate a C++ builder

Replacements mirror the Python path:

| Weakly typed (C++) | Strongly typed (C++) |
|---|---|
| `builder->createNetworkV2(0)` | `builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED))` |
| `config->setFlag(BuilderFlag::kFP16)` | **Remove.** |
| `config->setFlag(BuilderFlag::kBF16)` | **Remove.** |
| `config->setFlag(BuilderFlag::kINT8)` | **Remove.** |
| `config->setFlag(BuilderFlag::kTF32)` | **Keep.** `kTF32` is retained in 11.x — orthogonal to typing. |
| `config->setFlag(BuilderFlag::kOBEY_PRECISION_CONSTRAINTS / kPREFER_PRECISION_CONSTRAINTS)` | **Remove.** No per-layer precision exists to constrain. |
| `layer->setPrecision(DataType::kHALF)` | **Remove.** Cast the input tensor instead (`addCast`). |
| `layer->setComputePrecision(...)` | **Remove.** Deprecated 10.16, removed; compute precision follows input dtypes. |
| `layer->setOutputType(0, DataType::kHALF)` | **Remove.** For cast/quantize/dequantize/fill layers use `setToType()` instead. |

Worked rewrite:

```cpp
// Before (10.x weakly typed) — the builder picks precision from these hints:
auto network = std::unique_ptr<INetworkDefinition>(
    builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
config->setFlag(BuilderFlag::kFP16);                        // hint: builder may use FP16
config->setFlag(BuilderFlag::kINT8);                        // hint: builder may use INT8 (+ a calibrator)
config->setFlag(BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);  // force the per-layer overrides below
someLayer->setPrecision(DataType::kHALF);                   // per-layer override
someLayer->setOutputType(0, DataType::kHALF);               // per-layer override
```

Each hint above maps to a concrete graph edit — there is no longer a "tell the
builder to prefer X" knob, so state the dtype directly:

```cpp
// After (11.x strongly typed) — precision lives in the graph; every line above is gone.
auto network = std::unique_ptr<INetworkDefinition>(
    builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
    // kSTRONGLY_TYPED: required on 10.12–10.x, accepted-but-ignored on 11.x (0 works too).
auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
// No precision setFlag, no setPrecision / setOutputType.

// kFP16 hint  ->  cast the tensor's dtype explicitly:
auto* cast = network->addCast(*someTensor, DataType::kHALF);
// feed cast->getOutput(0) into the consumer layer

// kINT8 hint  ->  insert an explicit Quantize/Dequantize pair. The scale is a
// build-time constant (per-tensor scalar here); ModelOpt derives it from calibration,
// or supply it by hand. NOTE: TensorRT does not copy Weights, so qScale must stay
// alive until the engine is built.
float qScale = 1.0f / 127.0f;
auto* scale = network->addConstant(Dims{0, {}},
                                   Weights{DataType::kFLOAT, &qScale, 1})->getOutput(0);
auto* q  = network->addQuantize(*someTensor, *scale, DataType::kINT8)->getOutput(0);
auto* dq = network->addDequantize(*q, *scale, DataType::kFLOAT)->getOutput(0);
// feed dq into the consumer; TensorRT fuses the Q/DQ into it so it runs in INT8.

// kOBEY_PRECISION_CONSTRAINTS + setPrecision/setOutputType  ->  nothing to port:
// the dtype is now fixed by the cast / Q-DQ above, so there is no constraint to obey.
```

`setToType()` is the only surviving per-layer type control, and is valid **only**
on `ICastLayer`, `IDequantizeLayer`, `IDynamicQuantizeLayer`, `IFillLayer`, and
`IQuantizeLayer` — not on arbitrary layers.

If the build pipeline parses ONNX via `nvonnxparser::IParser`, no parser change is
required — the parser honors the strong typing flag set on the network.

## Step 3 — Rebuild and validate

1. Rebuild the engine with the migrated flow.
2. Run the captured baseline inputs through the new engine.
3. Compare against the weakly-typed baseline:
   - **Outputs match within `atol=5e-3, rtol=5e-3`**: migration is complete.
   - **Outputs diverge but stay within model accuracy tolerance**: weak typing
     was opportunistically using FP16/INT8 where the new graph keeps FP32 (or vice
     versa). Acceptable if the downstream metric (top-1, BLEU, WER) is unchanged;
     confirm with the original accuracy harness.
   - **Outputs diverge beyond tolerance**: see Common Errors below.

If the source was ONNX, also run the AutoCast'd ONNX through ONNX Runtime and
compare against the FP32 ONNX before blaming TRT — graph-level conversion errors
should be caught there.

## Migration Patterns Reference

| Pattern | Action |
|---------|--------|
| Network created with no flags (legacy implicit batch) | First migrate to explicit batch; only then add `STRONGLY_TYPED` |
| Multiple `set_flag` calls (FP16 + INT8 + TF32 + REFIT) | Keep `REFIT`, `SPARSE_WEIGHTS`, `kTF32`; drop the precision-hint flags (FP16/BF16/INT8/FP8) and the removed `kOBEY`/`kPREFER_PRECISION_CONSTRAINTS` |
| `set_calibration_profile` / `IInt8Calibrator` | **Remove.** PTQ calibration is replaced by Q/DQ in the ONNX. Use ModelOpt's PTQ for the model first. |
| Mixed FP16/FP32 per-layer overrides via `layer.precision` | Express the same intent with `Cast` ops in the ONNX, or in the Python network construction code |
| Plugin layers with explicit precision | Plugins must implement `IPluginV3OneBuild::getOutputDataTypes`; the network reads types from the plugin |
| Engine consumed by downstream framework expecting FP32 I/O | Set `keep_io_types=True` in AutoCast, or add explicit `Cast` to FP32 at the network outputs |

## Common Errors

- **`"BuilderFlag::kFP16 is not compatible with a strongly typed network"`** —
  a `setFlag`/`set_flag` call survived the migration. Re-run `migrate.py` or grep
  for `BuilderFlag\.` / `BuilderFlag::` in the migrated source.
- **`"--best is incompatible with --stronglyTyped"`** — drop `--best` from the
  `trtexec` line; precision choices live in the graph now.
- **`"Network has no marked outputs"` immediately after switching the flag** —
  some builders only call `mark_output` on the path conditioned by
  `platform_has_fast_fp16`. Remove the conditional.
- **AutoCast'd ONNX fails the parser with "If subgraph type mismatch"** — AutoCast
  can corrupt certain `If` node subgraphs. Add the affected `If` node names to
  `nodes_to_exclude` so they stay FP32.
- **Engine output is FP32 zeros after migration** — `keep_io_types=False` was used
  but the host code still feeds FP32 buffers. Either set `keep_io_types=True` or
  cast the host input before binding.
- **Accuracy regression of ~1-3% after migration** — known case for
  transformer-heavy networks: leftover `Cast(to=FLOAT32)` from AutoCast combined
  with FP16 weights causes thrash. Work around by excluding the residual cast op
  types in `op_types_to_exclude`.

## Pitfalls

- **Do not partially migrate.** Half-migrated builds (some `set_flag` calls
  removed, `STRONGLY_TYPED` not set) silently fall back to weak typing and the
  warning is easy to miss. Check the engine build log for `"Strongly typed network
  detected"` to confirm the flag took effect.
- **Do not migrate without a baseline.** Weak typing's per-layer precision choices
  are non-deterministic across TRT versions; without the old outputs you have no
  ground truth and cannot tell whether the new engine is correct.
- **Do not run `--stronglyTyped` against a FP32 ONNX expecting FP16 speedup.**
  Strong typing builds exactly what the graph specifies — a FP32 ONNX produces a
  FP32 engine. Run AutoCast first.
- **Plugins must declare types.** Old plugins that left output types as "follow
  input" will fail to build under strong typing. Update them to the
  `IPluginV3OneBuild` interface, the only plugin API that supports strong typing
  in 11.x.
- **Refit + strong typing**: refittable strongly-typed engines work, but the refit
  weights must match the precision declared in the network. A FP16 layer cannot be
  refit with FP32 weights — convert host weights before calling `setNamedWeights`.
- **Don't trust grep alone for completeness.** `migrate.py` uses AST matching, so
  it catches the precision flags regardless of how `tensorrt` is imported or
  aliased via `as` — something string-level grep misses. It matches direct
  attribute access (`trt.BuilderFlag.FP16`); dynamic forms such as
  `getattr(trt.BuilderFlag, 'FP16')` are *not* rewritten and need manual review.
  Run the AST helper as the source of truth.

## References

- **Migration guide:** [TensorRT 10.x → 11.x](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x.html)
  — the authoritative description of the weak → strong typing change.
- **Operator type rules:** the [TensorRT Operators Reference](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/operators/)
  — each operator's page lists its supported input/output types, i.e. how its output
  dtype follows its inputs in a strongly-typed graph.
- **API reference:** the TensorRT C++/Python API docs for `NetworkDefinitionCreationFlag`,
  `BuilderFlag`, `ILayer::setToType`, and `INetworkDefinition::addCast` / `addQuantize` /
  `addDequantize`.
- **TensorRT Model Optimizer:** <https://nvidia.github.io/Model-Optimizer> — AutoCast and
  quantization tooling for getting precision into the model.
- **Sample code** (TensorRT OSS, <https://github.com/NVIDIA/TensorRT>):
  [`samples/trtexec/README.md` Example 6](https://github.com/NVIDIA/TensorRT/blob/main/samples/trtexec/README.md#example-6-create-a-strongly-typed-plan-file)
  (strongly-typed plan), `samples/python/strongly_type_autocast/sample.py`
  (AutoCast → strongly-typed INT8), and `samples/python/network_api_pytorch_mnist/sample.py`
  (strongly-typed Python builder).
