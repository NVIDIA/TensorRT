# RMSNorm PluginV3 with CuteDSL

This sample shows how to author a TensorRT `IPluginV3` whose `enqueue()` invokes a kernel written in [CuteDSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html), CUTLASS's Python DSL. The operator is **RMSNorm**, a normalization used in essentially every modern LLM (Llama, Mistral, Qwen, etc.).

New to TensorRT plugins? Start with [`non_zero_plugin`](../non_zero_plugin) for a minimal end-to-end PluginV3 walkthrough. This sample assumes that flow and focuses on the **CuteDSL ↔ TensorRT** handoff rather than the kernel itself. For CuteDSL specifics, see the [CUTLASS Python DSL overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html), the [`cutlass.cute` API reference](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html), and the [`cutlass/examples/python`](https://github.com/NVIDIA/cutlass/tree/main/examples/python) kernels.

## What the sample demonstrates

- Plugging a RMSNorm CuteDSL kernel into TensorRT through `IPluginV3` + `IPluginV3OneCore` + `IPluginV3OneBuild` + `IPluginV3OneRuntime`.
- Sharing GPU buffers with the kernel zero-copy via `cupy.cuda.UnownedMemory` → `torch.as_tensor` → `cute.runtime.from_dlpack`.
- Launching the kernel on the CUDA stream TRT passes into `enqueue()` (not the default stream), by declaring a `CUstream` parameter on the JIT-compiled launcher.
- Reducing in FP32 for numerical stability on large hidden dims, then casting back to FP16 on store.

## Requirements

```bash
cd $TRT_OSSPATH/samples/python/cute_dsl_plugin
pip3 install -r requirements.txt
```

CuteDSL requires Ampere (SM80) or newer.

## Running the sample

```bash
python3 rms_norm_plugin_cutedsl.py
```

The sample builds one engine with `num_tokens` declared dynamic over `[1, 512]` (opt=128) and `hidden_dim=1024`, then runs `num_tokens=128` through it. It prints `Inference result correct!` or `Inference result incorrect!`. Exit code is 0 on success.

## Plugin interface

- Inputs:
  - `X`: rank-2 tensor, shape `(num_tokens, hidden_dim)`, FP16. `num_tokens` is dynamic (set by an optimization profile at build time); `hidden_dim` is static.
  - `weight`: rank-1 tensor, shape `(hidden_dim,)`, FP32.
- Output:
  - `Y`: rank-2 tensor, shape `(num_tokens, hidden_dim)`, FP16.
- Attribute:
  - `epsilon`: scalar FP32. Required; the plugin asserts it is set.

## Kernel design

One CUDA block per token, 256 threads per block. Each block:

1. Loads `X[token]` and accumulates `sum(x * x)` in FP32 across its threads.
2. Reduces the partial sums through shared memory and a final warp shuffle.
3. Computes `rms = rsqrt(sum_sq / H + eps)` once per block.
4. Writes `Y[token, i] = X[token, i] * weight[i] * rms`.

The reduction is always done in FP32. The result is cast back to FP16 only when storing `Y`.

## How CuteDSL is wired into the plugin

The CuteDSL side is **two free functions** plus a launch helper. The TensorRT side is the usual `IPluginV3` class. They meet inside `enqueue()`. Concretely:

### 1. Author the kernel as a `@cute.kernel`

```python
@cute.kernel
def rms_norm_kernel(mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor,
                    threads_per_block: cutlass.Constexpr,
                    hidden_dim: cutlass.Constexpr,
                    epsilon: cutlass.Constexpr):
    ...
```

Tensor arguments are typed `cute.Tensor`. Values you want baked in at compile time are typed `cutlass.Constexpr` (here: block size, hidden dim, epsilon).

### 2. Wrap the launch in a `@cute.jit`

```python
from cuda.bindings.driver import CUstream

@cute.jit
def rms_norm_launch(mX, mW, mY,
                    num_tokens: cutlass.Int32,
                    hidden_dim: cutlass.Constexpr,
                    epsilon: cutlass.Constexpr,
                    stream: CUstream):
    rms_norm_kernel(mX, mW, mY, THREADS_PER_BLOCK, hidden_dim, epsilon).launch(
        grid=(num_tokens, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
    )
```

Two things to notice:

- `num_tokens` is typed `cutlass.Int32`, **not** `cutlass.Constexpr`. That means it is a runtime value that changes per call (it controls the grid dimension), so one compiled kernel handles every sequence length the optimization profile allows. `hidden_dim` stays `Constexpr` because the kernel's inner unroll trip count depends on it.
- The `stream: CUstream` parameter is what tells the CuteDSL runtime which CUDA stream to launch on. It does not need to be passed to `.launch()` directly. The runtime picks it up automatically from any argument typed as `CUstream`.

### 3. Subclass `IPluginV3` and the three capability interfaces

```python
class RmsNormPlugin(
    trt.IPluginV3,            # top-level plugin interface
    trt.IPluginV3OneCore,     # capability: core metadata
    trt.IPluginV3OneBuild,    # capability: shape/format/serialization at build time
    trt.IPluginV3OneRuntime,  # capability: enqueue and runtime hooks
):
```

`IPluginV3` is the top-level interface (it owns `get_capability_interface()`, `clone()`, `destroy()`). The three `IPluginV3One*` mixins are the **capabilities** TRT will ask for at build and runtime. Inheriting from all four lets a single Python object answer to every request.

Call **each** base `__init__` explicitly in your own `__init__`, and set the four book-keeping attributes that TensorRT reads: `plugin_namespace`, `plugin_name`, `plugin_version`, `num_outputs`. Set `timing_cache_id = ""` so TRT doesn't try to time per-instance.

### 4. Methods you must implement

These are the methods this sample overrides. Everything else falls through to defaults.

| Method | Capability | Purpose |
|---|---|---|
| `get_capability_interface(type)` | `IPluginV3` | Return `self`. TRT calls this with `type ∈ {CORE, BUILD, RUNTIME}`. Our class inherits all three capability mixins, so the same object serves any of them. |
| `get_output_data_types(input_types)` | `Build` | `Y` is FP16, matching `X`. |
| `get_output_shapes(inputs, shape_inputs, exprBuilder)` | `Build` | `Y` has the same shape as `X` (`trt.DimsExprs(inputs[0])`). |
| `get_fields_to_serialize()` | `Build` | Tells TRT which plugin attributes to save into the engine file. TRT calls this when building the engine, then hands the same fields back to the plugin creator at load time. Here we save `epsilon`. |
| `configure_plugin(inp, out)` | `Build` | No-op for this kernel. Override if your kernel needs to precompute something from the I/O descriptors. |
| `supports_format_combination(pos, in_out, num_inputs)` | `Build` | `X` and `Y` are FP16, `weight` is FP32, all `LINEAR`. |
| `on_shape_change(inp, out)` | `Runtime` | No-op here. Override if you need to invalidate caches when shapes change. |
| `enqueue(input_desc, output_desc, inputs, outputs, workspace, stream)` | `Runtime` | Where the kernel launch happens during inference. See below. |
| `attach_to_context(context)` | `Runtime` | Return `self.clone()` so each execution context owns its own kernel cache. |
| `set_tactic(tactic)` | `Runtime` | No-op since this plugin is single-tactic. |
| `clone()` | `IPluginV3` | Return a fresh `RmsNormPlugin` with `__dict__` copied. Clear the JIT cache on the clone. |
| `destroy()` | `IPluginV3` | Release (clear) the JIT cache when TRT is done with the plugin. |

### 5. The handoff in `enqueue()`

This is the only place CuteDSL touches TensorRT.

```python
def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):
    # Read runtime shape and dtype from the descriptors.
    num_tokens, hidden_dim = int(input_desc[0].dims[0]), int(input_desc[0].dims[1])

    # Step A: raw device ptr -> cupy ndarray (zero copy, just a view).
    x_cp = UnownedMemory(inputs[0], (num_tokens, hidden_dim), trt.nptype(input_desc[0].type)).d
    # Step B: cupy ndarray -> torch tensor (still zero copy, shares __cuda_array_interface__).
    x_t = torch.as_tensor(x_cp, device="cuda")
    # Step C: torch tensor -> CuteDSL tensor via dlpack.
    mX = from_dlpack(x_t, assumed_align=16)
    # (same three steps for `weight` and `Y`)

    # Step D: compile once per (hidden_dim, epsilon) and cache.
    key = (hidden_dim, self.epsilon)
    if key not in self._compiled:
        self._compiled[key] = cute.compile(
            rms_norm_launch, mX, mW, mY,
            num_tokens, hidden_dim, self.epsilon, make_fake_stream())

    # Step E: launch on TRT's stream.
    self._compiled[key](mX, mW, mY, num_tokens, CUstream(stream))
```

A few things to note:

- `cute.compile()` is the expensive step. The `self._compiled` dict makes sure it runs only once per `(hidden_dim, epsilon)` for the life of the plugin instance. The first call is effectively a warmup.
- The `stream` argument that TRT passes into `enqueue()` is the CUDA stream the engine wants the work scheduled on. Always launch on that stream, not the default one, or you break TRT's stream ordering and risk hangs or races when an engine is run concurrently with other CUDA work.

#### Why the cupy → torch → dlpack chain?

TRT gives the plugin **raw integer device pointers** (`inputs[0]`, `outputs[0]`, ...) along with shape and dtype descriptors. CuteDSL wants a `cute.Tensor`. There is no single API that converts the first to the second, so the sample bridges them through two protocol hops. Each hop is **zero-copy**: we never touch the bytes, we only re-type them.

| Hop | What it does | Why it's there |
|---|---|---|
| `cupy.cuda.UnownedMemory` | Wraps the raw integer pointer plus shape/dtype into a `cupy.ndarray`. | The only public Python API that wraps a **foreign** device pointer (one we did not allocate and must not free). Plain `torch.as_tensor` does not accept a raw `int` + shape + dtype. |
| `torch.as_tensor(cp_array, device="cuda")` | Reads CuPy's `__cuda_array_interface__` and returns a torch view of the same memory. | We need a producer that exposes `__dlpack__`, and the existing TRT plugin samples already use torch for this; CuteDSL's `from_dlpack` consumes torch's DLPack capsule directly. |
| `cute.runtime.from_dlpack(torch_tensor, assumed_align=16)` | Reads torch's `__dlpack__()` capsule and produces a `cute.Tensor`. | The published CuteDSL entry point for "take an external GPU buffer and use it as a `cute.Tensor`". |

In short: TRT speaks raw pointers (because the C++ plugin ABI is framework-agnostic), CuteDSL speaks `cute.Tensor`, and the cupy → torch → dlpack chain is the shortest path between them that avoids both a copy and ownership confusion.

### 6. Plugin creator

```python
class RmsNormPluginCreator(trt.IPluginCreatorV3One):
    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = "RmsNormPlugin"
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection([
            trt.PluginField("epsilon", np.array([], dtype=np.float32),
                            trt.PluginFieldType.FLOAT32),
        ])

    def create_plugin(self, name, fc, phase):
        return RmsNormPlugin(fc)
```

Nothing CuteDSL-specific here. The `field_names` list the attributes that `create_plugin()` expects to find in the `PluginFieldCollection`.

### 7. Building the network

`network.add_plugin_v3([X, W], [], plugin)`. The first list is the data inputs, the second is shape-tensor inputs (none here), the third is the plugin object returned by the creator. The rest of the engine build (registering the creator, calling `engine_from_network`, running with `TrtRunner`) is the same as for any other PluginV3.

## Limitations

- `hidden_dim` is static at engine build time (the CuteDSL kernel is JIT-compiled per `hidden_dim`). `num_tokens` is dynamic within the optimization profile's `[min, max]` range.
- The kernel requires `hidden_dim >= 256` (the same as `THREADS_PER_BLOCK`), which is checked in `build_engine()`.
- As with all Python plugins, the engine cannot be deserialized outside a Python interpreter that has the plugin classes available.

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

## Changelog

May 2026: Initial release.
