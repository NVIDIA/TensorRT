---
name: trt-cpp-runtime-quickstart
description: >-
  Load and run a TensorRT engine (.plan / .engine) from C++ using the
  TensorRT 11 / 10.x **modern Runtime API**, avoiding the deprecated TRT
  8.x binding-index APIs that older guidance still promotes. Use whenever
  the user asks about loading
  or running a TensorRT .plan/.engine from C++, even on "minimal example"
  requests — without this skill the default reply uses deprecated
  enqueueV2-style code. Also use when the user hits "Engine plan file
  is generated on an incompatible device", deserializeCudaEngine returns
  nullptr, gets an enqueueV2 / IStreamReader deprecation warning, or
  wants to stream a .plan via IStreamReaderV2. Triggers: TensorRT C++
  inference, load TensorRT plan C++, run .plan from C++, IRuntime
  example, deserializeCudaEngine, enqueueV3, enqueueV2 deprecated,
  setTensorAddress, getBindingIndex, IStreamReaderV2, libnvinfer C++.
  NOT for building engines (`trt-onnx-quickstart`), Python deploy,
  plugins, multi-GPU.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
  version: "1.0"
  tags:
    - tensorrt
    - cpp
    - inference
    - deployment
    - runtime
---

# TensorRT C++ Runtime Deploy

Load a serialized TensorRT engine from disk and run inference from C++ using only the modern Runtime API. Produces a minimal, copy-pasteable deploy harness that drops next to any `.plan` / `.engine` file and extends to production.

Reference samples to open before writing new code:

- `quickstart/SemanticSegmentation/tutorial-runtime.cpp` — cleanest minimal load-and-run example. Mirrors Steps 1–7 below.
- `samples/sampleOnnxMNIST/sampleOnnxMNIST.cpp` — end-to-end sample that also builds the engine; the runtime portion shows realistic I/O wiring.
- Public headers: `include/NvInferRuntime.h` — read `IRuntime`, `ICudaEngine`, `IExecutionContext`, `IStreamReaderV2`.

## When to Use

| Situation                                                                                 | Use this skill? |
|-------------------------------------------------------------------------------------------|-----------------|
| You have a `.plan`/`.engine` and need to run it from a C++ binary                         | Yes             |
| You need a minimal harness that uses `enqueueV3` + `setTensorAddress`                     | Yes             |
| You want to load an engine from a `std::istream` or large file via `IStreamReaderV2`      | Yes             |
| You need to wire dynamic shapes (`setInputShape`) before inference                        | Yes             |
| You are *building* / optimizing the engine (calibration, INT8, sparsity, builder configs) | No - use trtexec or `IBuilder` directly |
| You are deploying in Python                                                               | No - use `tensorrt` Python bindings     |
| You are writing a plugin (`IPluginV3`) or custom layer                                    | No - separate plugin skill              |
| You need multi-GPU, MPS, MIG, or process-level orchestration                              | No - out of scope                       |

## Prerequisites

1. **TensorRT installed.** Verify `NvInferRuntime.h` is on the include path
   and `libnvinfer.so` is on the link path. On a TRT dev container these are
   in `/usr/include/x86_64-linux-gnu/` and `/usr/lib/x86_64-linux-gnu/` (or
   `/opt/tensorrt/...` for tarball installs).
2. **CUDA toolkit available.** `cuda_runtime_api.h` and `libcudart.so` must
   be reachable; `nvcc --version` should match the CUDA version the engine
   was built against.
3. **A serialized engine.** A `.plan`/`.engine` file built **on the same
   major TRT version and the same GPU architecture (compute capability) you
   will deploy on**. Engines are not portable across major TRT versions or
   across SMs unless the builder was given `--hardwareCompatibilityLevel`.
4. **The engine's I/O tensor names.** Inspect with:
   ```bash
   trtexec --loadEngine=model.plan --verbose 2>&1 | grep -E 'Input|Output'
   ```
5. A C++17 compiler (`g++ >= 9` or `clang++ >= 10`).

## Step 1: Create the IRuntime

The runtime owns engine deserialization and must outlive every
`ICudaEngine` it creates. Construct one per process for typical deployments.

```cpp
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, char const* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << msg << std::endl;
        }
    }
};

Logger gLogger;
std::unique_ptr<nvinfer1::IRuntime> runtime{
    nvinfer1::createInferRuntime(gLogger)};
if (!runtime) throw std::runtime_error("createInferRuntime failed");
```

A custom logger is mandatory - TensorRT does not log internally. Keep it
process-global so deserialization warnings (version skew, calibrator
mismatch) are not lost.

## Step 2: Read the Plan into Memory

For small/medium engines (< ~1 GiB) read the whole file into a
`std::vector<char>` and hand the pointer to
`IRuntime::deserializeCudaEngine(blob, size)`. This is what the
`SemanticSegmentation` tutorial does and the simplest correct path:

```cpp
std::ifstream f(planPath, std::ios::binary);
if (!f) throw std::runtime_error("cannot open " + planPath);
f.seekg(0, std::ios::end);
auto size = static_cast<size_t>(f.tellg());
f.seekg(0, std::ios::beg);
std::vector<char> blob(size);
if (!f.read(blob.data(), size))
    throw std::runtime_error("short read on " + planPath);
```

For very large engines, or when the bytes live behind a stream (HTTP,
mmap'd archive, encrypted store), implement an `IStreamReaderV2` - see
Step 3.

## Step 3 (optional): Use IStreamReaderV2 for Streaming Loads

`IStreamReader` (v1) is **deprecated in TensorRT 11.0**. Always use
`IStreamReaderV2`: it reads into both host and device memory and is the
only stream-reader form guaranteed for new code. Subclass and implement
`read(...)` and `seek(...)`:

```cpp
class FileStreamReader : public nvinfer1::IStreamReaderV2 {
public:
    explicit FileStreamReader(std::string const& path)
        : mFile(path, std::ios::binary) {
        if (!mFile) throw std::runtime_error("open failed: " + path);
    }
    int64_t read(void* dst, int64_t n,
                 cudaStream_t /*stream*/) noexcept override {
        mFile.read(static_cast<char*>(dst), n);
        return mFile.gcount();
    }
    bool seek(int64_t off, nvinfer1::SeekPosition where) noexcept override {
        auto dir = (where == nvinfer1::SeekPosition::kSET) ? std::ios::beg
                 : (where == nvinfer1::SeekPosition::kCUR) ? std::ios::cur
                 : std::ios::end;
        mFile.clear();
        mFile.seekg(off, dir);
        return static_cast<bool>(mFile);
    }
private:
    std::ifstream mFile;
};

FileStreamReader rd{planPath};
std::unique_ptr<nvinfer1::ICudaEngine> engine{
    runtime->deserializeCudaEngine(rd)};
```

## Step 4: Deserialize and Create an Execution Context

`ICudaEngine` is thread-safe for read-only queries; `IExecutionContext`
is **not** - allocate one per inference thread.

```cpp
std::unique_ptr<nvinfer1::ICudaEngine> engine{
    runtime->deserializeCudaEngine(blob.data(), blob.size())};
if (!engine) throw std::runtime_error("deserializeCudaEngine failed");

std::unique_ptr<nvinfer1::IExecutionContext> ctx{
    engine->createExecutionContext()};
if (!ctx) throw std::runtime_error("createExecutionContext failed");
```

## Step 5: Wire Tensors with setTensorAddress

Enumerate I/O tensors via `getNbIOTensors()` + `getIOTensorName(i)`. Use
`getTensorIOMode`, `getTensorDataType`, and `getTensorShape` to size and
allocate buffers. **Set every tensor address before `enqueueV3`** - the
modern API has no implicit binding-index map.

```cpp
for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    char const* name = engine->getIOTensorName(i);
    auto mode  = engine->getTensorIOMode(name);
    auto shape = engine->getTensorShape(name);   // -1 = dynamic dim
    if (mode == nvinfer1::TensorIOMode::kINPUT && hasDynamic(shape)) {
        // Fill in concrete shape, e.g. batch=1
        shape.d[0] = 1;
        ctx->setInputShape(name, shape);
    }
}
// After setInputShape on all dynamic inputs, query output shapes.
for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    char const* name = engine->getIOTensorName(i);
    auto bytes = elementCount(ctx->getTensorShape(name))
               * dtypeSize(engine->getTensorDataType(name));
    void* dev = nullptr;
    cudaMalloc(&dev, bytes);
    ctx->setTensorAddress(name, dev);
}
```

Always call `setInputShape` for dynamic inputs **before** querying output
shapes - the latter depends on the former.

## Step 6: Run enqueueV3

`enqueueV3(stream)` is the only non-deprecated enqueue API;
`enqueueV2`/`execute*` are gone in modern flows.

```cpp
cudaStream_t stream{};
cudaStreamCreate(&stream);

cudaMemcpyAsync(devInput, hostInput, inBytes,
                cudaMemcpyHostToDevice, stream);
if (!ctx->enqueueV3(stream))
    throw std::runtime_error("enqueueV3 failed");
cudaMemcpyAsync(hostOutput, devOutput, outBytes,
                cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
```

If you reuse buffers across iterations, skip the per-call
`setTensorAddress` - addresses persist on the context until overwritten.

## Step 7: Shutdown Order

Destroy in reverse construction order: contexts -> engines -> runtime,
then free CUDA memory and destroy the stream. With `std::unique_ptr` this
is automatic as long as the context is declared *after* the engine, and
the engine *after* the runtime. Free `cudaMalloc` allocations explicitly
(RAII wrapper recommended).

## Build

Wire the steps above into your application's build system. For a standalone smoke test, a minimal build is:

```bash
g++ -std=c++17 runtime.cpp -o run -lnvinfer -lcudart   # adjust CUDA/TRT include + lib paths
./run model.plan
```

## Common Errors

| Symptom                                                              | Likely cause                                                                 |
|----------------------------------------------------------------------|------------------------------------------------------------------------------|
| `deserializeCudaEngine` returns `nullptr`, log says "version tag"    | Engine built on a different TRT major version. Rebuild on the deploy version |
| `nullptr` with "engine plan file is generated on an incompatible device" | SM mismatch. Rebuild on the target SM or use `--hardwareCompatibilityLevel` |
| `enqueueV3` returns false, log mentions "Tensor X has no address"    | Forgot `setTensorAddress` for one of the I/O tensors                         |
| `enqueueV3` false, "shape" in message                                | Forgot `setInputShape` for a dynamic input, or supplied an out-of-profile shape |
| `cudaErrorIllegalAddress` on H->D / D->H copy                        | Mismatched element count / dtype between host buffer and engine tensor       |
| Process crashes inside TRT during destruction                        | Wrong destruction order - context outlived engine, or engine outlived runtime |
| `cudaErrorMemoryAllocation` during context creation                  | Workspace too big for the device; rebuild with smaller workspace             |

## Pitfalls

- **Do not use `IStreamReader` v1.** Deprecated in TRT 11.0. Use
  `IStreamReaderV2` (note `cudaStream_t` parameter on `read`).
- **Do not use `enqueueV2` / `execute` / binding indices.** These are
  legacy paths; the only stable modern path is name-based
  `setTensorAddress` + `enqueueV3`.
- **One `IExecutionContext` per thread.** Sharing contexts across threads
  is undefined behavior; sharing the engine is fine.
- **Stream lifetime.** The CUDA stream passed to `enqueueV3` must outlive
  the inference. Destroying it while work is in flight crashes or corrupts
  output.
- **Async vs sync copies.** Mixing synchronous `cudaMemcpy` with
  `enqueueV3` on a stream serializes the GPU; always pair `enqueueV3`
  with `cudaMemcpyAsync` on the same stream.
- **Engine portability.** A `.plan` is tied to (TRT major version, GPU SM,
  CUDA major version). Never check engines into a repo without recording
  these three facts.
- **Logger lifetime.** The logger passed to `createInferRuntime` must
  outlive the runtime; a stack-local logger in `main` is fine, a function-
  scope local is a use-after-free.
- **Refit / weight streaming.** Engines built with refit or weight
  streaming enabled need extra setup calls (`setWeightStreamingBudgetV2`,
  `IRefitter`); out of scope here.
