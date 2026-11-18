# TensorRT Safety Runtime Command-Line Wrapper: trtexec_safe

**Table Of Contents**
- [Description](#description)
- [Building `trtexec_safe`](#building-trtexec_safe)
- [Using `trtexec_safe`](#using-trtexec_safe)
- [Tool command line arguments](#tool-command-line-arguments)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

`trtexec_safe` is a performance testing tool for TensorRT safety runtime. `trtexec_safe` can load a serialized engine file and test the performance of running inference with TensorRT safety runtime. `trtexec_safe` provides many options, like iterations, average runs, streams, for performance timing.

**Note:** `trtexec_safe` only provides the functionality of doing inference with a pre-built engine file.  The engine should be built with `EngineCapability::kSAFETY`. `trtexec_safe` is not safety-certified, therefore, it could produce AUTOSAR violations in the source code.

## Building `trtexec_safe`

Compile the sample by following the build instructions in the [TensorRT README](https://github.com/NVIDIA/TensorRT/). This will build the `trtexec_safe` binary.

The options that control compiling against safety runtime or standard runtime are summarized in the following table.

| Compile Option                  | Default |Description|
| ------------------------------- | ------- |---------- |
|TRT_SAFETY_INFERENCE_ONLY        | OFF     |When enabled, build the infer part only, skip compiling the builder part.|

## Using `trtexec_safe`

1. `trtexec_safe` requires a prebuilt safe TensorRT engine as input. Use `trtexec` to build a safe TensorRT engine.
    ```
    trtexec --safe --saveEngine=/path/to/safe/engine --model=/path/to/model
    ```
  
    where `--safe` indicates `trtexec` to build a safety certified TensorRT engine and `--saveEngine` specifies the path to save the output engine. For other details of `trtexec`, see `<TensorRT root directory>/samples/trtexec/README.md`.


2. Use `trtexec_safe` to load the engine and run performance testing.

    ```
    trtexec_safe --loadEngine=/path/to/safe/engine
    ```
    `trtexec_safe` will do inference for several iterations and give a final median execution time on both the CPU and GPU. For other options, refer to the [Tool command line arguments](#tool-command-line-arguments) section.

## Tool command line arguments

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.

```
trtexec_safe [-h or --help]
```

`--debug` controls debug logging independently and does not enable memory tracking.
On Linux, `--reportMemoryUsage` automatically relaunches the tool under Compute
Sanitizer memcheck with full leak checking and an error exit code of 99. Install
`compute-sanitizer` on `PATH`; a missing sanitizer fails the command. Do not wrap
this flag in a second Compute Sanitizer invocation. Sanitizer instrumentation
affects timing and memory usage; use runs without this flag for performance measurements.
On QNX, the same flag uses the internal allocation tracker without launching
Compute Sanitizer. This does not change CI platform selection.
Pass `--reportMemoryUsage` to report requested allocation peaks separately for normal inference
and the tensor-profiling run. Placement breakdowns distinguish GPU, CPU, and pinned
host memory; purpose breakdowns distinguish immutable/weights, scratch, I/O, and
other allocations. Each category is an independent high-water mark: category peaks
can occur at different times and must not be summed to reconstruct the total peak.

Lifecycle snapshots show live bytes before graph creation, after creation/cloning,
after scratch setup, after inference and I/O cleanup, and after graph/scratch cleanup.
The callback tracker also captures transient allocation peaks between snapshots.
Graph and external scratch cleanup runs on error paths as well as successful runs.

To look for allocations retained across repeated graph lifetimes in the same process:

```bash
./trtexec_safe --loadEngine=model.mvmf --reportMemoryUsage --memoryCycles=5
```

`--memoryCycles` repeats the normal create/infer/destroy cycle (default 1, maximum
1000, requires `--reportMemoryUsage`); the existing tensor-profiling run follows once. `--iterations`
and `--avgRuns` still control inference execution within each graph lifetime.
The cleanup check identifies allocations by generation, so releasing an older
allocation and retaining a new allocation of the same size cannot hide a residual.
A newly retained tracked allocation reports a possible leak and fails the tool;
incomplete bookkeeping reports an inconclusive check. Previously live allocations
are excluded from that cycle's check. Independent per-category outstanding bytes
and counts help locate retained memory. Inference/cleanup failures return a nonzero
exit status and stop further cycles.

Tracking covers allocations through the supplied tool allocator and explicit
`--useScratch` buffers. It excludes CUDA driver/context memory, allocator overhead,
engine-file buffers, runtime host allocations outside that interface, and
plugin-private allocations that bypass it. Zero tracked residual does not prove
absence of all leaks. A persistent cache or driver reservation can also retain
process/system memory without leaving a tracked allocation. QNX system-pool
sampling and representative-network comparisons are separate measurements; see
`tools/memory_usage_safe` and TRTS-11759. Debug logging and tensor profiling can
affect timing and memory usage. No public runtime API is changed.

On Linux, the command below runs both the internal tracker and Compute Sanitizer:

```bash
./trtexec_safe --loadEngine=model.mvmf --reportMemoryUsage --memoryCycles=5 --noCudaGraph
```

See [Compute Sanitizer leak checking](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#leak-checking).
This checks CUDA device allocations; it does not replace host-memory or QNX
system-pool analysis.

## Additional resources

The following resources provide more details about `trtexec_safe`:

**Documentation**
- [TensorRT trtexec](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html#trtexec)
- [TensorRT Sample Support Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/sample-support-guide.html)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html)
documentation.

# Changelog

June 2020
This is the first release of this `README.md` file.

September 2020
This sample was updated to fit TensorRt API changes:
IRuntime::validateSerializedEngine is removed.

March 2022
This sample was updated for DriveOS 6.0 and later releases.

Dec. 2025
This sample was updated to use the CMake-based build system.

# Known issues

There are no known issues in this sample.

