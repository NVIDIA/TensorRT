# TensorRT Command-Line Wrapper: trtexec

**Table Of Contents**
- [TensorRT Command-Line Wrapper: trtexec](#tensorrt-command-line-wrapper-trtexec)
  - [Description](#description)
  - [Building `trtexec`](#building-trtexec)
  - [Using `trtexec`](#using-trtexec)
    - [Example 1: Profiling a custom layer](#example-1-profiling-a-custom-layer)
    - [Example 2: Running a network on DLA](#example-2-running-a-network-on-dla)
    - [Example 3: Running an ONNX model with full dimensions and dynamic shapes](#example-3-running-an-onnx-model-with-full-dimensions-and-dynamic-shapes)
    - [Example 4: Collecting and printing a timing trace](#example-4-collecting-and-printing-a-timing-trace)
    - [Example 5: Tune throughput with multi-streaming](#example-5-tune-throughput-with-multi-streaming)
    - [Example 6: Create a strongly typed plan file](#example-6-create-a-strongly-typed-plan-file)
    - [Example 7: Global performance tuner](#example-7-global-performance-tuner)
      - [7.1: Discovering tunable knobs](#71-discovering-tunable-knobs)
      - [7.2: Building one specific configuration](#72-building-one-specific-configuration)
      - [7.3: Sweeping a configuration space](#73-sweeping-a-configuration-space)
      - [7.4: Choosing a search algorithm](#74-choosing-a-search-algorithm)
      - [7.5: Accuracy-aware tuning](#75-accuracy-aware-tuning)
      - [7.6: The tuning cache (and resuming)](#76-the-tuning-cache-and-resuming)
      - [7.7: Other useful flags](#77-other-useful-flags)
      - [7.8: Caveats of tuning](#78-caveats-of-tuning)
  - [Tool command line arguments](#tool-command-line-arguments)
  - [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

Included in the `samples` directory is a command line wrapper tool, called `trtexec`. `trtexec` is a tool to quickly utilize TensorRT without having to develop your own application. The `trtexec` tool has two main purposes:
-   It’s useful for benchmarking networks on random or user-provided input data.
-   It’s useful for generating serialized engines from models.

**Benchmarking network** - If you have a model saved as an ONNX file, you can use the `trtexec` tool to test the performance of running inference on your network using TensorRT. The `trtexec` tool has many options for specifying inputs and outputs, iterations for performance timing, precision allowed, and other options.

**Serialized engine generation** - If you generate a saved serialized engine file, you can pull it into another application that runs inference. For example, you can use the [TensorRT Laboratory](https://github.com/NVIDIA/tensorrt-laboratory) to run the engine with multiple execution contexts from multiple threads in a fully pipelined asynchronous way to test parallel inference performance. Also, in INT8 mode, random weights are used.

**Using custom input data** - By default trtexec will run inference with randomly generated inputs. To provide custom inputs for an inference run, trtexec expects a binary file containing the data for each input tensor. It is recommended that this binary file be generated through `numpy`. For example, to create custom data of all ones to an ONNX model with one input named `data` with shape `(1,3,244,244)` and type `FLOAT`:

```
import numpy as np
data = np.ones((1,3,244,244), dtype=np.float32)
data.tofile("data.bin")
```

This binary file can be be loaded by trtexec during inference by using the `--loadInputs` flag:

```
./trtexec --onnx=model.onnx --loadInputs=data:data.bin
```

The name of the input can be optionally wrapped in single quotes to support absolute paths on Windows:

```
.\trtexec.exe --onnx=model.onnx --loadInputs='data':C:\Users\TRT\data.bin
```

## Building `trtexec`

`trtexec` can be used to build engines, using different TensorRT features (see command line arguments), and run inference. `trtexec` also measures and reports execution time and can be used to understand performance and possibly locate bottlenecks.

Compile the sample by following build instructions in [TensorRT README](https://github.com/NVIDIA/TensorRT/).

## Using `trtexec`

`trtexec` can build engines from models in ONNX format.

### Example 1: Profiling a custom layer

You can profile a custom layer, implemented as a [TensorRT plugin](https://github.com/NVIDIA/TensorRT/tree/main/plugin#tensorrt-plugins), by leveraging `trtexec`. Plugins need to be registered in the plugin registry (instance of `IPluginRegistry`) to be visible to TensorRT. `trtexec` will load the TensorRT standard plugin library (`libnvinfer_plugin.so` / `nvinfer_plugin.dll`) that provides plugin support to TensorRT. Checkout the [Non-Zero Plugins Sample](../sampleNonZeroPlugin/) for a quick sample, or the [Plugins section](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending) of the TensorRT Developer Guide for a more detailed walkthrough.

Plugins can be used with `trtexec` in the following 2 ways:

<details>
<summary> Using TensorRT-shipped Plugins </summary>


- If you are using TensorRT-shipped plugins (included in `libnvinfer_plugin.so` / `nvinfer_plugin.dll`), no extra steps are required from the user as these plugins are pre-registered with the plugin registry.
</details>

<details>
<summary> Using your own Plugin  </summary>

  - If you want to define your own plugin and have `trtexec` use it as part of the network, you should define your own _Plugin Shared library_ with specific entry-points recognized by TensorRT. Then, provide the shared plugin library path to `trtexec` using the `--dynamicPlugins` flag.
  - More information on Plugin Shared Libraries and how to define them can be seen in the [Plugin Shared Libraries](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#plugin-serialization) section of the [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html).

    In summary, there are two methods:
    1. The `REGISTER_TENSORRT_PLUGIN` macro can be applied to the plugin creator for each plugin that needs to be statically registered. i.e. Registered at load-time of the plugin library.
    2. For dynamic registration, the plugin shared library must expose the below symbols which will be the entry points for TensorRT:

        ```cpp
        extern "C" void setLoggerFinder(ILoggerFinder* finder);
        extern "C" IPluginCreatorInterface* const* getCreators(int32_t& nbCreators)
        ```
    In the above, `setLoggerFinder()` should accept a pointer to an `ILoggerFinder`, through which an `ILogger` instance can be retrieved for the purpose of logging inside the library code. `getCreators()` should return an array of plugin creators the library contains. Example implementations of these entry points can be found in [plugin/vc/vfcCommon.cpp](../../plugin/vc/vfcCommon.cpp) and [plugin/vc/vfcCommon.h](../../plugin/vc/vfcCommon.h).

      **Note**: Usage of `getPluginCreators` instead of `getCreators` is also valid, but deprecated.
  - If the user wants to build a TensorRT engine first and run later, the user has the option to serialize the shared plugin library as part of the engine itself by specifying `--setPluginsToSerialize`. By doing so, the user does not have to specify `--dynamicPlugins` to `trtexec` when running the built engine.
  - For more information on these flags, run `./trtexec --help`.
</details>

### Example 2: Running a network on DLA

To run the MNIST network on NVIDIA DLA (Deep Learning Accelerator) using `trtexec` in FP16 mode, issue:
```
./trtexec --onnx=data/mnist/mnist.onnx --useDLACore=1 --fp16 --allowGPUFallback
```
To run the MNIST network on DLA using `trtexec` in INT8 mode, issue:
```
./trtexec --onnx=data/mnist/mnist.onnx --useDLACore=1 --int8 --allowGPUFallback
```
To run the MNIST network on DLA using `trtexec`, issue:
```
./trtexec --onnx=data/mnist/mnist.onnx --useDLACore=0 --fp16 --allowGPUFallback
```

For more information about DLA, see [Working With DLA](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#dla_topic).

### Example 3: Running an ONNX model with full dimensions and dynamic shapes

To run an ONNX model in full-dimensions mode with static input shapes:

```
./trtexec --onnx=model.onnx
```

The following examples assumes an ONNX model with one dynamic input with name `input` and dimensions `[-1, 3, 244, 244]`

To run an ONNX model in full-dimensions mode with an given input shape:

```
./trtexec --onnx=model.onnx --shapes=input:32x3x244x244
```

To benchmark your ONNX model with a range of possible input shapes:

```
./trtexec --onnx=model.onnx --minShapes=input:1x3x244x244 --optShapes=input:16x3x244x244 --maxShapes=input:32x3x244x244 --shapes=input:5x3x244x244
```

### Example 4: Collecting and printing a timing trace

When running, `trtexec` prints the measured performance, but can also export the measurement trace to a json file:
```
./trtexec --onnx=data/mnist/mnist.onnx --exportTimes=trace.json
```
Once the trace is stored in a file, it can be printed using the `tracer.py` utility. This tool prints timestamps and duration of input, compute, and output, in different forms:
```
./tracer.py trace.json
```
Similarly, profiles can also be printed and stored in a json file. The utility `profiler.py` can be used to read and print the profile from a json file.

### Example 5: Tune throughput with multi-streaming

Tuning throughput may require running multiple concurrent streams of execution. This is the case for example when the latency achieved is well within the desired
threshold, and we can increase the throughput, even at the expense of some latency. For example, saving engines with different precisions and assume that both
execute within 2ms, the latency threshold:
```
trtexec --onnx=resnet50.onnx --saveEngine=g1.trt --int8 --skipInference
trtexec --onnx=resnet50.onnx --saveEngine=g2.trt --best --skipInference
```
Now, the saved engines can be tried to find the combination precision/streams below 2 ms that maximizes the throughput:
```
trtexec --loadEngine=g1.trt --streams=2
trtexec --loadEngine=g1.trt --streams=3
trtexec --loadEngine=g1.trt --streams=4
trtexec --loadEngine=g2.trt --streams=2
```

### Example 6: Create a strongly typed plan file
This flag will create a network with the `NetworkDefinitionCreationFlag::kSTRONGLY_TYPED` flag where tensor data types are inferred from network input types
and operator type specification.  Use of specific builder precision flags such as `--int8` or `--best` with this option is not allowed.
```
./trtexec --onnx=model.onnx --stronglyTyped
```

### Example 7: Global performance tuner

TensorRT exposes a number of internal builder knobs — heuristics, layer
selections, codegen toggles — that change how an engine is built. A specific
combination of values for these knobs is called a **build route**. Different
routes produce engines with different performance and (occasionally) different
numerical accuracy on the same model.

`trtexec` can drive a tuning loop that sweeps a set of build routes,
benchmarks each, optionally validates accuracy against reference outputs, and
records the best configuration. Any single iteration in a sweep is fully
reproducible by re-running `trtexec` with `--setBuildRoute=<route>` and the
same model — see [7.2](#72-building-one-specific-configuration).

Tuning is not supported on Windows.

This functionality is exposed through a small group of related flags. The
sub-sections below cover each in turn.

#### 7.1: Discovering tunable knobs

`--helpBuildRoute` prints the knob database as JSON. Each entry lists the
knob name, the allowed values, and a default. This is the source of truth
for what you can put inside a `--setBuildRoute` or `--tuneBuildRoutes`
expression.

```
./trtexec --helpBuildRoute
```

Filter to a single knob — the leading dash is optional:

```
./trtexec --helpBuildRoute=match_ragged_mha
./trtexec --helpBuildRoute=-match_ragged_mha
```

If the named knob does not exist, `trtexec` exits with a non-zero status
and a message pointing to the unfiltered listing. `--helpBuildRoute` does
not require `--onnx` and ignores other build flags; `--help` takes
precedence when both are passed.

#### 7.2: Building one specific configuration

`--setBuildRoute=<route>` builds a single engine with a chosen route. The
route is a space-separated list of `-knob=value` tokens (note the leading
dash on each knob):

```
./trtexec --onnx=model.onnx \
          --setBuildRoute="-match_ragged_mha=on -copy_ppg=off" \
          --saveEngine=model.plan
```

This is the easiest way to reproduce or debug a specific result from a
tuning sweep: take the route reported in the sweep's log for an iteration
of interest, plug it into `--setBuildRoute`, and run again.

#### 7.3: Sweeping a configuration space

`--tuneBuildRoutes=<expr>` runs an autotuning loop over an expression that
describes a set of routes. The expression uses two forms:

- `-knob=[a|b|c]` — variable knob; the loop iterates over each listed value.
- `-knob=fixed`   — fixed value pinned across every iteration.

Multiple tokens are space-separated. The expression is typically quoted to
protect the brackets from the shell. `--saveEngine=<path>` is optional —
without it the sweep still benchmarks every route but no engine is written
to disk. `--saveEngine` becomes required when `--loadRefOutputs` is also
set (see [7.5](#75-accuracy-aware-tuning)).

```
./trtexec --onnx=model.onnx \
          --tuneBuildRoutes="-match_ragged_mha=[on|off] -copy_ppg=[on|off]" \
          --saveEngine=best.plan
```

For long expressions, put them in a file (one token per line) and pass the
file with `--tuneBuildRouteFile`:

```
$ cat routes.txt
-match_ragged_mha=[on|off]
-copy_ppg=[on|off]

$ ./trtexec --onnx=model.onnx --tuneBuildRouteFile=routes.txt --saveEngine=best.plan
```

`--tuneBuildRoutes` and `--tuneBuildRouteFile` are mutually exclusive.

#### 7.4: Choosing a search algorithm

`--tuningSearch=<spec>` controls how the expression is expanded into the
list of routes that the loop will try:

| Value   | Behavior |
|---------|----------|
| `fast`  | (default) baseline run with each knob at its default, plus one-off variations that change one knob at a time. Linear in the number of variable knobs. |
| `full`  | Cartesian product over every variable knob. Exponential — use for small expressions. |
| `mixed` | A `fast` scan first to identify which knob values improve performance, then a full sweep over only those "positive" knobs. A pragmatic middle ground for larger spaces. |

To make the difference concrete, take an expression with three binary
knobs `-A=[on|off] -B=[on|off] -C=[on|off]` and defaults `A=on, B=on, C=on`:

- **`full`** generates **8 routes** — every combination of A × B × C.
- **`fast`** generates **4 routes** — the baseline `A=on B=on C=on`, plus
  three one-off variations that flip exactly one knob at a time
  (`A=off B=on C=on`, `A=on B=off C=on`, `A=on B=on C=off`).
- **`mixed`** runs the 4 `fast` routes first; whichever knobs improved
  performance over the baseline are then explored exhaustively in a
  second pass (e.g. if A and C improved, phase 2 sweeps the 4 routes of
  A × C with B pinned to its default).

`--dryRun` enumerates the route list and exits without building any engine —
useful for sanity-checking a large expression before paying for it.
`--dryRun` cannot be combined with `--tuningSearch=mixed`.

```
./trtexec --onnx=model.onnx \
          --tuneBuildRoutes="-match_ragged_mha=[on|off] -copy_ppg=[on|off]" \
          --tuningSearch=full --dryRun
```

#### 7.5: Accuracy-aware tuning

The tuning loop can validate each iteration's output against a reference
(typically a CPU/FP32 capture from another framework) and discard any engine
that drifts beyond a threshold. Combine `--loadInputs`, `--loadRefOutputs`,
and `--accuracyThreshold` with the tuning flags. When `--loadRefOutputs` is
present, `--accuracyThreshold` is required.

```
./trtexec --onnx=model.onnx \
          --tuneBuildRoutes="-match_ragged_mha=[on|off]" \
          --loadInputs=input:input.bin \
          --loadRefOutputs=output:ref_output.bin \
          --accuracyThreshold=0.5 \
          --saveEngine=best.plan
```

**Multiple input/output pairs.** Tuning often needs to be validated across
several inputs (e.g. different batch sizes or representative samples).
Group each `(input, reference-output)` pair behind `--refPair=N` and pass
multiple pairs; every iteration is validated against all of them.

```
./trtexec --onnx=model.onnx \
          --tuneBuildRoutes="-match_ragged_mha=[on|off]" \
          --refPair=0 --loadInputs=input:in0.bin --loadRefOutputs=output:ref0.bin \
          --refPair=1 --loadInputs=input:in1.bin --loadRefOutputs=output:ref1.bin \
          --accuracyThreshold=0.5 --saveEngine=best.plan
```

**Choosing the loss metric.** `--accuracyAlgorithm=<spec>` selects how
the loss is computed per output tensor. All five metrics are non-negative
(lower is better; `0.0` is a perfect match):

| Spec   | Metric |
|--------|--------|
| `l0`   | (default) fraction of elements outside `atol + rtol · abs(ref)`; tweak with `--atol` / `--rtol`. |
| `l1`   | mean absolute error. |
| `l2`   | mean squared error. |
| `lInf` | maximum absolute error. |
| `cos`  | `1 − cosine_similarity(actual, reference)`. |

```
./trtexec --onnx=model.onnx \
          --tuneBuildRoutes="-match_ragged_mha=[on|off]" \
          --loadInputs=input:input.bin --loadRefOutputs=output:ref.bin \
          --accuracyAlgorithm=l2 --accuracyThreshold=0.01 \
          --saveEngine=best.plan
```

Iterations that fail the accuracy check are recorded in the cache but are
excluded from "best engine" selection.

#### 7.6: The tuning cache (and resuming)

`--tuningCacheFile=<path>` writes a JSON record of the sweep: one line
describing the run, followed by one line per completed iteration with the
build route, GPU time, and per-output accuracy loss. This file is both a
human-readable record of the sweep and the input for resume.

```
./trtexec --onnx=model.onnx \
          --tuneBuildRoutes="-match_ragged_mha=[on|off] -copy_ppg=[on|off]" \
          --tuningCacheFile=tune.jsonl --saveEngine=best.plan
```

If the run is interrupted (Ctrl-C, OOM, timeout), resume it with:

```
./trtexec --continue --tuningCacheFile=tune.jsonl
```

`--continue` accepts **only** `--tuningCacheFile=<path>` — passing
`--onnx`, `--tuneBuildRoutes`, `--accuracyThreshold`, or any other flag
alongside it is rejected. The cache file carries everything needed to
continue the original run, and existing iteration results in it are
kept. The sweep picks up at the next iteration after the last one
already recorded.

#### 7.7: Other useful flags

- `--tuningTimeOut=<seconds>` — stop the loop after N elapsed seconds (the
  current iteration finishes first). `-1` (default) disables the timeout.
  Useful for capping a large `full` sweep at a deadline.
- `--saveAllEngines` — in addition to the best engine at `--saveEngine=<p>`,
  write every iteration's engine to `<p>.iter<N>`. Requires `--saveEngine`.
  Disk-heavy; intended for debugging accuracy regressions across iterations.
- `--setBuildRoute=<route>` — see [7.2](#72-building-one-specific-configuration).
  Useful for replaying any single iteration from a sweep by hand.

For the complete list with one-line descriptions, run `./trtexec --help`
and look at the **Build Route Tuning Options** section.

#### 7.8: Caveats of tuning

A few things to keep in mind when relying on a tuning result in production:

- **Improvement is opportunistic.** The default build route may already
  be the fastest one for the (model, hardware) combination you're tuning
  on. Treat any speedup as a bonus, not an expected outcome.

- **An explicit knob value may be overridden during compilation.** Even
  when you pin a knob with `--setBuildRoute=<route>`, the compiler is
  free to change that value internally if the network requires it. The
  engine that `--saveEngine` records is the source of truth — not the
  route string. Ship the saved engine when exact behavior matters.

- **Re-running a route doesn't reproduce engine bytes.** Engine builds
  are not bit-deterministic; kernel timings vary, the builder breaks
  ties accordingly, and the serialized engine reflects those picks.
  Re-running with the same `--setBuildRoute` on the same model and the
  same machine produces an engine with the same knob choices, but the
  bytes may differ slightly. Again — ship the saved engine, not the
  route string, when you need the exact engine that won the sweep.

- **Tuner version matters.** Across TensorRT releases, the set of
  tunable knobs and their default values can change. A route or cache
  produced by one version may reference knobs that no longer exist in
  another, and even the default route is not stable across versions.

- **Pick a sensible `--accuracyThreshold`.** A threshold set too tight
  will reject every iteration, and the sweep will report no winner. If
  you don't have a prior calibration, start loose and ratchet down.

- **Results are model-specific.** The optimal route depends on the
  exact ONNX. A different model — or even the same model rebuilt with
  different shapes or precision flags — invalidates a previously-saved
  result.

- **Results are hardware-specific.** The same model tuned on different
  GPU SKUs can pick different "best" routes. Re-tune when you move to a
  different target.

- **Re-tune after TensorRT / tuner upgrades.** A previously-tuned
  non-default build route is **not** guaranteed to keep its advantage
  after an upgrade — and may even regress relative to the new default.
  Re-tune whenever the tuner version or TensorRT version changes. Only
  performance regressions on the **default** build route are tracked as
  TensorRT performance bugs; non-default routes are best-effort.

## Tool command line arguments

To see the full list of available options and their descriptions, issue the `./trtexec --help` command.

**Note:** Specifying the `--safe` parameter turns the safety mode switch `ON`. By default, the `--safe` parameter is not specified; the safety mode switch is `OFF`. The layers and parameters that are contained within the `--safe` subset are restricted if the switch is set to `ON`. The switch is used for prototyping the safety restricted flows until the TensorRT safety runtime is made available. This parameter is required when loading or saving safe engines with the standard TensorRT package. For more information, see the [Working With Automotive Safety section in the TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#working_auto_safety).

## Additional resources

The following resources provide more details about `trtexec`:

**Documentation**
- [NVIDIA trtexec](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#trtexec)
- [TensorRT Sample Support Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.

# Changelog

April 2019
This is the first release of this `README.md` file.

# Known issues

There are no known issues in this sample.
