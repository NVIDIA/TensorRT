# [EXPERIMENTAL] Debug

## Table of Contents

- [Introduction](#introduction)
- [Subtools](#subtools)
- [Usage](#usage)
- [Examples](#examples)


## Introduction

The `debug` tool can help debug accuracy issues during inference.

## Subtools

`debug` provides subtools for different tasks:

- `build` can repeatedly build TensorRT engines and sort generated
    artifacts into `good` and `bad` directories. This is more efficient than
    running `polygraphy run` repeatedly since some of the work, like model
    parsing, can be shared across iterations.

    See the [example](../../../examples/cli/debug/01_debugging_flaky_trt_tactics/) for details.

- `precision` can be used to determine which layers of a TensorRT network need to be
    run in a higher precision in order to maintain the desired accuracy.

    The tool works by iteratively marking a subset of the layers in the network in the specified
    higher precision (`float32` by default) and generating engines similar to `build`.

- [EXPERIMENTAL] `reduce` can reduce failing ONNX models to a minimal subgraph of failing nodes.
    This can make further debugging significantly easier.

    See the [example](../../../examples/cli/debug/02_reducing_failing_onnx_models/) and help output for details.

- [EXPERIMENTAL] `repeat` can run an arbitrary command repeatedly, sorting generated artifacts
    into `good` and `bad` directories. This is more general than the other `debug` subtools, and is
    effectively equivalent to manually running a command repeatedly and moving files between runs.


## Usage

See `polygraphy debug -h` for usage information.


## Examples

For examples, see [this directory](../../../examples/cli/debug)
