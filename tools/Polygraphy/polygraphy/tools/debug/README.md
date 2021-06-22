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

- `precision` can be used to determine which layers of a TensorRT network need to be
    run in a higher precision in order to maintain the desired accuracy.

    The tool works by iteratively marking a subset of the layers in the network in the specified
    higher precision (`float32` by default) and generating engines similar to `build`.

- `diff-tactics` can determine potentially bad tactics given a set of known-good tactic replay
    files and a set of bad ones.

- [EXPERIMENTAL] `reduce` can reduce failing ONNX models to a minimal subgraph of failing nodes.
    This can make further debugging significantly easier.

    You can invoke it with the model and a command that can check intermediate models.
    The intermediate models will be written to `polygraphy_debug.onnx` by default.
    For example, to reduce a model with accuracy errors:
    ```bash
    polygraphy debug reduce model.onnx -o reduced.onnx \
        --check polygraphy run polygraphy_debug.onnx --onnxrt --trt
    ```

    When using a model with dynamic shapes, you can use `--model-input-shapes` to freeze the
    shapes of the intermediate tensors. In case ONNX shape inference is not able to freeze shapes,
    you can enable `--force-fallback-shape-inference`.
    Alternatively, you can use `--no-reduce-inputs` so that the model inputs are not modified.
    This can be useful in cases where it may not be trivial to implement a `--check` command
    that can determine the shapes to use for intermediate tensors.

- [EXPERIMENTAL] `repeat` can run an arbitrary command repeatedly, sorting generated artifacts
    into `good` and `bad` directories.


## Usage

See `polygraphy debug -h` for usage information.


## Examples

For examples, see [this directory](../../../examples/cli/debug)
