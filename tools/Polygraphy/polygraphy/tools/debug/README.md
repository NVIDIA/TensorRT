# [EXPERIMENTAL] Debug

## Table of Contents

- [Introduction](#introduction)
- [Subtools](#subtools)
- [Usage](#usage)
- [Examples](#examples)


## Introduction

The `debug` tool can help debug accuracy issues during inference.

All the `debug` tools work on the same general principles:

1. Iteratively generate models and various other artifacts.

    For example, `debug precision` generates a new TensorRT engine each iteration with some number
    of layers marked to run in a higher precision. `debug reduce` generates smaller and smaller subgraphs
    from the provided ONNX model.

    In many cases, it is also useful to save other artifacts, such as tactic replay files,
    from each iteration.

2. Each iteration, check whether the model generated in that iteration is good or bad, and sort artifacts
    into `good` and `bad` directories.

    In order to determine whether a model is good or bad, the subtool uses the `--check` command
    provided by the user (that's you!). This can be any command that checks some aspect of the generated
    model and determines whether it is a good or bad model. For example, to debug an accuracy issue, you
    could use `polygraphy run <model> --trt --load-outputs <golden_outputs.json>` or some other accuracy validation
    script.

    When the `--check` command exits with a failure (what qualifies as a "failure" can be controlled via various
    command-line options like `--fail-regex` and `--fail-returncode`), the iteration is counted as a failure, and
    any artifacts specified to `--artifacts` are moved into a `bad` directory.


Therefore, the general form of most `debug` subtools is:
```
polygraphy debug <subtool> <model> [--artifacts per_iteration_files_to_sort...] \
    --check <checker_script_or_tool> <per_iteration_model> [options to checker_script_or_tool]
```


## Subtools

`debug` provides subtools for different tasks:

- `build` can repeatedly build TensorRT engines and sort generated
    artifacts into `good` and `bad` directories. This is more efficient than
    running `polygraphy run` repeatedly since some of the work, like model
    parsing, can be shared across iterations.

    See the [example](../../../examples//cli/debug/01_debugging_flaky_trt_tactics/) for details.

- `precision` can be used to determine which layers of a TensorRT network need to be
    run in a higher precision in order to maintain the desired accuracy.

    The tool works by iteratively marking a subset of the layers in the network in the specified
    higher precision (`float32` by default) and generating engines similar to `build`.

- `diff-tactics` can determine potentially bad tactics given a set of known-good tactic replay
    files and a set of bad ones.

    See the [example](../../../examples//cli/debug/01_debugging_flaky_trt_tactics/) for details.

- [EXPERIMENTAL] `reduce` can reduce failing ONNX models to a minimal subgraph of failing nodes.
    This can make further debugging significantly easier.

    See the [example](../../../examples//cli/debug/02_reducing_failing_onnx_models/) for details.

- [EXPERIMENTAL] `repeat` can run an arbitrary command repeatedly, sorting generated artifacts
    into `good` and `bad` directories. This is more general than the other `debug` subtools, and is
    effectively equivalent to manually running a command repeatedly and moving files between runs.


## Usage

See `polygraphy debug -h` for usage information.


## Examples

For examples, see [this directory](../../../examples/cli/debug)
