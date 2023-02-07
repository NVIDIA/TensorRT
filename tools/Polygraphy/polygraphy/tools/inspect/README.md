# Inspect

## Table of Contents

- [Introduction](#introduction)
- [Subtools](#subtools)
- [Usage](#usage)
- [Examples](#examples)


## Introduction

The `inspect` tool can be used to display information about supported types of files.


## Subtools

- `model` displays information about models, such as network structure.
    This can be useful in case you are unfamiliar with the model, and need to know details about
    the inputs or outputs, or layers.

- `data` displays information about inference inputs and outputs from `Comparator.run()`, like those written by
    `polygraphy run --save-outputs/--save-inputs`.

- [EXPERIMENTAL] `tactics` displays information about the contents of tactic replay files, like those written by
    `polygraphy run --save-tactics`

- `capability` partitions an ONNX model into one or more subgraphs that are supported or unsupported by TensorRT.

- `diff-tactics` can determine potentially bad tactics given a set of known-good tactic replay
    files and a set of bad ones.

    See the [example](../../../examples/cli/debug/01_debugging_flaky_trt_tactics/) for details.

## Usage

See `polygraphy inspect -h` for usage information.


## Examples

For examples, see [this directory](../../../examples/cli/inspect)
