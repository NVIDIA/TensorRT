# Surgeon

## Table of Contents

- [Introduction](#introduction)
- [Subtools](#subtools)
- [Usage](#usage)
- [Examples](#examples)


## Introduction

The `surgeon` tool uses [ONNX-GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)
to modify an ONNX model.


## Subtools

`surgeon` provides subtools to perform different functions:

- `extract` can extract subgraphs from models. It can also be used for changing the shapes/datatypes of the
    model inputs/outputs by specifying the existing inputs/outputs, but with different shapes/datatypes. This
    can be useful to make an input dimension dynamic, for example.

- `sanitize` can simplify and sanitize an ONNX model by removing unused nodes and folding constants.

- [EXPERIMENTAL] `insert` can insert a node into a model, optionally replacing existing subgraphs.


## Usage

See `polygraphy surgeon -h` for usage information.


## Examples

For examples, see [this directory](../../../examples/cli/surgeon)
