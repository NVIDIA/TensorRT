# Surgeon

## Table of Contents

- [Introduction](#introduction)
- [Subtools](#subtools)
- [Usage](#usage)
- [Examples](#examples)


## Introduction

The `surgeon` tool uses [ONNX-GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon)
to modify an ONNX model.


## Subtools

`surgeon` provides subtools to perform different functions:

- `extract` can extract subgraphs from models. It can also be used for changing the shapes/datatypes of the
    model inputs/outputs by specifying the existing inputs/outputs, but with different shapes/datatypes. This
    can be useful to make an input dimension dynamic, for example.

- [EXPERIMENTAL] `insert` can insert a node into a model. Currently, this subtool does not support specifiying attributes.

- [EXPERIMENTAL] `sanitize` can remove unused nodes, topologically sort, and fold constants in an ONNX model.


## Usage

See `polygraphy surgeon -h` for usage information.


## Examples

For examples, see [this directory](../../../examples/cli/surgeon)
