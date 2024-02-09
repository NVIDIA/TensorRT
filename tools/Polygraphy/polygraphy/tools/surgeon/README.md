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

- [EXPERIMENTAL] `prune` can prune an ONNX model to have [2:4 structured sparsity](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/) by setting the some values of the weights to zero. Note that this will *not* retain the accuracy of the model and should hence be used only for functional testing.

- [EXPERIMENTAL] `weight-strip` can strip the initializers of selective nodes in an ONNX model. If an initializer is identified as 2:4 structured sparse, the sparsity structure is encoded in the initializer to preserve sparsity during reconstruction. The tool only supports 2:4 structured sparsity.

- [EXPERIMENTAL] `weight-reconstruct` reads a weightless ONNX model and fills the empty initializers with proxy tensors. The reconstruction preserves 2:4 structed sparsity for initializers that were identified as sparse during weight stripping.

## Usage

See `polygraphy surgeon -h` for usage information.


## Examples

For examples, see [this directory](../../../examples/cli/surgeon)
