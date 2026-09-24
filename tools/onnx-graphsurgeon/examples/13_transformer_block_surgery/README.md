# Transformer Block Surgery

## Introduction

Transformer-style ONNX graphs often contain shape-only or bookkeeping operators around
attention blocks. This example shows how to use ONNX GraphSurgeon for conservative graph
surgery on a small transformer-like block while keeping the model in standard ONNX
operators.

The example performs two local rewrites:

- Remove `Identity` nodes by rewiring their consumers.
- Cancel adjacent `Transpose` nodes when their permutations compose to the identity permutation.

These cleanups are intentionally small and semantics-preserving. They can make generated
transformer graphs easier to inspect in Netron and prepare for downstream tooling without
introducing custom fused operators.

## Running the example

1. Generate a transformer-like ONNX model:
    ```bash
    python3 generate.py
    ```

2. Remove no-op graph structure:
    ```bash
    python3 surgeon.py --input model.onnx --output cleaned.onnx
    ```

The generated model contains a residual projection with an `Identity` and a canceling
`Transpose` pair. The surgery pass exports a checked ONNX model after cleanup and
topological sorting.
