# Folding GroupNorm-via-InstanceNorm

## Introduction

PyTorch versions prior to 2.5 export `torch.nn.GroupNorm` as a sequence of
five operators when targeting ONNX opset versions below 18. The pattern is

```
    x  ->  Reshape  ->  InstanceNormalization  ->  Reshape  ->  Mul  ->  Add  ->  y
```

where the `InstanceNormalization` runs with constant scale `1` and bias `0`,
and the trailing `Mul`/`Add` apply the learned per-channel `gamma`/`beta`. The
second `Reshape` reads its target shape from a `Shape` op fed by `x`.

This example demonstrates how to detect that pattern and rewrite it as a
single native `GroupNormalization` node (opset 21+). Doing so avoids the
fused-norm code path inside TensorRT, which has been observed to drift from
ONNX Runtime when reduction extents are very large or when `num_groups`
equals `num_channels`.

## Running The Example

1. Generate a small model containing the legacy pattern.

   ```bash
   python3 generate.py
   ```

2. Fold the pattern into a `GroupNormalization` op.

   ```bash
   python3 fold.py model.onnx folded.onnx
   ```

3. Inspect the resulting graph in [Netron](https://netron.app) to confirm the
   five-op subgraph collapsed to a single `GroupNormalization`.

## How It Works

`fold.py` walks every `InstanceNormalization` node in the graph and verifies
that its surrounding nodes match the legacy template. When the match is good
it pulls `num_groups` from the upstream `Reshape` constant, lifts the
per-channel weights out of the trailing `Mul`/`Add`, flattens them to 1D, and
wires a single `GroupNormalization` node that consumes the original input
tensor and produces the original output tensor. `graph.cleanup()` then
removes the now-orphaned reshapes, the `Shape` op, and the dangling
constants.
