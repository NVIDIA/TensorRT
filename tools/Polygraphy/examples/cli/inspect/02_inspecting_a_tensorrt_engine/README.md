# Inspecting A TensorRT Engine

The `inspect model` subtool can load and display information
about TensorRT engines, i.e. plan files:

For example, first we'll generate an engine with dynamic shapes
and 2 profiles:

```bash
polygraphy run ../../../models/identity.onnx --trt \
    --trt-min-shapes X,1x2x1x1 --trt-opt-shapes X,1x2x3x3 --trt-max-shapes X,1x2x5x5 \
    --trt-min-shapes X,1x2x2x2 --trt-opt-shapes X,1x2x4x4 --trt-max-shapes X,1x2x6x6 \
    --save-engine identity.engine
```

Next, we can inspect it:

```bash
polygraphy inspect model identity.engine
```

This will display something like:

```
[I] ==== TensorRT Engine ====
    Name: Unnamed Network 0 | Explicit Batch Engine (2 layers)

    ---- 1 Engine Inputs ----
    {x [dtype=float32, shape=(1, 1, 2, 2)]}

    ---- 1 Engine Outputs ----
    {y [dtype=float32, shape=(1, 1, 2, 2)]}

    ---- Memory ----
    Workspace Memory: 0 bytes

    ---- 2 Profiles (2 Bindings Each) ----
    - Profile: 0
        Binding Index: 0 (Input)  [Name: x]             | Shapes: min=(1, 1, 2, 2), opt=(1, 1, 2, 2), max=(1, 1, 2, 2)
        Binding Index: 1 (Output) [Name: y]             | Shape: (1, 1, 2, 2)
    - Profile: 1
        Binding Index: 2 (Input)  [Name: x [profile 1]] | Shapes: min=(1, 1, 2, 2), opt=(1, 1, 2, 2), max=(1, 1, 2, 2)
        Binding Index: 3 (Output) [Name: y [profile 1]] | Shape: (1, 1, 2, 2)
```
