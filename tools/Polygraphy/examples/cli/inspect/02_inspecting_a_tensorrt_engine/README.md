# Inspecting A TensorRT Engine


## Introduction

The `inspect model` subtool can load and display information
about TensorRT engines, i.e. plan files:


## Running The Example

1. Generate an engine with dynamic shapes and 2 profiles:

    ```bash
    polygraphy run dynamic_identity.onnx --trt \
        --trt-min-shapes X:[1,2,1,1] --trt-opt-shapes X:[1,2,3,3] --trt-max-shapes X:[1,2,5,5] \
        --trt-min-shapes X:[1,2,2,2] --trt-opt-shapes X:[1,2,4,4] --trt-max-shapes X:[1,2,6,6] \
        --save-engine dynamic_identity.engine
    ```

2. Inspect the engine:

    ```bash
    polygraphy inspect model dynamic_identity.engine
    ```

    This will display something like:

    ```
    [I] ==== TensorRT Engine ====
        Name: Unnamed Network 0 | Explicit Batch Engine (2 layers)

        ---- 1 Engine Input(s) ----
        {X [dtype=float32, shape=(1, 2, -1, -1)]}

        ---- 1 Engine Output(s) ----
        {Y [dtype=float32, shape=(1, 2, -1, -1)]}

        ---- Memory ----
        Device Memory: 0 bytes

        ---- 2 Profile(s) (2 Binding(s) Each) ----
        - Profile: 0
            Binding Index: 0 (Input)  [Name: X]             | Shapes: min=(1, 2, 1, 1), opt=(1, 2, 3, 3), max=(1, 2, 5, 5)
            Binding Index: 1 (Output) [Name: Y]             | Shape: (1, 2, -1, -1)

        - Profile: 1
            Binding Index: 2 (Input)  [Name: X [profile 1]] | Shapes: min=(1, 2, 2, 2), opt=(1, 2, 4, 4), max=(1, 2, 6, 6)
            Binding Index: 3 (Output) [Name: Y [profile 1]] | Shape: (1, 2, -1, -1)
    ```
