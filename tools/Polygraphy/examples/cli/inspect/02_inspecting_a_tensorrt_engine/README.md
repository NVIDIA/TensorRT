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
    polygraphy inspect model dynamic_identity.engine \
        --show layers
    ```

    NOTE: `--show layers` only works if the engine was built with a `profiling_verbosity` other than `NONE`.
        Higher verbosities make more per-layer information available.

    This will display something like:

    ```
    [I] ==== TensorRT Engine ====
        Name: Unnamed Network 0 | Explicit Batch Engine

        ---- 1 Engine Input(s) ----
        {X [dtype=float32, shape=(1, 2, -1, -1)]}

        ---- 1 Engine Output(s) ----
        {Y [dtype=float32, shape=(1, 2, -1, -1)]}

        ---- Memory ----
        Device Memory: 0 bytes

        ---- 2 Profile(s) (2 Tensor(s) Each) ----
        - Profile: 0
            Tensor: X          (Input), Index: 0 | Shapes: min=(1, 2, 1, 1), opt=(1, 2, 3, 3), max=(1, 2, 5, 5)
            Tensor: Y         (Output), Index: 1 | Shape: (1, 2, -1, -1)

        - Profile: 1
            Tensor: X          (Input), Index: 0 | Shapes: min=(1, 2, 2, 2), opt=(1, 2, 4, 4), max=(1, 2, 6, 6)
            Tensor: Y         (Output), Index: 1 | Shape: (1, 2, -1, -1)

        ---- 1 Layer(s) Per Profile ----
        - Profile: 0
            Layer 0    | node_of_Y [Op: Reformat]
                {X [shape=(1, 2, -1, -1)]}
                 -> {Y [shape=(1, 2, -1, -1)]}

        - Profile: 1
            Layer 0    | node_of_Y [profile 1] [Op: Reformat]
                {X [profile 1] [shape=(1, 2, -1, -1)]}
                 -> {Y [profile 1] [shape=(1, 2, -1, -1)]}
    ```

    It is also possible to show more detailed layer information using `--show layers attrs`.
