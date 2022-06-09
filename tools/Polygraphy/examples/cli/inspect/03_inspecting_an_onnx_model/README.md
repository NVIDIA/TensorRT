# Inspecting An ONNX Model


## Introduction

The `inspect model` subtool can display ONNX models.


## Running The Example

1. Inspect the ONNX model:

    ```bash
    polygraphy inspect model identity.onnx --show layers
    ```

    This will display something like:

    ```
    [I] ==== ONNX Model ====
        Name: test_identity | ONNX Opset: 8

        ---- 1 Graph Input(s) ----
        {x [dtype=float32, shape=(1, 1, 2, 2)]}

        ---- 1 Graph Output(s) ----
        {y [dtype=float32, shape=(1, 1, 2, 2)]}

        ---- 0 Initializer(s) ----
        {}

        ---- 1 Node(s) ----
        Node 0    |  [Op: Identity]
            {x [dtype=float32, shape=(1, 1, 2, 2)]}
             -> {y [dtype=float32, shape=(1, 1, 2, 2)]}
    ```

    It is also possible to show detailed layer information, including layer attributes, using `--show layers attrs weights`.
