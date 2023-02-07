# Inspecting A TensorFlow Graph


## Introduction

The `inspect model` subtool can display TensorFlow graphs.


## Running The Example

1. Inspect a TensorFlow frozen model:

    ```bash
    polygraphy inspect model identity.pb --model-type=frozen
    ```

    This will display something like:

    ```
    [I] ==== TensorFlow Graph ====
        ---- 1 Graph Inputs ----
        {Input:0 [dtype=float32, shape=(1, 15, 25, 30)]}

        ---- 1 Graph Outputs ----
        {Identity_2:0 [dtype=float32, shape=(1, 15, 25, 30)]}

        ---- 4 Nodes ----
    ```
