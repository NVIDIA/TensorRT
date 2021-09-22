# Inspecting Input Data


## Introduction

The `inspect data` subtool can display information about input data generated
by a data loader.


## Running The Example
1. Generate some input data by running inference:

    ```bash
    polygraphy run identity.onnx --onnxrt --save-inputs inputs.json
    ```

2. Inspect the input data:

    ```bash
    polygraphy inspect data inputs.json --show-values
    ```

    This will display something like:

    ```
    [I] ==== Data (1 iterations) ====

        x [dtype=float32, shape=(1, 1, 2, 2)] | Stats: mean=0.35995, std-dev=0.25784, var=0.066482, median=0.35968, min=0.00011437 at (0, 0, 1, 0), max=0.72032 at (0, 0, 0, 1), avg-magnitude=0.35995
            [[[[4.17021990e-01 7.20324516e-01]
               [1.14374816e-04 3.02332580e-01]]]]
    ```
