# Comparing Across Runs


## Introduction

In some cases, it may be be useful to compare results across different runs of `polygraphy run` -
for example, comparing between different machines, or comparing between multiple different versions
of the same library.

In this example, we'll compare results generated on two different (hypothetical) systems: System A and System B.


## Running The Example

1. First save the results from System A:

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --save-outputs system_a_results.json
    ```

2. Next, run the model on System B, and load the results saved from
    System A to compare against:

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --load-outputs system_a_results.json
    ```
