# Comparing Frameworks


## Introduction

One of the core features of Polygraphy is comparison of model outputs across multiple
different backends. This makes it possible to check the accuracy of one backend with
respect to another.

In this example, we'll look at how you can use the Polygraphy API to run inference
with synthetic input data using ONNX-Runtime and TensorRT, and then compare the results.


## Running The Example

1. Install prerequisites
    * Ensure that TensorRT is installed
    * Install other dependencies with `python3 -m pip install -r requirements.txt`

2. Run the example
    ```bash
    python3 example.py
    ```

3. **[Optional]** Inspect the inference outputs from the example:

    ```bash
    polygraphy inspect data inference_results.json
    ```
