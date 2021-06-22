# Converting To TensorRT And Running Inference


## Introduction

Polygraphy includes a high-level Python API that can convert models
and run inference with various backends. For an overview of the Polygraphy
Python API, see [here](../../../polygraphy/).

In this example, we'll look at how you can leverage the API to easily convert an ONNX
model to TensorRT and run inference with FP16 precision enabled.


## Running the Example

1. Install prerequisites
    * Ensure that TensorRT is installed
    * Install other dependencies with `python3 -m pip install -r requirements.txt`

2. **[Optional]** Inspect the model before running the example:
    ```bash
    polygraphy inspect model identity.onnx
    ```

3. Run the example:
    ```bash
    python3 example.py
    ```

4. **[Optional]** Inspect the TensorRT engine built by the example:
    ```bash
    polygraphy inspect model identity.engine
    ```

## Further Reading

For more details on the Polygraphy Python API, see the
[Polygraphy API reference](https://docs.nvidia.com/deeplearning/tensorrt/polygraphy/docs/index.html).
