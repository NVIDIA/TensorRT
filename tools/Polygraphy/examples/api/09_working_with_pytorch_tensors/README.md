# Working With PyTorch Tensors

## Introduction

Some runners like `OnnxrtRunner` and `TrtRunner` can accept and return PyTorch tensors
in addition to NumPy arrays. When PyTorch tensors are provided in the inputs, the runner
will return the outputs as PyTorch tensors as well. This can be especially useful in
cases where PyTorch supports a data type that is not supported by NumPy, such as BFloat16.

Polygraphy's included TensorRT `Calibrator` can also accept PyTorch tensors directly.

This example uses PyTorch tensors on the GPU where possible (i.e. if a GPU-enabled version
of PyTorch is installed). When the tensors already reside on GPU memory, no additional copies
are required in the runner/calibrator.

## Running The Example

1. Install prerequisites
    * Ensure that TensorRT is installed
    * Install other dependencies with `python3 -m pip install -r requirements.txt`


2. Run the example:

    ```bash
    python3 example.py
    ```


## See Also

* [Inference With TensorRT](../00_inference_with_tensorrt/)
* [INT8 Calibration In TensorRT](../04_int8_calibration_in_tensorrt/)
