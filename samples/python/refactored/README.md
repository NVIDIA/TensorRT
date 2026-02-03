# TensorRT Refactored Samples

This directory contains refactored and improved versions of TensorRT samples, demonstrating best practices and modern implementations.

## Available Samples

| Sample Name | Description | Format |
|-------------|-------------|---------|
| [1_run_onnx_with_tensorrt](./1_run_onnx_with_tensorrt) | Demonstrates ONNX model conversion to TensorRT and inference comparison | `ipynb` |
| [2_construct_network_with_layer_apis](./2_construct_network_with_layer_apis) | Constructing a Network with TensorRT Layer APIs | `ipynb` |



## Launch Instructions

1.  Navigate to the desired sample directory and start the Jupyter server:
    ```bash
    pip install notebook
    cd 1_run_onnx_with_tensorrt # or any other sample
    jupyter notebook
    ```
2.  Then, open the `main.ipynb` file in the Jupyter Notebook interface that opens in your web browser.

# Changelog

October 2025
Migrate to strongly typed APIs.

August 2025
Removed support for Python versions < 3.10.
