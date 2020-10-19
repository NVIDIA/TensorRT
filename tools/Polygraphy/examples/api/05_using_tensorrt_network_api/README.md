# Using The TensorRT Network API


## Introduction

In addition to loading existing models, TensorRT allows you to define networks by hand
using the network API.

In this example, we'll look at how you can use Polygraphy's `extend` decorator, covered in
[example 03](../03_interoperating_with_tensorrt), in conjunction with the `CreateNetwork`
loader to seamlessly integrate a network defined using TensorRT APIs with Polygraphy.


## Running the Example

1. Install prerequisites
    a. Ensure that TensorRT is installed
    b. Install other dependencies with `python3 -m pip install -r requirements.txt`

2. Run the example:
    ```bash
    python3 example.py
    ```


## Further Reading

For more information on the TensorRT Network API, see the
[TensorRT API documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/pyGraph.html)
