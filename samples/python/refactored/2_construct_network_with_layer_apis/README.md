# Sample 2: Constructing a Network with TensorRT Layer APIs

This sample demonstrates how to build a TensorRT network definition from scratch using the TensorRT Layer APIs, focusing on constructing a recurrent neural network (LSTM) and utilizing advanced builder features.

## Description

This sample constructs a simple, single-layer Long Short-Term Memory (LSTM) network using the TensorRT Layer APIs. The primary goal is to illustrate how to:

1.  Define individual network layers and their connections programmatically using Python (**TensorRT Layer API**). This includes layers like constants, matrix multiply, element-wise operations, activations, and slicing.
2.  Implement recurrent logic by building an LSTM cell and using TensorRT's `add_loop` construct to create a recurrent LSTM layer (**Recurrent Network Construction**).
3.  Monitor the potentially lengthy engine build process by implementing `IProgressMonitor` for real-time feedback (**Build Progress Monitoring**).
4.  Configure the builder for engine portability using `BuilderFlag.VERSION_COMPATIBLE` to create more portable engines (**Version-Compatible Engines**).
5.  Run inference and verify the custom network's correctness by utilizing Polygraphy's `TrtRunner` for simplified engine loading/execution (**Inference with Polygraphy**) and comparing the TensorRT engine's output against a reference NumPy implementation (**NumPy Verification**).

## Additional Resources
*   [tensorrtx repo](https://github.com/wang-xinyu/tensorrtx): Offers real-world examples of constructing complex networks using the TensorRT Layer APIs.

# Changelog

August 2025
Removed support for Python versions < 3.10.
