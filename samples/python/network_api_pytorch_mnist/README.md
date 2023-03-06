# “Hello World” For TensorRT Using PyTorch And Python

**Table Of Contents**

- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
    * [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
    * [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, `network_api_pytorch_mnist`, trains a convolutional model on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset and runs inference with a TensorRT engine.

## How does this sample work?

This sample is an end-to-end sample that trains a model in PyTorch, recreates the network in TensorRT, imports weights from the trained model, and finally runs inference with a TensorRT engine. For more information, see [Creating A Network Definition In Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#network_python).

The `sample.py` script imports the functions from the `mnist.py` script for training the PyTorch model, as well as retrieving test cases from the PyTorch Data Loader.

### TensorRT API layers and ops

In this sample, the following layers are used. For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions. Specifically, this sample uses the Activation layer with the type `RELU`.

[Convolution layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#convolution-layer)
The Convolution layer computes a 2D (channel, height, and width) convolution, with or without bias.

[MatrixMultiplyLayer](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#matrixmultiply-layer)
The MatrixMultiply layer implements a matrix multiplication.
(The [FullyConnected layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#fullyconnected-layer) is deprecated since 8.4.
The bias of FullyConnected semantic can be added with an
[ElementwiseLayer](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#elementwise-layer) of `SUM` operation.)

[Pooling layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#pooling-layer)
The Pooling layer implements pooling within a channel. Supported pooling types are `maximum`, `average` and `maximum-average blend`.

## Prerequisites

1. Upgrade pip version and install the sample dependencies.
    ```bash
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
    ```

To run this sample you must be using Python 3.6 or newer.

On PowerPC systems, you will need to manually install PyTorch using IBM's [PowerAI](https://www.ibm.com/support/knowledgecenter/SS5SF7_1.6.0/navigation/pai_install.htm).

2. The MNIST dataset can be found under the data directory (usually `/usr/src/tensorrt/data/mnist`) if using the TensorRT containers. It is also bundled along with the [TensorRT tarball](https://developer.nvidia.com/nvidia-tensorrt-download).

## Running the sample

1.  Run the sample to create a TensorRT inference engine and run inference:
    `python3 sample.py`

2.  Verify that the sample ran successfully. If the sample runs successfully you should see a match between the test case and the prediction.
     ```
    Test Case: 0
    Prediction: 0
    ```

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.

# Additional resources

The following resources provide a deeper understanding about getting started with TensorRT using Python:

**Model**
- [MNIST model](https://github.com/pytorch/examples/tree/master/mnist)

**Dataset**
- [MNIST database](http://yann.lecun.com/exdb/mnist/)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

September 2021
Updated the sample to use explicit batch network definition.

March 2021
Documented the Python version limitations.

February 2019
This `README.md` file was recreated, updated and reviewed.

# Known issues

This sample only supports Python 3.6+ due to `torch` and `torchvision` version requirements.
