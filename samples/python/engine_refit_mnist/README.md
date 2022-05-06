# Refitting an Engine in Python

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

This sample, `engine_refit_mnist`, trains an MNIST model in PyTorch, recreates the network in TensorRT with dummy weights, and finally refits the TensorRT engine with weights from the model. Refitting allows us to quickly modify the weights in a TensorRT engine without needing to rebuild.

## How does this sample work?

This sample first reconstructs the model using the TensorRT network API. In the first pass, the weights for one of the conv layers (`conv_1`) are populated with dummy values resulting in an incorrect inference result. In the second pass, we refit the engine with the trained weights for the `conv_1` layer and run inference again. With the weights now set correctly, inference should provide correct results.

### TensorRT API layers and ops

In this sample, the following layers are used. For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions. Specifically, this sample uses the Activation layer with the type `kRELU`.

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

On Jetson Nano, you will need nvcc in the `PATH` for installing pycuda:
```bash
export PATH=${PATH}:/usr/local/cuda/bin/
```

## Running the sample

1.  Run the sample to create a TensorRT engine and run inference:
    `python3 sample.py [-d DATA_DIR]`

    to run the sample with Python 3.

    **Note:** If the TensorRT sample data is not installed in the default location, for example `/usr/src/tensorrt/data/`, the data directory must be specified. For example:
    `python3 sample.py -d /path/to/my/data/`.

2.  Verify that the sample ran successfully. If the sample runs successfully you should see a match between the test case and the prediction after refitting.
    ```
    Accuracy Before Engine Refit
    Got 892 correct predictions out of 10000 (8.9%)
    Accuracy After Engine Refit (expecting 98.0% correct predictions)
    Got 9798 correct predictions out of 10000 (98.0%)
    ```

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
usage: sample.py [-h]

Description for this sample

optional arguments:
    -h, --help show this help message and exit
```

# Additional resources

The following resources provide a deeper understanding about the engine refitting functionality and the network used in this sample:

**Network**
- [MNIST network](http://yann.lecun.com/exdb/lenet/)

**Dataset**
- [MNIST dataset](http://yann.lecun.com/exdb/mnist/)

**Documentation**
- [Introduction to NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working with TensorRT Using the Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [Refitting an Engine](http://sw-docs-dgx-station.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#refitting-engine-c)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

March 2021
Documented the Python version limitations.

March 2019
This `README.md` file was recreated, updated and reviewed.

# Known issues

This sample only supports Python 3.6+ due to `torch` and `torchvision` version requirements.
