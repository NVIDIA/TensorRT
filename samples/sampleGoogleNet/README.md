# Building And Running GoogleNet In TensorRT

**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
    * [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Preparing sample data](#preparing-sample-data)
- [Running the sample](#running-the-sample)
    * [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleGoogleNet, demonstrates how to import a model trained with Caffe into TensorRT using GoogleNet as an example. Specifically, this sample builds a TensorRT engine from the saved Caffe model, sets input values to the engine, and runs it.

## How does this sample work?

This sample constructs a network based on a saved Caffe model and network description. This sample comes with a pre-trained model called `googlenet.caffemodel` located in the `data/googlenet` directory. The model used by this sample was trained using ImageNet. For more information, see the [BAIR/BVLC GitHub page](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet). The sample reads two Caffe files to build the network:

- `googlenet.prototxt` - The prototxt file that contains the network design.    
- `googlenet.caffemodel` - The model file which contains the trained weights for the network.

For more information, see [Importing A Caffe Model Using The C++ Parser API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_caffe_c).

The sample then builds the TensorRT engine using the constructed network. See [Building an Engine in C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#build_engine_c) for more information on this. Finally, the sample runs the engine with the test input (all zeroes) and reports if the sample ran as expected.

### TensorRT API layers and ops

In this sample, the following layers are used.  For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions. Specifically, this sample uses the Activation layer with the type `kRELU`.

[Concatenation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#concatenation-layer)
The Concatenation layer links together multiple tensors of the same non-channel sizes along the channel dimension.

[Convolution layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#convolution-layer)
The Convolution layer computes a 2D (channel, height, and width) convolution, with or without bias.

[FullyConnected layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#fullyconnected-layer)
The FullyConnected layer implements a matrix-vector product, with or without bias.

[LRN layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#lrn-layer)
The LRN layer implements cross-channel Local Response Normalization.

[Pooling layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#pooling-layer)
The Pooling layer implements pooling within a channel. Supported pooling types are `maximum`, `average` and `maximum-average blend`.

[SoftMax layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#softmax-layer)
The SoftMax layer applies the SoftMax function on the input tensor along an input dimension specified by the user.

## Preparing sample data

1. Download the sample data from [TensorRT release tarball](https://developer.nvidia.com/nvidia-tensorrt-download#), if not already mounted under `/usr/src/tensorrt/data` (NVIDIA NGC containers) and set it to `$TRT_DATADIR`.
    ```bash
    export TRT_DATADIR=/usr/src/tensorrt/data
    ```

## Running the sample

1. Compile the sample by following build instructions in [TensorRT README](https://github.com/NVIDIA/TensorRT/).

2. Run the sample to build and run a GPU inference engine for GoogleNet.
    ```bash
    ./sample_googlenet --datadir=<path_to_data_directory> --useDLACore=N
    ```

    For example:
    ```bash
    ./sample_googlenet --datadir $TRT_DATADIR/googlenet
    ```

  **NOTE:** By default, this sample assumes both `googlenet.prototxt` and `googlenet.caffemodel` files are located in either the `data/samples/googlenet/` or `data/googlenet/` directories. The default directory can be changed by supplying the `--datadir=<new_path/>` path as a command line argument.

3. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
	&&&& RUNNING TensorRT.sample_googlenet # ./sample_googlenet
	[I] Building and running a GPU inference engine for GoogleNet
	[I] [TRT] Detected 1 input and 1 output network tensors.
	[I] Ran ./sample_googlenet with:
	[I] Input(s): data
	[I] Output(s): prob
	&&&& PASSED TensorRT.sample_googlenet # ./sample_googlenet
	```
	This output shows that the input to the sample is called `data`, the output tensor is called `prob` and the sample ran successfully; `PASSED`.


### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.


# Additional resources

The following resources provide a deeper understanding about GoogleNet:

**GoogleNet**
- [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- [BVLC/BAIR Caffe GitHub](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)

**Documentation**
- [TensorRT Sample Support Guide: sampleGoogleNet](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#googlenet_sample)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.

# Changelog

February 2019
This `README.md` file was recreated, updated and reviewed.

# Known issues

There are no known issues in this sample.
