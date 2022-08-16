# Algorithm Selection API usage example based off sampleMNIST in TensorRT

**Table Of Contents**

- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
	- [Setup the algorithm selectors](#setup-the-algorithm-selectors)
- [Preparing sample data](#preparing-sample-data)
- [Running the sample](#running-the-sample)
	- [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleAlgorithmSelector, shows an example of how to use the algorithm selection API based on sampleOnnxMNIST ([documentation](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html#onnx_mnist_sample)).

This sample demonstrates the usage of `IAlgorithmSelector` to deterministically build TRT engines.
It also shows the usage of `IAlgorithmSelector::selectAlgorithms` to define heuristics for selection of algorithms.

## How does this sample work?

This sample uses a Onnx model that was trained on the [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md).

Specifically, this sample performs the following steps:
- Performs the basic setup and initialization of TensorRT using the Onnx parser
- [Imports a trained Onnx model using Onnx parser](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#import_onnx_c)
- Preprocesses the input and stores the result in a managed buffer
- [Sets up three instances of algorithm selector](#setup-the-algorithm-selectors)
- [Builds three engines using the algorithm selectors](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#build_engine_c)
- [Serializes and deserializes the engines](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#serial_model_c)
- [Uses the engines to perform inference on an input image](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_c)

To verify whether the engine is operating correctly, this sample picks a 28x28 image of a digit at random and runs inference on it using the engine it created. The output of the network is a probability distribution on the digit, showing which digit is likely to be that in the image.

### Setup the algorithm selectors
1. AlgorithmCacheWriter - Uses `IAlgorithmSelector::reportAlgorithms` to write TensorRT's default algorithm choices to a file "AlgorithmChoices.txt".
2. AlgorithmCacheReader - Uses `IAlgorithmSelector::selectAlgorithms` to replicate algorithm choices from the file "AlgorithmChoices.txt" and verifies the choices using `IAlgorithmSelector::reportAlgorithms`.
3. MinimumWorkspaceAlgorithmSelector - Uses `IAlgorithmSelector::selectAlgorithms` to select algorithms with minimum workspace requirements.

## Preparing sample data

1. Download the sample data from [TensorRT release tarball](https://developer.nvidia.com/nvidia-tensorrt-download#), if not already mounted under `/usr/src/tensorrt/data` (NVIDIA NGC containers) and set it to `$TRT_DATADIR`.
    ```bash
    export TRT_DATADIR=/usr/src/tensorrt/data
    pushd $TRT_DATADIR/mnist
    pip3 install Pillow
    popd
    ```

## Running the sample

1. Compile the sample by following build instructions in [TensorRT README](https://github.com/NVIDIA/TensorRT/).

2. Run the sample to perform inference on the digit:
    ```bash
    ./sample_algorithm_selector [-h] [--datadir=/path/to/data/dir/] [--useDLA=N] [--fp16 or --int8]
    ```

    For example:
    ```bash
    ./sample_algorithm_selector --datadir $TRT_DATADIR/mnist --fp16
    ```

	This sample reads the `mnist.onnx` file to build the network:

	This sample can be run in FP16 and INT8 modes as well.

	**Note:** By default, the sample expects these files to be in either the `data/samples/mnist/` or `data/mnist/` directories. The list of default directories can be changed by adding one or more paths with `--datadir=/new/path/` as a command line argument.

3.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following; ASCII rendering of the input image with digit 3:
    ```
	&&&& RUNNING TensorRT.sample_algorithm_selector # ./sample_algorithm_selector
	[I] Building and running a GPU inference engine for MNIST
	[I] Input:
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@#-:.-=@@@@@@@@@@@@@@
	@@@@@%=     . *@@@@@@@@@@@@@
	@@@@% .:+%%%  *@@@@@@@@@@@@@
	@@@@+=#@@@@@# @@@@@@@@@@@@@@
	@@@@@@@@@@@%  @@@@@@@@@@@@@@
	@@@@@@@@@@@: *@@@@@@@@@@@@@@
	@@@@@@@@@@- .@@@@@@@@@@@@@@@
	@@@@@@@@@:  #@@@@@@@@@@@@@@@
	@@@@@@@@:   +*%#@@@@@@@@@@@@
	@@@@@@@%         :+*@@@@@@@@
	@@@@@@@@#*+--.::     +@@@@@@
	@@@@@@@@@@@@@@@@#=:.  +@@@@@
	@@@@@@@@@@@@@@@@@@@@  .@@@@@
	@@@@@@@@@@@@@@@@@@@@#. #@@@@
	@@@@@@@@@@@@@@@@@@@@#  @@@@@
	@@@@@@@@@%@@@@@@@@@@- +@@@@@
	@@@@@@@@#-@@@@@@@@*. =@@@@@@
	@@@@@@@@ .+%%%%+=.  =@@@@@@@
	@@@@@@@@           =@@@@@@@@
	@@@@@@@@*=:   :--*@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	[I] Output:
	Prob 1  0.0000 Class 1:
	Prob 2  0.0000 Class 2:
	Prob 3  1.0000 Class 3: **********
	Prob 4  0.0000 Class 4:
	Prob 5  0.0000 Class 5:
	Prob 6  0.0000 Class 6:
	Prob 7  0.0000 Class 7:
	Prob 8  0.0000 Class 8:
	Prob 9  0.0000 Class 9:

	&&&& PASSED TensorRT.sample_algorithm_selector # ./sample_algorithm_selector
	```

	This output shows that the sample ran successfully; `PASSED`.


### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
Usage: ./sample_algorithm_selector [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]
--help Display help information
--datadir Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)
--useDLACore=N Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform.
--int8 Run in Int8 mode.
--fp16 Run in FP16 mode.
```

# Additional resources

The following resources provide a deeper understanding about sampleAlgorithmSelector:

**MNIST**
- [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

**August 2022**
- Migrated code from parsing a `caffe` model to an `onnx` model.

**November 2019**
- This `README.md` file was recreated, updated and reviewed.

# Known issues

There are no known issues in this sample.
