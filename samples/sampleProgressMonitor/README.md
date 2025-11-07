# Progress Monitor API usage example based off sampleMNIST in TensorRT

**Table Of Contents**

- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
    - [Progress bar display](#progress-bar-display)
- [Preparing sample data](#preparing-sample-data)
- [Running the sample](#running-the-sample)
	- [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleProgressMonitor, shows an example of how to use the progress monitor API based on sampleOnnxMNIST ([documentation](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html#onnx_mnist_sample)).

This sample demonstrates the usage of `IProgressMonitor` to report the status of TRT engine-building operations.

## How does this sample work?

This sample uses a Onnx model that was trained on the [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md).

Specifically, this sample performs the following steps:
- Performs the basic setup and initialization of TensorRT using the Onnx parser
- [Imports a trained Onnx model using Onnx parser](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#import_onnx_c)
- Preprocesses the input and stores the result in a managed buffer
- Builds an engine using incremental progress reporting
- [Serializes and deserializes the engines](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#serial_model_c)
- [Uses the engines to perform inference on an input image](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_c)

To verify whether the engine is operating correctly, this sample picks a 28x28 image of a digit at random and runs inference on it using the engine it created. The output of the network is a probability distribution on the digit, showing which digit is likely to be that in the image.

### Progress bar display

This sample implements an `IProgressMonitor` to display progress while building a TensorRT engine. Each long-running step of the process can define a new progress phase, nesting them as necessary.
1. Phase entry - The `IProgressMonitor::phaseBegin` callback determines an appropriate nesting level for the new phase and updates the terminal display.
2. Phase progress - The `IProgressMonitor::stepComplete` callback increments the progress bar for the selected phase and updates the terminal display. This sample always returns `true` from `stepComplete` in order to progress the build unconditionally. If you wish to cancel a build in progress, such as in response to user input, you can return `false` from this function to stop the build early.
3. Phase completion - The `IProgressMonitor::phaseEnd` callback removes the line corresponding to the completed phase and updates the terminal display.

The progress bars are drawn using virtual terminal escape sequences to manipulate the terminal's cursor and clear lines.

## Prerequisites
1. Preparing sample data
See [Preparing sample data](../README.md#preparing-sample-data) in the main samples README.

## Running the sample

1. Compile the sample by following build instructions in [TensorRT README](https://github.com/NVIDIA/TensorRT/).

2. Run the sample to perform inference on the digit:
    ```bash
    ./sample_progress_monitor [-h] [--datadir=/path/to/data/dir/] [--useDLA=N] [--fp16 or --int8]
    ```

    For example:
    ```bash
    ./sample_progress_monitor --datadir $TRT_DATADIR/mnist --fp16
    ```

	This sample reads the `mnist.onnx` file to build the network:

	This sample can be run in FP16 and INT8 modes as well.

	**Note:** By default, the sample expects these files to be in either the `data/samples/mnist/` or `data/mnist/` directories. The list of default directories can be changed by adding one or more paths with `--datadir=/new/path/` as a command line argument.

	**Note:** The sample should be run from a terminal. It uses xterm-style escape sequences to animate its output, and is not designed to be redirected to a file.

3.  Verify that the sample ran successfully. If the sample runs successfully you should see animated progress bars during the network build phase and output similar to the following:
    ```
	&&&& RUNNING TensorRT.sample_progress_monitor [TensorRT v8700] # ./sample_progress_monitor
	[I] Building and running a GPU inference engine for MNIST.
	[I] [TRT] [MemUsageChange] Init CUDA: CPU +14, GPU +0, now: CPU 19, GPU 1217 (MiB)
	[I] [TRT] [MemUsageChange] Init builder kernel library: CPU +1450, GPU +266, now: CPU 1545, GPU 1483 (MiB)
	[I] [TRT] ----------------------------------------------------------------
	[I] [TRT] Input filename:   ../../../../data/samples/mnist/mnist.onnx
	[I] [TRT] ONNX IR version:  0.0.3
	[I] [TRT] Opset version:    8
	[I] [TRT] Producer name:    CNTK
	[I] [TRT] Producer version: 2.5.1
	[I] [TRT] Domain:           ai.cntk
	[I] [TRT] Model version:    1
	[I] [TRT] Doc string:       
	[I] [TRT] ----------------------------------------------------------------
	[W] [TRT] onnx2trt_utils.cpp:374: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
	[I] [TRT] Graph optimization time: 0.00293778 seconds.
	[I] [TRT] Local timing cache in use. Profiling results in this builder pass will not be stored.
	[=======---] Building engine 3/4
	 [----------] Building engine from subgraph 0/1
	  [----------] Computing profile costs 0/1
	   [=======---] Timing graph nodes 11/15
	    [===-------] Finding fastest tactic for Times212 12/37
	     [==========] Measuring tactic time 4/4
    ```
    After the TensorRT network has been constructed, you should see output similar to the following. An ASCII rendering of the input image with digit 3:
    ```
	&&&& RUNNING TensorRT.sample_progress_monitor # ./sample_progress_monitor
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

	&&&& PASSED TensorRT.sample_progress_monitor # ./sample_progress_monitor
	```

	This output shows that the sample ran successfully; `PASSED`.


### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
Usage: ./sample_progress_monitor [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]
--help Display help information
--datadir Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)
--useDLACore=N Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform.
--int8 Run in Int8 mode.
--fp16 Run in FP16 mode.
```

# Additional resources

The following resources provide a deeper understanding about sampleProgressMonitor:

**MNIST**
- [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

**May 2023**
- This `README.md` file was created and reviewed.

# Known issues

There are no known issues in this sample.
