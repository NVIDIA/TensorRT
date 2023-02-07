# Specifying I/O Formats


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample-help-options)
- [Preparing sample data](#preparing-sample-data)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleIOFormats, uses a Onnx model that was trained on the [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md) and performs engine building and inference using TensorRT. The correctness of outputs is then compared to the golden reference. Specifically, it shows how to use APIs to explicitly specify input formats to `TensorFormat::kLINEAR` for Float32, and additionally `TensorFormat::kCHW2` and `TensorFormat::kHWC8` for Float16 and INT8 precision.

## How does this sample work?

`ITensor::setAllowedFormats` is invoked to specify which format is expected to be supported.

	```
	bool SampleIOFormats::build(int dataWidth)
	{
		...

		network->getInput(0)->setAllowedFormats(static_cast<TensorFormats>(1 << static_cast<int>(mTensorFormat)));
		...
	}
	```

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

2.  Run inference on the digit looping from 0 to 9:
    ```bash
    ./sample_io_formats --datadir=<path/to/data> --useDLACore=N
    ```

    For example:
    ```bash
    ./sample_io_formats --datadir $TRT_DATADIR/mnist
    ```

3.  Verify that all 10 digits match correctly. If the sample runs successfully, you should see output similar to the following:
	```
	&&&& RUNNING TensorRT.sample_io_formats # ./sample_io_formats
	[I] The test chooses MNIST as the network and recognizes a randomly generated digit
	[I] Firstly it runs the FP32 as the golden data, then INT8/FP16 with different formats will be tested
	[I]
	[I] Building and running a FP32 GPU inference to get golden input/output
	[I] [TRT] Detected 1 input and 1 output network tensors.
	[I] Input:
	... (omitted message)
	&&&& PASSED TensorRT.sample_io_formats
	```
	This output shows that the sample ran successfully; `PASSED`.


### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.


## Additional resources

The following resources provide a deeper understanding about this sample:

**Models**
- [MNIST](https://keras.io/datasets/#mnist-database-of-handwritten-digits)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


## Changelog

**August 2022**
- Migrated code from parsing a `caffe` model to an `onnx` model.

**Oct 2021**
- Change names and topic from "reformat-free" to "I/O formats", because `BuilderFlag::kSTRICT_TYPES`
is deprecated. "Reformat-free I/O" (see `BuilderFlag::kDIRECT_IO`) is generally counterproductive
and fragile, since it constrains the optimizer from choosing the fastest implementation,
and depends upon what kernels are available on a particular target.

**June 2019**
- This is the first release of the `README.md` file and sample.


## Known issues

There are no known issues in this sample.
