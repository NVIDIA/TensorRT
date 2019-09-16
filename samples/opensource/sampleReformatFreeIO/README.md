# Specifying I/O Formats Using The Reformat Free I/O APIs


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleReformatFreeIO, uses a Caffe model that was trained on the [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md) and performs engine building and inference using TensorRT. The correctness of outputs is then compared to the golden reference. Specifically, it shows how to use reformat free I/O APIs to explicitly specify I/O formats to `TensorFormat::kLINEAR`, `TensorFormat::kCHW2` and `TensorFormat::kHWC8` for Float16 and INT8 precision.

## How does this sample work?

`ITensor::setAllowedFormats` is invoked to specify which format is expected to be supported so that the unnecessary reformatting will not be inserted to convert from/to FP32 formats for I/O tensors. `BuilderFlag::kSTRICT_TYPES` is also assigned to the builder configuration to let the builder choose a reformat free path rather than the fastest path.

**Note:** If the reformat free path is not implemented, then the fastest path with reformatting will be selected with the following warning message:
`Warning: no implementation obeys reformatting-free rules, ....`

	```
	bool SampleReformatFreeIO::build(int dataWidth)
	{
		...

		network->getInput(0)->setAllowedFormats(static_cast<TensorFormats>(1 << static_cast<int>(mTensorFormat)));
		network->getOutput(0)->setAllowedFormats(static_cast<TensorFormats>(1 << static_cast<int>(mTensorFormat)));
		...
		config->setFlag(BuilderFlag::kSTRICT_TYPES);
		...
	}
	```

## Running the sample

1.  Compile this sample by running `make` in the `<TensorRT root directory>/samples/sampleReformatFreeIO` directory. The binary named `sample_reformat_free_io` will be created in the `<TensorRT root directory>/bin` directory.
	```
	cd <TensorRT root directory>/samples/sampleReformatFreeIO
	make
	```

	Where `<TensorRT root directory>` is where you installed TensorRT.

2.  Run inference on the digit looping from 0 to 9:
    `./sample_reformat_free_io`

3.  Verify that all 10 digits match correctly. If the sample runs successfully, you should see output similar to the following:
	```
	&&&& RUNNING TensorRT.sample_reformat_free_io # ./sample_reformat_free_io
	[I] The test chooses MNIST as the network and recognizes a randomly generated digit
	[I] Firstly it runs the FP32 as the golden data, then INT8/FP16 with different formats will be tested
	[I]
	[I] Building and running a FP32 GPU inference to get golden input/output
	[I] [TRT] Detected 1 input and 1 output network tensors.
	[I] Input:
	... (omitted message)
	&&&& PASSED TensorRT.sample_reformat_free_io
	```
	This output shows that the sample ran successfully; `PASSED`.

### Sample `--help` options

To see the full list of available options and their descriptions, use the `./sample_reformat_free_io --help` command.

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

June 2019
This is the first release of the `README.md` file and sample.


## Known issues

There are no known issues in this sample.
