# Introduction To Building and Refitting Weight-stripped Engines from ONNX Models


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sample_weight_stripping, is a Python sample which uses TensorRT to build a weight-stripped engine and later refit to a full engine for inference.

## How does this sample work?

This sample demonstrates how to build a weight-stripped engine from an ONNX model file using TensorRT Python API which can reduce the saved engine size. Later, the weight-stripped engine is refitted by parser refitter with the original ONNX model as input. The refitted full engine is used for inference and guarantees no performance and accuracy loss. In this sample, we use ResNet50 to showcase our features.

## Prerequisites

1. Install the dependencies for Python.

	```bash
	pip3 install -r requirements.txt
	```

2. Preparing sample data
See [Preparing sample data](../../README.md#preparing-sample-data) in the main samples README.

## Running the sample

1.  Build and save both normal engine and weight-stripped engine:

	```
	python3 build_engines.py --output_stripped_engine=stripped_engine.trt --output_normal_engine=normal_engine.trt
	```

	After running this step, you can see two saved TensorRT engines. `stripped_engine.trt` contains a stripped engine (~2.3MB) and `normal_engine.trt` contains a normal engine with all weights included (~51MB). By using stripped engine build, we can greatly reduce the size of the saved engine file.


	**Note:** If the TensorRT sample data is not installed in the default location, for example `/usr/src/tensorrt/data/`, the model directory must be specified. For example: `--stripped_onnx=/path/to/my/data/` sets the model path for building weight-stripped engine and `--original_onnx=/path/to/my/data/` sets the model path for building normal engine. In most of the cases, they can use the same ONNX model.

2.  Refit the weight-stripped engine and perform inference with the weight-stripped engine and the normal engine:
	```
	python3 refit_engine_and_infer.py --stripped_engine=stripped_engine.trt -–normal_engine=normal_engine.trt
	```
3.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following. The prediction results of the refitted stripped engine is the same as the normal engine. There is no performance loss.
	```
	Normal engine inference time on 100 cases: 0.1066 seconds
	Refitted stripped engine inference time on 100 cases: 0.0606 seconds
	Normal engine correctly recognized data/samples/resnet50/tabby_tiger_cat.jpg as tiger cat
	Refitted stripped engine correctly recognized data/samples/resnet50/tabby_tiger_cat.jpg as tiger cat
	```
### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.

# Additional resources

The following resources provide a deeper understanding about importing a model into TensorRT using Python:

**ResNet-50**
- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

August 2025
Removed support for Python versions < 3.10.

February 2024
Initial release of this sample.

# Known issues

There are no known issues in this sample.
