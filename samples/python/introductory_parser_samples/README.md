# Introduction To Importing ONNX Models Into TensorRT Using Python


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
	* [onnx_resnet50](#onnx_resnet50)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, introductory_parser_samples, is a Python sample which uses TensorRT and its included ONNX parser, to perform inference with ResNet-50 models saved in ONNX format.

## How does this sample work?

### onnx_resnet50

This sample demonstrates how to build an engine from an ONNX model file using the open-source ONNX parser and then run inference. The ONNX parser can be used with any framework that supports the ONNX format (typically `.onnx` files).

## Prerequisites

1. Install the dependencies for Python.

```bash
pip3 install -r requirements.txt
```

On Jetson Nano, you will need nvcc in the `PATH` for installing pycuda:
```bash
export PATH=${PATH}:/usr/local/cuda/bin/
```

## Running the sample

1.  Run the sample to create a TensorRT inference engine and run inference:
	`python3 onnx_resnet50.py`

	**Note:** If the TensorRT sample data is not installed in the default location, for example `/usr/src/tensorrt/data/`, the `data` directory must be specified. For example: `python3 onnx_resnet50.py -d /path/to/my/data/`

2.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	`Correctly recognized data/samples/resnet50/reflex_camera.jpeg as reflex camera`

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
usage: onnx_resnet50.py [-h] [-d DATADIR]

Runs a ResNet50 network with a TensorRT inference engine.

optional arguments:
 -h, --help            show this help message and exit
 -d DATADIR, --datadir DATADIR
                       Location of the TensorRT sample data directory.
                       (default: /usr/src/tensorrt/data)
```

# Additional resources

The following resources provide a deeper understanding about importing a model into TensorRT using Python:

**ResNet-50**
- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

**Parsers**
- [ONNX Parser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Onnx/pyOnnx.html)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [Importing A Model Using A Parser In Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_model_python)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

Auguest 2022
Removed options for Caffe and UFF parsers.

February 2019
This `README.md` file was recreated, updated and reviewed.

# Known issues

There are no known issues in this sample
