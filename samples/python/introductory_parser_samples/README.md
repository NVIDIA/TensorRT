# Introduction To Importing Caffe, TensorFlow And ONNX Models Into TensorRT Using Python


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
	* [caffe_resnet50](#caffe_resnet50)
	* [onnx_resnet50](#onnx_resnet50)
	* [uff_resnet50](#uff_resnet50)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, introductory_parser_samples, is a Python sample which uses TensorRT and its included suite of parsers (the UFF, Caffe and ONNX parsers), to perform inference with ResNet-50 models trained with various different frameworks.

## How does this sample work?

This sample is a collection of three smaller samples, with each focusing on a specific parser. The following sections describe how each sample works.

### caffe_resnet50

This sample demonstrates how to build an engine from a trained Caffe model using the Caffe parser and then run inference. The Caffe parser is used for Caffe2 models. After training, you can invoke the Caffe parser directly on the model file (usually `.caffemodel`) and deploy file (usually `.prototxt`).

### onnx_resnet50

This sample demonstrates how to build an engine from an ONNX model file using the open-source ONNX parser and then run inference. The ONNX parser can be used with any framework that supports the ONNX format (typically `.onnx` files).

### uff_resnet50

This sample demonstrates how to build an engine from a UFF model file (converted from a TensorFlow protobuf) and then run inference. The UFF parser is used for TensorFlow models. After freezing a TensorFlow graph and writing it to a protobuf file, you can convert it to UFF with the `convert-to-uff` utility included with TensorRT. This sample ships with a pre-generated UFF file.

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
	`python3 <parser>_resnet50.py`

	Where `<parser>` is either `caffe`, `onnx`, or `uff`.

	**Note:** If the TensorRT sample data is not installed in the default location, for example `/usr/src/tensorrt/data/`, the `data` directory must be specified.
	`python3 <parser>_resnet50.py [-d DATA_DIR]`

	For example: `python3 caffe_resnet50.py -d /path/to/my/data/`

2.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	`Correctly recognized data/samples/resnet50/reflex_camera.jpeg as reflex camera`

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
usage: caffe_resnet50.py|uff_resnet50.py|onnx_resnet50.py [-h] [-d DATADIR]

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
- [Caffe Parser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Caffe/pyCaffe.html)
- [ONNX Parser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Onnx/pyOnnx.html)
- [UFF Parser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Uff/pyUff.html)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [Importing A Model Using A Parser In Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_model_python)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

February 2019
This `README.md` file was recreated, updated and reviewed.

# Known issues

There are no known issues in this sample
