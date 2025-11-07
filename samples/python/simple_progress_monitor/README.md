# Introduction To IProgressMonitor Callbacks Using Python


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
	* [simple_progress_monitor](#simple_progress_monitor)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, simple_progress_monitor, is a Python sample which uses TensorRT and its included ONNX parser, to perform inference with ResNet-50 models saved in ONNX format. It displays animated progress bars while TensorRT builds the engine.

## How does this sample work?

### simple_progress_monitor

This sample demonstrates how to build an engine from an ONNX model file using the open-source ONNX parser and then run inference. The ONNX parser can be used with any framework that supports the ONNX format (typically `.onnx` files). An `IProgressMonitor` object receives updates on the progress of the build, and displays them as ASCII progress bars on stdout.

## Prerequisites

1. Install the dependencies for Python.

```bash
pip3 install -r requirements.txt
```
2. Preparing sample data
See [Preparing sample data](../../README.md#preparing-sample-data) in the main samples README.

## Running the sample

1.  Run the sample from a terminal to create a TensorRT inference engine and run inference:
	`python3 simple_progress_monitor.py`

	**Note:** If the TensorRT sample data is not installed in the default location, the `data` directory must be specified. For example: `python3 simple_progress_monitor.py -d $TRT_DATADIR`

	**Note:** Do not redirect the output of this script to a file or pipe.

2.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	`Correctly recognized data/samples/resnet50/reflex_camera.jpeg as reflex camera`

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
usage: simple_progress_monitor.py [-h] [-d DATADIR]

Runs a ResNet50 network with a TensorRT inference engine. Displays intermediate build progress.

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

**Terminal Escape Sequences**
- Linux: [XTerm Control Sequences](https://invisible-island.net/xterm/ctlseqs/ctlseqs.html)
- Windows: [Console Virtual Terminal Sequences](https://learn.microsoft.com/en-us/windows/console/console-virtual-terminal-sequences)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

October 2025
Migrate to strongly typed APIs.

August 2025
Removed support for Python versions < 3.10.

August 2023
Removed support for Python versions < 3.8.

June 2023
This `README.md` file was created and reviewed.

# Known issues

There are no known issues in this sample
