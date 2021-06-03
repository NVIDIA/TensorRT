# TensorRT Inference of ONNX models with custom layers.

**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
    * [Cloning the packnet repository](#cloning-the-packnet-repository)
    * [Conversion to ONNX](#conversion-to-onnx)
    * [Inference with TensorRT](#inference-with-tensorrt)
	* [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, samplePackNet, is a Python sample which uses TensorRT to perform inference with PackNet network. PackNet is a self-supervised monocular depth estimation network used in autonomous driving.


## How does this sample work?

This sample converts the Pytorch graph into ONNX and uses ONNX-parser included in TensorRT to parse the ONNX graph. The sample also demonstrates

* Use of custom layers (plugins) in ONNX graph. These plugins would be automatically registered in TensorRT by using `REGISTER_TENSORRT_PLUGIN` API.
* Use of ONNX-graphsurgeon (ONNX-GS) API to modify layers or subgraphs in the ONNX graph. For this network, we transform Group Normalization, upsample and pad layers to remove unnecessary
  nodes for inference with TensorRT.


## Prerequisites

Dependencies required for this sample

1. Install the dependencies for Python:
   ```bash
   python3 -m pip install -r requirements.txt
   ```

    * NOTE: The included scripts do not currently work with Torch 1.5.0. Thus, it is recommended that you use 1.4.0, which is the default version in the requirements file.

2. Install [ONNX-GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon)
   ```bash
   pip3 install --no-cache-dir --extra-index-url https://pypi.ngc.nvidia.com onnx-graphsurgeon
   ```

## Running the sample

### Cloning the packnet repository

Clone the <a href="https://github.com/TRI-ML/packnet-sfm">packnet</a> repository and set `PYTHONPATH` variable accordingly.

   ```bash
   git clone https://github.com/TRI-ML/packnet-sfm.git packnet-sfm --depth 1 --branch v0.1.2
   export PYTHONPATH=$PYTHONPATH:$PWD/packnet-sfm
   ```

### Conversion to ONNX
Run the following command to convert the Packnet pytorch network to ONNX graph. This step also includes handling custom layers (Group Normalization) and using ONNX-GS to modify upsample and pad layers.

   ```bash
   python3 convert_to_onnx.py --output model.onnx
   ```

### Inference with TensorRT

Once the ONNX graph is generated, use `trtexec` tool (located in `bin` directory of TensorRT package) to perform inference on a random input image.

   ```bash
   trtexec --onnx=model.onnx --explicitBatch
   ```

Please refer to [trtexec documentation](https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/trtexec) for detailed usage options.

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
    ```bash
    convert_to_onnx.py -h
    ```

# Additional resources

The following resources provide a deeper understanding about PackNet network and importing a model into TensorRT using Python:

**PackNet**
- [3D Packing for Self-Supervised Monocular Depth Estimation](https://arxiv.org/pdf/1905.02693.pdf)
- [TRI-ML Monocular Depth Estimation Repository](https://github.com/TRI-ML/packnet-sfm)

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

June 2020: Initial release of this sample

# Known issues

There are no known issues in this sample
