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

1. Upgrade pip version and install the sample dependencies.
    ```bash
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
    ```

On PowerPC systems, you will need to manually install PyTorch using IBM's [PowerAI](https://www.ibm.com/support/knowledgecenter/SS5SF7_1.6.0/navigation/pai_install.htm).

On Jetson Nano, you will need nvcc in the `PATH` for installing pycuda:
```bash
export PATH=${PATH}:/usr/local/cuda/bin/
```


## Running the sample

### Preparing packnet

Clone the [packnet](https://github.com/TRI-ML/packnet-sfm) repository and update `PYTHONPATH`.

```
git clone https://github.com/TRI-ML/packnet-sfm.git packnet-sfm
pushd packnet-sfm && git checkout tags/v0.1.2 && popd
export PYTHONPATH=$PWD/packnet-sfm # Note on Windows, the export command is: set PYTHONPATH=%cd%\packnet-sfm
```

### Conversion to ONNX
Run the following command to convert the Packnet pytorch network to ONNX graph. This step also includes handling custom layers (Group Normalization) and using ONNX-GS to modify upsample and pad layers.

```
python3 convert_to_onnx.py --output model.onnx
```

### Inference with TensorRT

Once the ONNX graph is generated, use `trtexec` tool (located in `bin` directory of TensorRT package) to perform inference on a random input image.

```
trtexec --onnx=model.onnx --explicitBatch
```

Please refer to `trtexec` tool for more commandline options.

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
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

August 2021: Update sample to work with latest torch version
June 2020: Initial release of this sample

# Known issues

There are no known issues in this sample
