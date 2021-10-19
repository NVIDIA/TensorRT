# TensorRT Engine Refitting of ONNX models.

**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample shows how to refit an engine built from an ONNX model via parsers. A modified version of the [ONNX BiDAF model](https://github.com/onnx/models/tree/master/text/machine_comprehension/bidirectional_attention_flow) is used as the sample model, which implements the Bi-Directional Attention Flow (BiDAF) network described in the paper [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603).

## How does this sample work?

This sample replaces unsupported nodes (HardMax / Compress) in the original ONNX model via ONNX-graphsurgeon (in `prepare_model.py`) and build a refittable TensorRT engine.
The engine is then refitted with fake weights and correct weights, each followed by inference on sample context and query sentences in `build_and_refit_engine.py`.

## Prerequisites

Dependencies required for this sample

1. Install the dependencies for Python:
```bash
pip3 install -r requirements.txt
```

On Jetson Nano, you will need nvcc in the `PATH` for installing pycuda:
```bash
export PATH=${PATH}:/usr/local/cuda/bin/
```

2. TensorRT

3. [ONNX-GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)

4. Download sample data. See the "Download Sample Data" section of [the general setup guide](../README.md).

## Running the sample

The data directory needs to be specified (either via `-d /path/to/data` or environment varaiable `TRT_DATA_DIR`)
when running these scripts. An error will be thrown if not. Taking `TRT_DATA_DIR` approach in following example.

* Prepare the ONNX model. (The data directory needs to be specified.)
  ```bash
  python3 prepare_model.py
  ```

The output should look similar to the following:
```
Modifying the ONNX model ...
Modified ONNX model saved as bidaf-modified.onnx
Done.
```

The script will modify the original model from [onnx/models](https://github.com/onnx/models/raw/c02f8c8699fc12273649e658b8d2a1a8e32a35d0/text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9.onnx) and save an ONNX model that can be parsed and run by TensorRT.

The original ONNX model contains four CategoryMapper nodes to map the four input string arrays to int arrays.
Since TensorRT does not support string data type and CategoryMapper nodes, we dump out the four maps for the four nodes as json files (`model/CategoryMapper_{4-6}.json`) and use them to preprocess input data.
Now the four inputs become four outputs of the original CategoryMapper nodes.

And unsupported HardMax nodes and Compress nodes are replaced by ArgMax nodes and Gather nodes, respectively.


* Build a TensorRT engine, refit the engine and run inference.
`python3 build_and_refit_engine.py`

The script will build a TensorRT engine from the modified ONNX model, and then refit the engine and run inference on sample context and query sentences.

When running the above command for the first time, the output should look similar to the following:
```
Loading ONNX file from path bidaf-modified.onnx...
Beginning ONNX file parsing
[TensorRT] WARNING: onnx2trt_utils.cpp:283: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
[TensorRT] WARNING: Tensor DataType is determined at build time for tensors not marked as input or output.
[TensorRT] WARNING: Tensor DataType is determined at build time for tensors not marked as input or output.
Completed parsing of ONNX file
Network inputs:
CategoryMapper_4 <class 'numpy.int32'> (-1, 1)
CategoryMapper_5 <class 'numpy.int32'> (-1, 1, 1, 16)
CategoryMapper_6 <class 'numpy.int32'> (-1, 1)
CategoryMapper_7 <class 'numpy.int32'> (-1, 1, 1, 16)
Building an engine from file bidaf-modified.onnx; this may take a while...
Completed creating Engine
Refitting engine...
Doing inference...
Refitting engine...
Doing inference...
Passed
```

When running the above command again, engine will be deserialized from the plan file, the output should look similar to the following:
```
Reading engine from file bidaf.trt
Refitting engine...
Doing inference...
Refitting engine...
Doing inference...
Passed
```

# Additional resources

The following resources provide a deeper understanding about the model used in this sample:

**Model**
- [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [Importing A Model Using A Parser In Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_model_python)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

October 2020: This sample was recreated, updated and reviewed.

# Known issues

There are no known issues in this sample.
