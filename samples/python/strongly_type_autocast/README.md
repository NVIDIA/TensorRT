# Convert FP32 ONNX models to mixed FP32-FP16 precision for TensorRT strong typing usage

**Table Of Contents**

- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
    * [Data preparation and the original ONNX model verification](#data-preparation-and-the-original-onnx-model-verification)
    * [Model conversion and the converted ONNX model verification](#model-conversion-and-the-converted-onnx-model-verification)
    * [Build and verify TensorRT engine from the converted ONNX model](#build-and-verify-tensorrt-engine-from-the-converted-onnx-model)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
    * [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, `strongly_type_autocast`, uses ModelOpt's AutoCast tool to convert a FP32 ONNX model to mixed FP32-FP16 precision, and builds engine / runs inference with TensorRT's strong typing mode.

[AutoCast](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/8_autocast.html) is a tool for converting FP32 ONNX models to mixed precision FP32-FP16 or FP32-BF16 models. AutoCast intelligently selects nodes to keep in FP32 precision to maintain model accuracy while benefiting from reduced precision on the rest of the nodes. AutoCast automatically injects cast operations around the selected nodes.

[Strong Typing vs Weak Typing](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/capabilities.html#strong-vs-weak-typing)
For strong typing, TensorRT adheres strictly to the type semantics in ONNX frameworks. For weak typing, TensorRT may substitute different precisions for tensors if it increases performance. Weak typing has been deprecated in 10.12. We recommend using AutoCast tool to convert the FP32 ONNX model to mixed precision before doing TensorRT strong typing optimization.

## How does this sample work?

This sample consists of three stages:
- [Data preparation and the original ONNX model verification](#data-preparation-and-the-original-onnx-model-verification)
- [Model conversion and the converted ONNX model verification](#model-conversion-and-the-converted-onnx-model-verification)
- [Build and verify TensorRT engine from the converted ONNX model](#build-and-verify-tensorrt-engine-from-the-converted-onnx-model)

### Data preparation and the original ONNX model verification

The original input data is in pgm format, including ten pictures of numbers from 0 to 9. The input data need to be transformed to npz format.

The original ONNX model is in fp32 precision. To verify the original model, use ONNX Runtime to run inference, printing the predicted numbers and saving all model outputs as gold references.

### Model conversion and the converted ONNX model verification

The original fp32 ONNX model can be converted to fp32-fp16 mixed precision using ModelOpt's AutoCast python package:

    ```
    from modelopt.onnx.autocast import convert_to_mixed_precision

    converted_model = convert_to_mixed_precision(
        onnx_path="model.onnx",
        low_precision_type="fp16",            # or "bf16"
        nodes_to_exclude=None,                # optional list of node name patterns to keep in FP32
        op_types_to_exclude=None,             # optional list of op types to keep in FP32
        data_max=512,                         # maximum absolute I/O values for nodes to convert
        init_max=65504,                       # maximum absolute I/O values for initializers to convert
        keep_io_types=False,                  # whether to preserve input/output types
        calibration_data=None,                # optional path to input data file
    )
    ```  

After the fp32-fp16 mixed precision model generated, use ONNX Runtime to run inference on the converted ONNX model, printing the predicted numbers and comparing the model outputs with gold references.

### Build and verify TensorRT engine from the converted ONNX model

Build TensorRT engine from the converted ONNX model, enabling strong type through `NetworkDefinitionCreationFlag.STRONGLY_TYPED`:

    ```
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    ```

Run inference on the generated trt engine, printing the predicted numbers and comparing the model outputs with gold references.

## Prerequisites

Dependencies required for this sample

1. Install the dependencies for Python:
    ```bash
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
    ```

2. TensorRT

3. The MNIST dataset can be found under the data directory (usually `/usr/src/tensorrt/data/mnist`) if using the TensorRT containers. It is also bundled along with the [TensorRT tarball](https://developer.nvidia.com/nvidia-tensorrt-download).

## Running the sample

1.  Run the sample:
    `python3 sample.py [--mnist_dir] [--working_dir]`

2.  Verify that the sample ran successfully. If the sample runs successfully you should see the following message:
    ```
    Sample finished successfully.
    ```

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.

# Additional resources

The following resources provide a deeper understanding about AutoCast and TensorRT strong typing:

**Documentation**
- [Guide of TensorRT-Model-Optimizer Autocast](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/8_autocast.html)
- [TensorRT Strong Typing vs Weak Typing](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/capabilities.html#strong-vs-weak-typing)
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

September 2025
This is the first version of this `README.md` file.

# Known issues

There are no known issues in this sample.
