# Adding A Custom Layer Implementation to Your ONNX Network

**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
- [Prerequisites](#prerequisites)
- [Download and preprocess the ONNX model](#download-the-onnx-model)
- [Running the sample](#running-the-sample)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, `onnx_custom_plugin`, demonstrates how to use plugins written in C++ with the TensorRT Python bindings and ONNX Parser. This sample uses the [BiDAF Model](https://github.com/onnx/models/tree/main/text/machine_comprehension/bidirectional_attention_flow) from ONNX Model Zoo.

## How does this sample work?

This sample implements a Hardmax layer using cuBLAS, wraps the implementation in a TensorRT plugin (with a corresponding plugin creator) and then generates a shared library module containing its code. The user then dynamically loads this library in Python, which causes the plugin to be registered in TensorRT's PluginRegistry and makes it available to the ONNX parser.

This sample includes:

`plugin/`
This directory contains files for the Hardmax layer plugin.

`customHardmaxPlugin.cpp`
A custom TensorRT plugin implementation.

`customHardmaxPlugin.h`
The Hardmax Plugin headers.

`model.py`
This script downloads the BiDAF onnx model and uses Onnx Graphsurgeon to replace layers unsupported by TensorRT.

`sample.py`
This script loads the ONNX model and performs inference using TensorRT.

`load_plugin_lib.py`
This script contains a helper function to load the customHardmaxPlugin library in Python.

`test_custom_hardmax_plugin.py`
This script tests the Hardmax Plugin against a reference numpy implementation.

`requirements.txt`
This file specifies all the Python packages required to run this Python sample.

## Prerequisites

For specific software versions, see the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html).

1. Install the dependencies for Python.

```bash
pip3 install -r requirements.txt
```

On Jetson Nano, you will need nvcc in the `PATH` for installing pycuda:
```bash
export PATH=${PATH}:/usr/local/cuda/bin/
```

2. [Install CMake](https://cmake.org/download/).

3. (For Windows builds) [Visual Studio](https://visualstudio.microsoft.com/vs/older-downloads/) 2017 Community or Enterprise edition

## Download and preprocess the ONNX model

Run the model script to download the BiDAF model from the Onnx Model Zoo. The script will replace the `Hardmax` layer with an op called `CustomHardmax` to match the custom Plugin name. It will also replace the unsupported `Compress` node with an equivalent operation, and remove the `CategoryMapper` nodes which do a String-to-Int conversion of the model inputs.

```bash
python3 model.py
```

## Running the sample

1.  Build the plugin and its corresponding Python bindings.

   - On Linux, run:
      ```bash
      mkdir build && pushd build
      cmake .. && make -j
      popd
      ```

      **NOTE:** If any of the dependencies are not installed in their default locations, you can manually specify them. For example:
      ```bash
      cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-x.x/bin/nvcc # (Or adding /path/to/nvcc into $PATH)
               -DCUDA_INC_DIR=/usr/local/cuda-x.x/include/  # (Or adding /path/to/cuda/include into $CPLUS_INCLUDE_PATH)
               -DTRT_LIB=/path/to/tensorrt/lib/
               -DTRT_INCLUDE=/path/to/tensorrt/include/
      ```

   - On Windows, run the following in Powershell, replacing paths appropriately:
      ```ps1
      mkdir build; pushd build
      cmake .. -G "Visual Studio 15 Win64" /
         -DTRT_LIB=C:\path\to\tensorrt\lib /
         -DTRT_INCLUDE=C:\path\to\tensorrt\lib /
         -DCUDA_INC_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<CUDA_VERSION>\include" /
         -DCUDA_LIB_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<CUDA_VERSION>\lib\x64"
      # NOTE: msbuild is usually located under C:\Program Files (x86)\Microsoft Visual Studio\2017\<EDITION>\MSBuild\<VERSION>\Bin
      #   You should add this path to your PATH environment variable.
      msbuild ALL_BUILD.vcxproj
      popd
      ```

   The command `cmake ..` displays a complete list of configurable variables. If a variable is set to `VARIABLE_NAME-NOTFOUND`, then you’ll need to specify it manually or set the variable it is derived from correctly.

2.  Run inference using TensorRT with the custom Hardmax plugin implementation:
   ```bash
   python3 sample.py
   ```

3.  Verify that the sample ran successfully.
   ```
   === Testing ===

   Input context: Garry the lion is 5 years old. He lives in the savanna.
   Input query: Where does the lion live?
   Model prediction:  savanna

   Input context: A quick brown fox jumps over the lazy dog.
   Input query: What color is the fox?
   Model prediction:  brown   
   ```

   The model can also be run interactively:
   ```bash
   python3 sample.py --interactive
   ```

   The context and query can then be entered from the command line:

   ```
   === Testing ===
   Enter context: Waldo wears a striped shirt. He also wears glasses.
   Enter query: Who wears glasses?
   Model prediction:  waldo
   ```

# Additional resources

The following resources provide a deeper understanding about getting started with TensorRT using Python:

**Model**
- [BiDAF model](https://allenai.github.io/bi-att-flow/)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

September 2022
This `README.md` file was created and reviewed.

# Known issues

There are no known issues in this sample.
