# Adding A Custom Layer To Your TensorFlow Network In TensorRT In Python

**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
- [Generate the UFF model](#generate-the-uff-model)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, `uff_custom_plugin`, demonstrates how to use plugins written in C++ with the TensorRT Python bindings and UFF Parser. This sample uses the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## How does this sample work?

This sample implements a clip layer (as a CUDA kernel), wraps the implementation in a TensorRT plugin (with a corresponding plugin creator) and then generates a shared library module containing its code. The user then dynamically loads this library in Python, which causes the plugin to be registered in TensorRT's PluginRegistry and makes it available to the UFF parser.

This sample includes:
`plugin/`
This directory contains files for the Clip layer plugin.

`clipKernel.cu`
A CUDA kernel that clips input.

`clipKernel.h`
The header exposing the CUDA kernel to C++ code.

`customClipPlugin.cpp`
A custom TensorRT plugin implementation, which uses the CUDA kernel internally.

`customClipPlugin.h`
The ClipPlugin headers.

`lenet5.py`
This script trains an MNIST network.

`model.py`
This script converts the MNIST tensorflow model to UFF that replaces ReLU6 activation with the clip plugin.

`sample.py`
This script transforms the trained model into UFF (delegating ReLU6 activations to ClipPlugin instances) and runs inference in TensorRT.

`requirements.txt`
This file specifies all the Python packages required to run this Python sample.

## Generate the UFF model

1. If running this sample in a test container, launch [NVIDIA tf1 (Tensorflow 1.x)](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running) container in a separate terminal for generating the UFF model.
    ```bash
    docker run --rm -it --gpus all -v `pwd`:/workspace nvcr.io/nvidia/tensorflow:21.03-tf1-py3 /bin/bash
    ```

    Alternatively, install Tensorflow 1.15
    `pip3 install tensorflow>=1.15.5,<2.0`

  NOTE
  - On PowerPC systems, you will need to manually install TensorFlow using IBM's [PowerAI](https://www.ibm.com/support/knowledgecenter/SS5SF7_1.6.0/navigation/pai_install.htm).
  - On Jetson boards, you will need to manually install TensorFlow by following the documentation for [Xavier](https://docs.nvidia.com/deeplearning/dgx/install-tf-xavier/index.html) or [TX2](https://docs.nvidia.com/deeplearning/dgx/install-tf-jetsontx2/index.html).

2. Install the UFF toolkit and graph surgeon depending on your [TensorRT installation method](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing), or from PyPI:
    ```bash
    pip3 install --no-cache-dir --extra-index-url https://pypi.ngc.nvidia.com uff
    pip3 install --no-cache-dir --extra-index-url https://pypi.ngc.nvidia.com graphsurgeon
    ```

3. Run the sample to train the model, covert to UFF and save the model. Also save the test data:
    ```bash
    mkdir -p models
    python3 lenet5.py
    python3 model.py
    ```

## Prerequisites

For specific software versions, see the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html).

1. Switch back to test container (if applicable) and install the dependencies for Python.

```bash
pip3 install -r requirements.txt
```

On Jetson Nano, you will need nvcc in the `PATH` for installing pycuda:
```bash
export PATH=${PATH}:/usr/local/cuda/bin/
```

2. [Install CMake](https://cmake.org/download/).

3. (For Windows builds) [Visual Studio](https://visualstudio.microsoft.com/vs/older-downloads/) 2017 Community or Enterprise edition

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
         -DCUDA_INC_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<CUDA_VERSION>\include"
      # NOTE: msbuild is usually located under C:\Program Files (x86)\Microsoft Visual Studio\2017\<EDITION>\MSBuild\<VERSION>\Bin
      #   You should add this path to your PATH environment variable.
      msbuild ALL_BUILD.vcxproj
      popd
      ```

	`cmake ..` displays a complete list of configurable variables. If a variable is set to `VARIABLE_NAME-NOTFOUND`, then you’ll need to specify it manually or set the variable it is derived from correctly.

2.  Run inference using TensorRT with the custom clip plugin implementation:
   ```bash
   python3 sample.py
   ```

3.  Verify that the sample ran successfully. If the sample runs successfully you should see a match between the test case and the prediction.
   ```
   === Testing ===
   Loading Test Case: 3
   Prediction: 3
   ```

# Additional resources

The following resources provide a deeper understanding about getting started with TensorRT using Python:

**Model**
- [LeNet model](http://yann.lecun.com/exdb/lenet/)

**Dataset**
- [MNIST dataset](http://yann.lecun.com/exdb/mnist/)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

September 2021
Updated with instructions for building the plugin on Windows.

March 2019
This `README.md` file was recreated, updated and reviewed.

# Known issues

There are no known issues in this sample.
