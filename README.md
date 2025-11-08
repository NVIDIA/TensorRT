[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Documentation](https://img.shields.io/badge/TensorRT-documentation-brightgreen.svg)](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html) [![Roadmap](https://img.shields.io/badge/Roadmap-Q1_2025-brightgreen.svg)](documents/tensorrt_roadmap_2025q1.pdf)

# TensorRT Open Source Software

This repository contains the Open Source Software (OSS) components of NVIDIA TensorRT. It includes the sources for TensorRT plugins and ONNX parser, as well as sample applications demonstrating usage and capabilities of the TensorRT platform. These open source software components are a subset of the TensorRT General Availability (GA) release with some extensions and bug-fixes.

- For code contributions to TensorRT-OSS, please see our [Contribution Guide](CONTRIBUTING.md) and [Coding Guidelines](CODING-GUIDELINES.md).
- For a summary of new additions and updates shipped with TensorRT-OSS releases, please refer to the [Changelog](CHANGELOG.md).
- For business inquiries, please contact [researchinquiries@nvidia.com](mailto:researchinquiries@nvidia.com)
- For press and other inquiries, please contact Hector Marinez at [hmarinez@nvidia.com](mailto:hmarinez@nvidia.com)

Need enterprise support? NVIDIA global support is available for TensorRT with the [NVIDIA AI Enterprise software suite](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/). Check out [NVIDIA LaunchPad](https://www.nvidia.com/en-us/launchpad/ai/ai-enterprise/) for free access to a set of hands-on labs with TensorRT hosted on NVIDIA infrastructure.

Join the [TensorRT and Triton community](https://www.nvidia.com/en-us/deep-learning-ai/triton-tensorrt-newsletter/) and stay current on the latest product updates, bug fixes, content, best practices, and more.

# Prebuilt TensorRT Python Package

We provide the TensorRT Python package for an easy installation. \
To install:

```bash
pip install tensorrt
```

You can skip the **Build** section to enjoy TensorRT with Python.

# Build

## Prerequisites

To build the TensorRT-OSS components, you will first need the following software packages.

**TensorRT GA build**

- TensorRT v10.14.1.48
  - Available from direct download links listed below

**System Packages**

- [CUDA](https://developer.nvidia.com/cuda-toolkit)
  - Recommended versions:
  - cuda-13.0.0
  - cuda-12.9.0
- [CUDNN (optional)](https://developer.nvidia.com/cudnn)
  - cuDNN 8.9
- [GNU make](https://ftp.gnu.org/gnu/make/) >= v4.1
- [cmake](https://github.com/Kitware/CMake/releases) >= v3.31
- [python](https://www.python.org/downloads/) >= v3.10, <= v3.13.x
- [pip](https://pypi.org/project/pip/#history) >= v19.0
- Essential utilities
  - [git](https://git-scm.com/downloads), [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/), [wget](https://www.gnu.org/software/wget/faq.html#download)

**Optional Packages**

- Containerized build
  - [Docker](https://docs.docker.com/install/) >= 19.03
  - [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- PyPI packages (for demo applications/tests)
  - [onnx](https://pypi.org/project/onnx/)
  - [onnxruntime](https://pypi.org/project/onnxruntime/)
  - [tensorflow-gpu](https://pypi.org/project/tensorflow/) >= 2.5.1
  - [Pillow](https://pypi.org/project/Pillow/) >= 9.0.1
  - [pycuda](https://pypi.org/project/pycuda/) < 2021.1
  - [numpy](https://pypi.org/project/numpy/)
  - [pytest](https://pypi.org/project/pytest/)
- Code formatting tools (for contributors)

  - [Clang-format](https://clang.llvm.org/docs/ClangFormat.html)
  - [Git-clang-format](https://github.com/llvm-mirror/clang/blob/master/tools/clang-format/git-clang-format)

  > NOTE: [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt), [cub](http://nvlabs.github.io/cub/), and [protobuf](https://github.com/protocolbuffers/protobuf.git) packages are downloaded along with TensorRT OSS, and not required to be installed.

## Downloading TensorRT Build

1. #### Download TensorRT OSS

   ```bash
   git clone -b main https://github.com/nvidia/TensorRT TensorRT
   cd TensorRT
   git submodule update --init --recursive
   ```

2. #### (Optional - if not using TensorRT container) Specify the TensorRT GA release build path

   If using the TensorRT OSS build container, TensorRT libraries are preinstalled under `/usr/lib/x86_64-linux-gnu` and you may skip this step.

   Else download and extract the TensorRT GA build from [NVIDIA Developer Zone](https://developer.nvidia.com) with the direct links below:

   - [TensorRT 10.14.1.48 for CUDA 13.0, Linux x86_64](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/tars/TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-13.0.tar.gz)
   - [TensorRT 10.14.1.48 for CUDA 12.9, Linux x86_64](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/tars/TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-12.9.tar.gz)
   - [TensorRT 10.14.1.48 for CUDA 13.0, Windows x86_64](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/zip/TensorRT-10.14.1.48.Windows.win10.cuda-13.0.zip)
   - [TensorRT 10.14.1.48 for CUDA 12.9, Windows x86_64](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/zip/TensorRT-10.14.1.48.Windows.win10.cuda-12.9.zip)

   **Example: Ubuntu 22.04 on x86-64 with cuda-13.0**

   ```bash
   cd ~/Downloads
   tar -xvzf TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-13.0.tar.gz
   export TRT_LIBPATH=`pwd`/TensorRT-10.14.1.48
   ```

   **Example: Windows on x86-64 with cuda-12.9**

   ```powershell
   Expand-Archive -Path TensorRT-10.14.1.48.Windows.win10.cuda-12.9.zip
   $env:TRT_LIBPATH="$pwd\TensorRT-10.14.1.48\lib"
   ```

## Setting Up The Build Environment

For Linux platforms, we recommend that you generate a docker container for building TensorRT OSS as described below. For native builds, please install the [prerequisite](#prerequisites) _System Packages_.

1. #### Generate the TensorRT-OSS build container.

   **Example: Ubuntu 24.04 on x86-64 with cuda-13.0 (default)**

   ```bash
   ./docker/build.sh --file docker/ubuntu-24.04.Dockerfile --tag tensorrt-ubuntu24.04-cuda13.0
   ```

   **Example: Rockylinux8 on x86-64 with cuda-13.0**

   ```bash
   ./docker/build.sh --file docker/rockylinux8.Dockerfile --tag tensorrt-rockylinux8-cuda13.0
   ```

   **Example: Ubuntu 24.04 cross-compile for Jetson (aarch64) with cuda-13.0 (JetPack SDK)**

   ```bash
   ./docker/build.sh --file docker/ubuntu-cross-aarch64.Dockerfile --tag tensorrt-jetpack-cuda13.0
   ```

   **Example: Ubuntu 24.04 on aarch64 with cuda-13.0**

   ```bash
   ./docker/build.sh --file docker/ubuntu-24.04-aarch64.Dockerfile --tag tensorrt-aarch64-ubuntu24.04-cuda13.0
   ```

2. #### Launch the TensorRT-OSS build container.
   **Example: Ubuntu 24.04 build container**
   ```bash
   ./docker/launch.sh --tag tensorrt-ubuntu24.04-cuda13.0 --gpus all
   ```
   > NOTE:
   > <br> 1. Use the `--tag` corresponding to build container generated in Step 1.
   > <br> 2. [NVIDIA Container Toolkit](#prerequisites) is required for GPU access (running TensorRT applications) inside the build container.
   > <br> 3. `sudo` password for Ubuntu build containers is 'nvidia'.
   > <br> 4. Specify port number using `--jupyter <port>` for launching Jupyter notebooks.
   > <br> 5. Write permission to this folder is required as this folder will be mounted inside the docker container for uid:gid of 1000:1000.

## Building TensorRT-OSS

- Generate Makefiles and build

  **Example: Linux (x86-64) build with default cuda-13.0**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out
  make -j$(nproc)
  ```

  **Example: Linux (aarch64) build with default cuda-13.0**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64-native.toolchain
  make -j$(nproc)
  ```

  **Example: Native build on Jetson Thor (aarch64) with cuda-13.0**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DTRT_PLATFORM_ID=aarch64
  CC=/usr/bin/gcc make -j$(nproc)
  ```

  > NOTE: C compiler must be explicitly specified via CC= for native aarch64 builds of protobuf.

  **Example: Ubuntu 24.04 Cross-Compile for Jetson Thor (aarch64) with cuda-13.0 (JetPack)**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64_cross.toolchain
  make -j$(nproc)
  ```

  **Example: Ubuntu 24.04 Cross-Compile for DriveOS (aarch64) with cuda-13.0**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64_dos_cross.toolchain
  make -j$(nproc)
  ```

  **Example: Native builds on Windows (x86) with cuda-13.0**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build
  cd -p build
  cmake .. -DTRT_LIB_DIR="$env:TRT_LIBPATH" -DTRT_OUT_DIR="$pwd\\out"
  msbuild TensorRT.sln /property:Configuration=Release -m:$env:NUMBER_OF_PROCESSORS
  ```

  > NOTE: The default CUDA version used by CMake is 13.0. To override this, for example to 12.9, append `-DCUDA_VERSION=12.9` to the cmake command.

- Required CMake build arguments are:
  - `TRT_LIB_DIR`: Path to the TensorRT installation directory containing libraries.
  - `TRT_OUT_DIR`: Output directory where generated build artifacts will be copied.
- Optional CMake build arguments:
  - `CMAKE_BUILD_TYPE`: Specify if binaries generated are for release or debug (contain debug symbols). Values consists of [`Release`] | `Debug`
  - `CUDA_VERSION`: The version of CUDA to target, for example [`12.9.9`].
  - `CUDNN_VERSION`: The version of cuDNN to target, for example [`8.9`].
  - `PROTOBUF_VERSION`: The version of Protobuf to use, for example [`3.20.1`]. Note: Changing this will not configure CMake to use a system version of Protobuf, it will configure CMake to download and try building that version.
  - `CMAKE_TOOLCHAIN_FILE`: The path to a toolchain file for cross compilation.
  - `BUILD_PARSERS`: Specify if the parsers should be built, for example [`ON`] | `OFF`. If turned OFF, CMake will try to find precompiled versions of the parser libraries to use in compiling samples. First in `${TRT_LIB_DIR}`, then on the system. If the build type is Debug, then it will prefer debug builds of the libraries before release versions if available.
  - `BUILD_PLUGINS`: Specify if the plugins should be built, for example [`ON`] | `OFF`. If turned OFF, CMake will try to find a precompiled version of the plugin library to use in compiling samples. First in `${TRT_LIB_DIR}`, then on the system. If the build type is Debug, then it will prefer debug builds of the libraries before release versions if available.
  - `BUILD_SAMPLES`: Specify if the samples should be built, for example [`ON`] | `OFF`.
  - `GPU_ARCHS`: GPU (SM) architectures to target. By default we generate CUDA code for all major SMs. Specific SM versions can be specified here as a quoted space-separated list to reduce compilation time and binary size. Table of compute capabilities of NVIDIA GPUs can be found [here](https://developer.nvidia.com/cuda-gpus). Examples: - NVidia A100: `-DGPU_ARCHS="80"` - RTX 50 series: `-DGPU_ARCHS="120"` - Multiple SMs: `-DGPU_ARCHS="80 120"`
  - `TRT_PLATFORM_ID`: Bare-metal build (unlike containerized cross-compilation). Currently supported options: `x86_64` (default).

# References

## TensorRT Resources

- [TensorRT Developer Home](https://developer.nvidia.com/tensorrt)
- [TensorRT QuickStart Guide](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [TensorRT Sample Support Guide](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html)
- [TensorRT ONNX Tools](https://docs.nvidia.com/deeplearning/tensorrt/index.html#tools)
- [TensorRT Discussion Forums](https://devtalk.nvidia.com/default/board/304/tensorrt/)
- [TensorRT Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)

## Known Issues

- Please refer to [TensorRT Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes)
