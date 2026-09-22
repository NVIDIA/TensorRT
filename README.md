[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Documentation](https://img.shields.io/badge/TensorRT-documentation-brightgreen.svg)](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html) [![Roadmap](https://img.shields.io/badge/Roadmap-Q3_2026-brightgreen.svg)](documents/tensorrt_roadmap_2026q3.pdf)

# :mega::mega: Announcement :mega::mega:

TensorRT 11.X is now released with powerful new capabilities designed to accelerate your AI inference workflows. With this major version bump, TensorRT's API has been streamlined and a few legacy features from 10.X have been removed.

Below provides migration guides for the following features:
- Weakly-typed networks and related APIs have been removed, replaced by [Strongly Typed Networks](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/advanced.html#strongly-typed-networks).
- Implicit quantization and related APIs have been removed, replaced by [Explicit Quantization](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html)
- IPluginV2 and related APIs have been removed, replaced by [IPluginV3](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html#migrating-v2-plugins-to-ipluginv3)
- TREX tool has been removed, replaced by [Nsight Deep Learning Designer](https://docs.nvidia.com/nsight-dl-designer/UserGuide/index.html#visualizing-a-tensorrt-engine)
- Python bindings for Python 3.9 and older versions have been removed. RPM packages for RHEL/Rocky Linux 8 and RHEL/Rocky Linux 9 now depend on Python 3.12.

# TensorRT Open Source Software

This repository contains the Open Source Software (OSS) components of NVIDIA TensorRT. It includes the sources for TensorRT plugins and ONNX parser, as well as sample applications demonstrating usage and capabilities of the TensorRT platform. These open source software components are a subset of the TensorRT General Availability (GA) release with some extensions and bug-fixes.

- For step-by-step walkthroughs of the TensorRT import paths (ONNX, Torch-TensorRT, HuggingFace/Optimum, Network Definition API) with examples and tooling tips, see the [Import Workflows Guide](documents/import_workflows.md).
- For the per-model support matrix across import paths (LLM, encoder-NLP, vision, audio, diffusion, multimodal), see [Supported Models](documents/supported_models.md).
- For code contributions to TensorRT-OSS, please see our [Contribution Guide](CONTRIBUTING.md) and [Coding Guidelines](CODING-GUIDELINES.md).
- For a summary of new additions and updates shipped with TensorRT-OSS releases, please refer to the [Changelog](CHANGELOG.md).
- For business inquiries, please contact [researchinquiries@nvidia.com](mailto:researchinquiries@nvidia.com)
- For press and other inquiries, please contact Hector Marinez at [hmarinez@nvidia.com](mailto:hmarinez@nvidia.com)

Need enterprise support? NVIDIA global support is available for TensorRT with the [NVIDIA AI Enterprise software suite](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/). Check out [NVIDIA LaunchPad](https://www.nvidia.com/en-us/launchpad/ai/ai-enterprise/) for free access to a set of hands-on labs with TensorRT hosted on NVIDIA infrastructure.

Join the [TensorRT and Triton community](https://www.nvidia.com/en-us/deep-learning-ai/triton-tensorrt-newsletter/) and stay current on the latest product updates, bug fixes, content, best practices, and more.

# Agentic Coding Skills
Various skills related to TensorRT usage and benchmarking are available [here](.agents/skills). For installation, refer to the instructions of your preferred coding agent.

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

- TensorRT v11.3.0.99
  - Available from direct download links listed below

**System Packages**

- [CUDA](https://developer.nvidia.com/cuda-toolkit)
  - Recommended versions:
  - cuda-13.4.0
  - cuda-12.9.0
- [CUDNN (optional)](https://developer.nvidia.com/cudnn)
  - cuDNN 8.9
- [GNU make](https://ftp.gnu.org/gnu/make/) >= v4.1
- [cmake](https://github.com/Kitware/CMake/releases) >= v3.31
- [python](https://www.python.org/downloads/) >= v3.10, <= v3.14.x
- [pip](https://pypi.org/project/pip/#history) >= v19.0
- Essential utilities
  - [git](https://git-scm.com/downloads), [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/), [wget](https://www.gnu.org/software/wget/faq.html#download)

**Optional Packages**

- [NCCL](https://developer.nvidia.com/nccl/nccl-download) >= v2.19, < v3.0 — only when building with multi-device support (`-DTRT_BUILD_ENABLE_MULTIDEVICE=ON`) for the `sampleDistCollective` sample.
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

   - [TensorRT 11.3.0.99 for CUDA 13.4, Linux x86_64](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/11.3.0/tars/TensorRT-Enterprise-11.3.0.99-Linux-x86_64-cuda-13.4-Release-external.tar.zst)
   - [TensorRT 11.3.0.99 for CUDA 12.9, Linux x86_64](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/11.3.0/tars/TensorRT-Enterprise-11.3.0.99-Linux-x86_64-cuda-12.9-Release-external.tar.zst)
   - [TensorRT 11.3.0.99 for CUDA 13.4, Windows x86_64](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/11.3.0/zip/TensorRT-Enterprise-11.3.0.99-Windows-amd64-cuda-13.4-Release-external.zip)
   - [TensorRT 11.3.0.99 for CUDA 12.9, Windows x86_64](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/11.3.0/zip/TensorRT-Enterprise-11.3.0.99-Windows-amd64-cuda-12.9-Release-external.zip)

   **Example: Ubuntu 22.04 on x86-64 with cuda-13.4**

   ```bash
   cd ~/Downloads
   tar --zstd -xvf TensorRT-Enterprise-11.3.0.99-Linux-x86_64-cuda-13.4-Release-external.tar.zst
   export TRT_LIBPATH=`pwd`/TensorRT-11.3.0.99/lib
   ```

   **Example: Windows on x86-64 with cuda-12.9**

   ```powershell
   Expand-Archive -Path TensorRT-Enterprise-11.3.0.99-Windows-amd64-cuda-12.9-Release-external.zip
   $env:TRT_LIBPATH="$pwd\TensorRT-11.3.0.99\lib"
   ```

## Setting Up The Build Environment

For Linux platforms, we recommend that you generate a docker container for building TensorRT OSS as described below. For native builds, please install the [prerequisite](#prerequisites) _System Packages_.

1. #### Generate the TensorRT-OSS build container.

   **Example: Ubuntu 24.04 on x86-64 with cuda-13.4 (default)**

   ```bash
   ./docker/build.sh --file docker/ubuntu-24.04.Dockerfile --tag tensorrt-ubuntu24.04-cuda13.4
   ```

   **Example: Rockylinux8 on x86-64 with cuda-13.4**

   ```bash
   ./docker/build.sh --file docker/rockylinux8.Dockerfile --tag tensorrt-rockylinux8-cuda13.4
   ```

   **Example: Ubuntu 24.04 cross-compile for Jetson (aarch64) with cuda-13.4 (JetPack SDK)**

   ```bash
   ./docker/build.sh --file docker/ubuntu-cross-aarch64.Dockerfile --tag tensorrt-jetpack-cuda13.4
   ```

   **Example: Ubuntu 24.04 on aarch64 with cuda-13.4**

   ```bash
   ./docker/build.sh --file docker/ubuntu-24.04-aarch64.Dockerfile --tag tensorrt-aarch64-ubuntu24.04-cuda13.4
   ```

2. #### Launch the TensorRT-OSS build container.
   **Example: Ubuntu 24.04 build container**
   ```bash
   ./docker/launch.sh --tag tensorrt-ubuntu24.04-cuda13.4 --gpus all
   ```
   > NOTE:
   > <br> 1. Use the `--tag` corresponding to build container generated in Step 1.
   > <br> 2. [NVIDIA Container Toolkit](#prerequisites) is required for GPU access (running TensorRT applications) inside the build container.
   > <br> 3. `sudo` password for Ubuntu build containers is 'nvidia'.
   > <br> 4. Specify port number using `--jupyter <port>` for launching Jupyter notebooks.
   > <br> 5. Write permission to this folder is required as this folder will be mounted inside the docker container for uid:gid of 1000:1000.

## Building TensorRT-OSS

- Generate Makefiles and build

  **Example: Linux (x86-64) build with default cuda-13.4**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DCMAKE_PREFIX_PATH=$TRT_ROOT \
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=`pwd`/out \
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=`pwd`/out \
      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=`pwd`/out
  make -j$(nproc)
  ```

  **Example: Linux (aarch64) build with default cuda-13.4**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DCMAKE_PREFIX_PATH=$TRT_ROOT \
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=`pwd`/out \
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=`pwd`/out \
      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=`pwd`/out \
      -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64-native.toolchain
  make -j$(nproc)
  ```

  **Example: Native build on Jetson Thor (aarch64) with cuda-13.4**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DCMAKE_PREFIX_PATH=$TRT_ROOT \
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=`pwd`/out \
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=`pwd`/out \
      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=`pwd`/out
  CC=/usr/bin/gcc make -j$(nproc)
  ```

  > NOTE: C compiler must be explicitly specified via CC= for native aarch64 builds of protobuf.

  **Example: Ubuntu 24.04 Cross-Compile for Jetson Thor (aarch64) with cuda-13.4 (JetPack)**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DCMAKE_PREFIX_PATH=$TRT_ROOT -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64_cross.toolchain
  make -j$(nproc)
  ```

  **Example: Ubuntu 24.04 Cross-Compile for DriveOS (aarch64) with cuda-13.4**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DTRT_BUILD_PRODUCT=automotive -DCMAKE_PREFIX_PATH=$TRT_ROOT \
      -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64_cross.toolchain
  make -j$(nproc)
  ```

  **Example: Native builds on Windows (x86) with cuda-13.4**

  ```bash
  cd $TRT_OSSPATH
  New-Item -ItemType Directory -Path build
  cd build
  cmake .. -DCMAKE_PREFIX_PATH="$env:TRT_ROOT" `
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$pwd\\out" `
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$pwd\\out" `
      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY="$pwd\\out"
  msbuild TensorRT.sln /property:Configuration=Release -m:$env:NUMBER_OF_PROCESSORS
  ```

  > NOTE: The default CUDA version used by CMake is 13.4. To override this, for example to 12.9, append `-DCUDA_VERSION=12.9` to the cmake command.

- Required CMake build arguments are:
  - `CMAKE_PREFIX_PATH`: Path to the TensorRT package root containing the product CMake config.
- Optional CMake build arguments:
  - `CMAKE_RUNTIME_OUTPUT_DIRECTORY`, `CMAKE_LIBRARY_OUTPUT_DIRECTORY`, and `CMAKE_ARCHIVE_OUTPUT_DIRECTORY`: Native CMake output directories for executables, shared libraries, and archives. Set all three to the same path to collect generated build artifacts in one directory.
  - `CMAKE_BUILD_TYPE`: Specify if binaries generated are for release or debug (contain debug symbols). Values consists of [`Release`] | `Debug`
  - `CUDA_VERSION`: The version of CUDA to target, for example [`12.9.9`].
  - `CUDNN_VERSION`: The version of cuDNN to target, for example [`8.9`].
  - `PROTOBUF_VERSION`: The version of Protobuf to use, for example [`3.20.1`]. Note: Changing this will not configure CMake to use a system version of Protobuf, it will configure CMake to download and try building that version.
  - `CMAKE_TOOLCHAIN_FILE`: The path to a toolchain file for cross compilation.
  - `BUILD_PARSERS`: Specify if the parsers should be built, for example [`ON`] | `OFF`. If turned OFF, CMake uses the parser target imported from the TensorRT package config.
  - `BUILD_PLUGINS`: Specify if the plugins should be built, for example [`ON`] | `OFF`. If turned OFF, CMake uses the plugin targets imported from the TensorRT package config.
  - `BUILD_SAMPLES`: Specify if the samples should be built, for example [`ON`] | `OFF`.
  - `BUILD_SAFE_SAMPLES`: Specify if safety samples should be built, for example [`ON`] | `OFF`.
  - `TRT_SAFETY_INFERENCE_ONLY`: Specify if only build the safety inference components, for example [`ON`] | `OFF`. If turned ON, all other components will be turned OFF except `BUILD_SAFE_SAMPLES`.
  - `TRT_BUILD_PRODUCT`: Select the TensorRT product package to import: `enterprise`, `automotive`, or `safe_inference`. If omitted, the build will infer the product based on the values of `BUILD_SAFE_SAMPLES` and `TRT_SAFETY_INFERENCE_ONLY`.
  - `TRT_BUILD_ENABLE_MULTIDEVICE`: Enable the multi-device sample (`sampleDistCollective`). Use `-DTRT_BUILD_ENABLE_MULTIDEVICE=ON` to build it; requires [NCCL](https://developer.nvidia.com/nccl/nccl-download) >= v2.19, < v3.0.
  - `TRT_BUILD_TESTING` : Build gTests for samples. Requires [gtest](https://github.com/google/googletest) if available; otherwise fetches googletest at configure time.

## Building TensorRT DriveOS Samples

- Generate Makefiles and build

  **Example: Cross-Compile for DOS7 Linux (aarch64)**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DBUILD_SAMPLES=ON -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF \
      -DTRT_BUILD_PRODUCT=automotive \
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=`pwd`/bin_dynamic_cross \
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=`pwd`/bin_dynamic_cross \
      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=`pwd`/bin_dynamic_cross \
      -DCMAKE_PREFIX_PATH=$TRT_ROOT \
      -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64_cross.toolchain
  make -j$(nproc)
  ```

  **Example: Cross-Compile for DOS6.5 Linux (aarch64)**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DBUILD_SAMPLES=ON -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF \
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=`pwd`/bin_dynamic_cross \
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=`pwd`/bin_dynamic_cross \
      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=`pwd`/bin_dynamic_cross \
      -DCMAKE_PREFIX_PATH=$TRT_ROOT \
      -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64_cross.toolchain \
      -DCUDA_VERSION=11.4 -DCMAKE_CUDA_ARCHITECTURES=87
  make -j$(nproc)
  ```

  **Example: Native build for DOS6.5 and DOS7 Linux (aarch64)**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DCMAKE_PREFIX_PATH=$TRT_ROOT \
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=`pwd`/out \
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=`pwd`/out \
      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=`pwd`/out \
      -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64-native.toolchain \
      -DBUILD_SAMPLES=ON -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF \
      -DTRT_BUILD_PRODUCT=automotive
  make -j$(nproc)
  ```

  **Example: Cross-Compile for DOS6.5 QNX (aarch64)**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  export CUDA_VERSION=11.4
  export CUDA=cuda-$CUDA_VERSION
  export CUDA_ROOT=/usr/local/cuda-safe-$CUDA_VERSION
  export QNX_BASE=/drive/toolchains/qnx_toolchain  # Set to your QNX toolchain installation path
  export QNX_HOST=$QNX_BASE/host/linux/x86_64/
  export QNX_TARGET=$QNX_BASE/target/qnx7/
  export PATH=$PATH:$QNX_HOST/usr/bin
  cmake .. -DBUILD_SAMPLES=ON -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF -DBUILD_SAFE_SAMPLES=OFF \
      -DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc \
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=`pwd`/bin_dynamic_cross \
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=`pwd`/bin_dynamic_cross \
      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=`pwd`/bin_dynamic_cross \
      -DCMAKE_PREFIX_PATH=$TRT_ROOT \
      -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_qnx.toolchain \
      -DCUDA_VERSION=$CUDA_VERSION -DCMAKE_CUDA_ARCHITECTURES=87
  make -j$(nproc)
  ```

  > NOTE: Set `QNX_BASE` to your QNX toolchain installation path.
  > If your CUDA version is not the same as in the example, set `CUDA_VERSION` (for examples that use it in multiple places) or add `-DCUDA_VERSION=<version>` to the cmake command.

  **Example: Cross-Compile for DOS6.5 QNX Safety (aarch64)**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  export CUDA_VERSION=11.4
  export QNX_BASE=/drive/toolchains/qnx_toolchain  # Set to your QNX toolchain installation path
  export QNX_HOST=$QNX_BASE/host/linux/x86_64/
  export QNX_TARGET=$QNX_BASE/target/qnx7/
  export PATH=$PATH:$QNX_HOST/usr/bin
  export CUDA=cuda-$CUDA_VERSION
  export CUDA_ROOT=/usr/local/cuda-safe-$CUDA_VERSION
  cmake .. -DBUILD_SAMPLES=OFF -DBUILD_SAFE_SAMPLES=ON -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF \
      -DTRT_SAFETY_INFERENCE_ONLY=ON \
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=`pwd`/bin_dynamic_cross \
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=`pwd`/bin_dynamic_cross \
      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=`pwd`/bin_dynamic_cross \
      -DCMAKE_PREFIX_PATH=$TRT_ROOT \
      -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_qnx_safe.toolchain \
      -DCUDA_VERSION=$CUDA_VERSION -DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES=87
  make -j$(nproc)
  ```

  > NOTE: Set `QNX_BASE` to your QNX toolchain installation path.
  > If your CUDA version is not the same as in the example, set `CUDA_VERSION` (for examples that use it in multiple places) or add `-DCUDA_VERSION=<version>` to the cmake command.

  **Example: Cross-Compile for DOS7 QNX (aarch64)**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  export CUDA_VERSION=13.4
  export CUDA=cuda-$CUDA_VERSION
  export CUDA_ROOT=/usr/local/cuda-$CUDA_VERSION
  export QNX_BASE=/drive/toolchains/qnx_toolchain  # Set to your QNX toolchain installation path
  export QNX_HOST=$QNX_BASE/host/linux/x86_64/
  export QNX_TARGET=$QNX_BASE/target/qnx/
  export PATH=$PATH:$QNX_HOST/usr/bin
  cmake .. -DBUILD_SAMPLES=ON -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF -DBUILD_SAFE_SAMPLES=OFF \
      -DTRT_BUILD_PRODUCT=automotive \
      -DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc \
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=`pwd`/bin_dynamic_cross \
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=`pwd`/bin_dynamic_cross \
      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=`pwd`/bin_dynamic_cross \
      -DCMAKE_PREFIX_PATH=$TRT_ROOT \
      -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_qnx.toolchain \
      -DCUDA_VERSION=$CUDA_VERSION -DCMAKE_CUDA_ARCHITECTURES=110
  make -j$(nproc)
  ```

  > NOTE: Set `QNX_BASE` to your QNX toolchain installation path.
  > If your CUDA version is not the same as in the example, set `CUDA_VERSION` (for examples that use it in multiple places) or add `-DCUDA_VERSION=<version>` to the cmake command.

  **Example: Cross-Compile for DOS7 QNX Safety (aarch64)**

  DOS7 QNX and QNX Safety use the same QNX 8 SDK through `QNX_HOST` and
  `QNX_TARGET`. The QNX Safety build additionally uses SafeCUDA, the matching
  TensorRT SafeInference package, and `cmake_qnx_safe.toolchain`. The standard
  DriveOS QNX Safety build environment provides `PDK_TOP`; the toolchain uses
  the libraries under `$PDK_TOP/drive-qnx-safety/lib-target`.

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  export CUDA_VERSION=13.4
  export CUDA=cuda-$CUDA_VERSION
  export CUDA_ROOT=/usr/local/cuda-$CUDA_VERSION-safe
  export QNX_BASE=/drive/toolchains/qnx_toolchain  # Set to your QNX 8 toolchain installation path
  export QNX_HOST=$QNX_BASE/host/linux/x86_64/
  export QNX_TARGET=$QNX_BASE/target/qnx/
  export PATH=$PATH:$QNX_HOST/usr/bin
  cmake .. -DBUILD_SAMPLES=OFF -DBUILD_SAFE_SAMPLES=ON -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF \
      -DTRT_BUILD_PRODUCT=safe_inference \
      -DTRT_SAFETY_INFERENCE_ONLY=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=`pwd`/bin_dynamic_cross \
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=`pwd`/bin_dynamic_cross \
      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=`pwd`/bin_dynamic_cross \
      -DCMAKE_PREFIX_PATH=$TRT_ROOT \
      -DTensorRT-SafeInference_DIR=$TRT_ROOT/cmake/TensorRT-SafeInference \
      -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_qnx_safe.toolchain \
      -DCUDA_VERSION=$CUDA_VERSION -DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES=110
  make -j$(nproc)
  ```

  > NOTE: Set `QNX_BASE` to the same QNX 8 SDK used for DOS7 QNX builds. The
  > generated QNX Safety binaries are placed in `build/bin_dynamic_cross`.

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
