[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Documentation](https://img.shields.io/badge/TensorRT-documentation-brightgreen.svg)](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)

# TensorRT Open Source Software
This repository contains the Open Source Software (OSS) components of NVIDIA TensorRT. Included are the sources for TensorRT plugins and parsers (Caffe and ONNX), as well as sample applications demonstrating usage and capabilities of the TensorRT platform. These open source software components are a subset of the TensorRT General Availability (GA) release with some extensions and bug-fixes.

* For code contributions to TensorRT-OSS, please see our [Contribution Guide](CONTRIBUTING.md) and [Coding Guidelines](CODING-GUIDELINES.md).
* For a summary of new additions and updates shipped with TensorRT-OSS releases, please refer to the [Changelog](CHANGELOG.md).

# Build

## Prerequisites
To build the TensorRT-OSS components, you will first need the following software packages.

**TensorRT GA build**
* [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download) v7.2.1
    - See [Downloading TensorRT Builds](#downloading-tensorrt-builds) for details

**System Packages**
* [CUDA](https://developer.nvidia.com/cuda-toolkit)
  * Recommended versions:
  * cuda-11.1 + cuDNN-8.0
  * cuda-11.0 + cuDNN-8.0
  * cuda-10.2 + cuDNN-8.0
* [GNU make](https://ftp.gnu.org/gnu/make/) >= v4.1
* [cmake](https://github.com/Kitware/CMake/releases) >= v3.13
* [python](<https://www.python.org/downloads/>) >= v3.6.5
* [pip](https://pypi.org/project/pip/#history) >= v19.0
* Essential utilities
  * [git](https://git-scm.com/downloads), [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/), [wget](https://www.gnu.org/software/wget/faq.html#download), [zlib](https://zlib.net/)

**Optional Packages**
* Containerized build
  * [Docker](https://docs.docker.com/install/) >= 19.03
  * [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
* Toolchains and SDKs
  * (Cross compilation for Jetson platform) [NVIDIA JetPack](https://developer.nvidia.com/embedded/jetpack) >= 4.4
  * (For Windows builds) [Visual Studio](https://visualstudio.microsoft.com/vs/older-downloads/) 2017 Community or Enterprise edition
  * (Cross compilation for QNX platform) [QNX Toolchain](https://blackberry.qnx.com/en)
* PyPI packages (for demo applications/tests)
  * [numpy](https://pypi.org/project/numpy/)
  * [onnx](https://pypi.org/project/onnx/1.6.0/) 1.6.0
  * [onnxruntime](https://pypi.org/project/onnxruntime/) >= 1.3.0
  * [pytest](https://pypi.org/project/pytest/)
  * [tensorflow-gpu](https://pypi.org/project/tensorflow/1.15.4/) 1.15.4
* Code formatting tools (for contributors)
  * [Clang-format](https://clang.llvm.org/docs/ClangFormat.html)
  * [Git-clang-format](https://github.com/llvm-mirror/clang/blob/master/tools/clang-format/git-clang-format)

  > NOTE: [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt), [cub](http://nvlabs.github.io/cub/), and [protobuf](https://github.com/protocolbuffers/protobuf.git) packages are downloaded along with TensorRT OSS, and not required to be installed.

## Downloading TensorRT Build

1. #### Download TensorRT OSS
	**On Linux: Bash**
	```bash
	git clone -b master https://github.com/nvidia/TensorRT TensorRT
	cd TensorRT
	git submodule update --init --recursive
	export TRT_SOURCE=`pwd`
	```
	**On Windows: Powershell**
	```powershell
	git clone -b master https://github.com/nvidia/TensorRT TensorRT
	cd TensorRT
	git submodule update --init --recursive
	$Env:TRT_SOURCE = $(Get-Location)
	```

2. #### Download TensorRT GA
	To build TensorRT OSS, obtain the corresponding TensorRT GA build from [NVIDIA Developer Zone](https://developer.nvidia.com/nvidia-tensorrt-download).

	**Example: Ubuntu 18.04 on x86-64 with cuda-11.1**

	Download and extract the latest *TensorRT 7.2.1 GA package for Ubuntu 18.04 and CUDA 11.1*
	```bash
	cd ~/Downloads
	tar -xvzf TensorRT-7.2.1.6.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz
	export TRT_RELEASE=`pwd`/TensorRT-7.2.1.6
	```
	**Example: Ubuntu 18.04 on PowerPC with cuda-11.0**

	Download and extract the latest *TensorRT 7.2.1 GA package for Ubuntu 18.04 and CUDA 11.0*
	```bash
	cd ~/Downloads
	tar -xvzf TensorRT-7.2.1.6.Ubuntu-18.04.powerpc64le-gnu.cuda-11.0.cudnn8.0.tar.gz
	export TRT_RELEASE=`pwd`/TensorRT-7.2.1.6
	```
	**Example: CentOS/RedHat 7 on x86-64 with cuda-11.0**

	Download and extract the *TensorRT 7.2.1 GA for CentOS/RedHat 7 and CUDA 11.0 tar package*
	```bash
	cd ~/Downloads
	tar -xvzf TensorRT-7.2.1.6.CentOS-7.6.x86_64-gnu.cuda-11.0.cudnn8.0.tar.gz
	export TRT_RELEASE=`pwd`/TensorRT-7.2.1.6
	```
	**Example: Ubuntu18.04 Cross-Compile for QNX with cuda-10.2**

	Download and extract the *TensorRT 7.2.1 GA for QNX and CUDA 10.2 tar package*
	```bash
	cd ~/Downloads
	tar -xvzf TensorRT-7.2.1.6.Ubuntu-18.04.aarch64-qnx.cuda-10.2.cudnn7.6.tar.gz
	export TRT_RELEASE=`pwd`/TensorRT-7.2.1.6
	export QNX_HOST=/<path-to-qnx-toolchain>/host/linux/x86_64
	export QNX_TARGET=/<path-to-qnx-toolchain>/target/qnx7
	```
	**Example: Windows on x86-64 with cuda-11.0**

	Download and extract the *TensorRT 7.2.1 GA for Windows and CUDA 11.0 zip package* and add *msbuild* to *PATH*
	```powershell
	cd ~\Downloads
	Expand-Archive .\TensorRT-7.2.1.6.Windows10.x86_64.cuda-11.0.cudnn8.0.zip
	$Env:TRT_RELEASE = '$(Get-Location)\TensorRT-7.2.1.6'
	$Env:PATH += 'C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\'
	```

3. #### (Optional) JetPack SDK for Jetson builds
    Using the JetPack SDK manager, download the host components. Steps:
    1. Download and launch the SDK manager. Login with your developer account.
    2. Select the  platform and target OS  (example: Jetson AGX Xavier, `Linux Jetpack 4.4`), and click Continue.
    3. Under `Download & Install Options` change the download folder and select `Download now, Install later`. Agree to the license terms and click Continue.
    4. Move the extracted files into the `$TRT_SOURCE/docker/jetpack_files` folder.

## Setting Up The Build Environment

For native builds, install the [prerequisite](#prerequisites) *System Packages*. Alternatively (recommended for non-Windows builds), install Docker and generate a build container as described below:

1. #### Generate the TensorRT-OSS build container.
    The TensorRT-OSS build container can be generated using the Dockerfiles and build script included with TensorRT-OSS. The build container is bundled with packages and environment required for building TensorRT OSS.

    **Example: Ubuntu 18.04 on x86-64 with cuda-11.1**
    ```bash
    ./docker/build.sh --file docker/ubuntu.Dockerfile --tag tensorrt-ubuntu --os 18.04 --cuda 11.1
    ```
    **Example: Ubuntu 18.04 on PowerPC with cuda-11.0**
    ```bash
    ./docker/build.sh --file docker/ubuntu-cross-ppc64le.Dockerfile --tag tensorrt-ubuntu-ppc --os 18.04 --cuda 11.0
    ```
    **Example: CentOS/RedHat 7 on x86-64 with cuda-11.0**
    ```bash
    ./docker/build.sh --file docker/centos.Dockerfile --tag tensorrt-centos --os 7 --cuda 11.0
    ```
    **Example: Ubuntu 18.04 Cross-Compile for Jetson (arm64) with cuda-10.2 (JetPack)**
    ```bash
    ./docker/build.sh --file docker/ubuntu-cross-aarch64.Dockerfile --tag tensorrt-cross-jetpack --os 18.04 --cuda 10.2
    ```

2. #### Launch the TensorRT-OSS build container.
    **Example: Ubuntu 18.04 build container**
	```bash
	./docker/launch.sh --tag tensorrt-ubuntu --gpus all --release $TRT_RELEASE --source $TRT_SOURCE
	```
	> NOTE:
	1. Use the tag corresponding to the build container you generated in 
	2. To run TensorRT/CUDA programs in the build container, install [NVIDIA Container Toolkit](#prerequisites). Docker versions < 19.03 require `nvidia-docker2` and `--runtime=nvidia` flag for docker run commands. On versions >= 19.03, you need the `nvidia-container-toolkit` package and `--gpus all` flag.

## Building TensorRT-OSS
* Generate Makefiles or VS project (Windows) and build.

   **Example: Linux (x86-64) build with default cuda-11.1**
	```bash
	cd $TRT_SOURCE
	mkdir -p build && cd build
	cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_OUT_DIR=`pwd`/out
	make -j$(nproc)
	```
    **Example: Native build on Jetson (arm64) with cuda-10.2**
    ```bash
    cd $TRT_SOURCE
    mkdir -p build && cd build
    cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_OUT_DIR=`pwd`/out -DTRT_PLATFORM_ID=aarch64 -DCUDA_VERSION=10.2
    make -j$(nproc)
    ```
    **Example: Ubuntu 18.04 Cross-Compile for Jetson (arm64) with cuda-10.2 (JetPack)**
	```bash
	cd $TRT_SOURCE
	mkdir -p build && cd build
	cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_OUT_DIR=`pwd`/out -DCMAKE_TOOLCHAIN_FILE=$TRT_SOURCE/cmake/toolchains/cmake_aarch64.toolchain -DCUDA_VERSION=10.2
	make -j$(nproc)
	```
    **Example: Cross-Compile for QNX with cuda-10.2**
	```bash
	cd $TRT_SOURCE
	mkdir -p build && cd build
	cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_OUT_DIR=`pwd`/out -DCMAKE_TOOLCHAIN_FILE=$TRT_SOURCE/cmake/toolchains/cmake_qnx.toolchain -DCUDA_VERSION=10.2
	make -j$(nproc)
	```
    **Example: Windows (x86-64) build in Powershell**
	```powershell
	cd $Env:TRT_SOURCE
	mkdir -p build ; cd build
	cmake .. -DTRT_LIB_DIR=$Env:TRT_RELEASE\lib -DTRT_OUT_DIR='$(Get-Location)\out' -DCMAKE_TOOLCHAIN_FILE=..\cmake\toolchains\cmake_x64_win.toolchain
	msbuild ALL_BUILD.vcxproj
	```
	> NOTE:
	1. The default CUDA version used by CMake is 11.1. To override this, for example to 10.2, append `-DCUDA_VERSION=10.2` to the cmake command.
	2. If samples fail to link on CentOS7, create this symbolic link: `ln -s $TRT_OUT_DIR/libnvinfer_plugin.so $TRT_OUT_DIR/libnvinfer_plugin.so.7`
* Required CMake build arguments are:
	- `TRT_LIB_DIR`: Path to the TensorRT installation directory containing libraries.
	- `TRT_OUT_DIR`: Output directory where generated build artifacts will be copied.
* Optional CMake build arguments:
	- `CMAKE_BUILD_TYPE`: Specify if binaries generated are for release or debug (contain debug symbols). Values consists of [`Release`] | `Debug`
	- `CUDA_VERISON`: The version of CUDA to target, for example [`11.1`].
	- `CUDNN_VERSION`: The version of cuDNN to target, for example [`8.0`].
	- `NVCR_SUFFIX`: Optional nvcr/cuda image suffix. Set to "-rc" for CUDA11 RC builds until general availability. Blank by default.
	- `PROTOBUF_VERSION`:  The version of Protobuf to use, for example [`3.0.0`]. Note: Changing this will not configure CMake to use a system version of Protobuf, it will configure CMake to download and try building that version.
	- `CMAKE_TOOLCHAIN_FILE`: The path to a toolchain file for cross compilation.
	- `BUILD_PARSERS`: Specify if the parsers should be built, for example [`ON`] | `OFF`.  If turned OFF, CMake will try to find precompiled versions of the parser libraries to use in compiling samples. First in `${TRT_LIB_DIR}`, then on the system. If the build type is Debug, then it will prefer debug builds of the libraries before release versions if available.
	- `BUILD_PLUGINS`: Specify if the plugins should be built, for example [`ON`] | `OFF`. If turned OFF, CMake will try to find a precompiled version of the plugin library to use in compiling samples. First in `${TRT_LIB_DIR}`, then on the system. If the build type is Debug, then it will prefer debug builds of the libraries before release versions if available.
	- `BUILD_SAMPLES`: Specify if the samples should be built, for example [`ON`] | `OFF`.
	- `CUB_VERSION`: The version of CUB to use, for example [`1.8.0`].
	- `GPU_ARCHS`: GPU (SM) architectures to target. By default we generate CUDA code for all major SMs. Specific SM versions can be specified here as a quoted space-separated list to reduce compilation time and binary size. Table of compute capabilities of NVIDIA GPUs can be found [here](https://developer.nvidia.com/cuda-gpus). Examples:
        - NVidia A100: `-DGPU_ARCHS="80"`
        - Tesla T4, GeForce RTX 2080: `-DGPU_ARCHS="75"`
        - Titan V, Tesla V100: `-DGPU_ARCHS="70"`
        - Multiple SMs: `-DGPU_ARCHS="80 75"`
	- `TRT_PLATFORM_ID`: Bare-metal build (unlike containerized cross-compilation) on non Linux/x86 platforms must explicitly specify the target platform. Currently supported options: `x86_64` (default), `aarch64`

#### (Optional) Install TensorRT python bindings

* The TensorRT python API bindings must be installed for running TensorRT python applications

    **Example: install TensorRT wheel for python 3.6**
    ```bash
    pip3 install $TRT_RELEASE/python/tensorrt-7.2.1.6-cp36-none-linux_x86_64.whl
    ```

# References

## TensorRT Resources

* [TensorRT Homepage](https://developer.nvidia.com/tensorrt)
* [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
* [TensorRT Sample Support Guide](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html)
* [TensorRT Discussion Forums](https://devtalk.nvidia.com/default/board/304/tensorrt/)
* [TensorRT Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html).

## Known Issues

#### TensorRT 7.2.1
* None
