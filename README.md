[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Documentation](https://img.shields.io/badge/TensorRT-documentation-brightgreen.svg)](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)

# TensorRT Open Source Software
This repository contains the Open Source Software (OSS) components of NVIDIA TensorRT. Included are the sources for TensorRT plugins and parsers (Caffe and ONNX), as well as sample applications demonstrating usage and capabilities of the TensorRT platform. These open source software components are a subset of the TensorRT General Availability (GA) release with some extensions and bug-fixes.

* For code contributions to TensorRT-OSS, please see our [Contribution Guide](CONTRIBUTING.md) and [Coding Guidelines](CODING-GUIDELINES.md).
* For a summary of new additions and updates shipped with TensorRT-OSS releases, please refer to the [Changelog](CHANGELOG.md).
* For business inquiries, please contact [researchinquiries@nvidia.com](mailto:researchinquiries@nvidia.com)
* For press and other inquiries, please contact Hector Marinez at [hmarinez@nvidia.com](mailto:hmarinez@nvidia.com)


# Build

## Prerequisites
To build the TensorRT-OSS components, you will first need the following software packages.

**TensorRT GA build**
* [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download) v8.4.1.5

**System Packages**
* [CUDA](https://developer.nvidia.com/cuda-toolkit)
  * Recommended versions:
  * cuda-11.6.x + cuDNN-8.4
  * cuda-10.2 + cuDNN-8.4
* [GNU make](https://ftp.gnu.org/gnu/make/) >= v4.1
* [cmake](https://github.com/Kitware/CMake/releases) >= v3.13
* [python](<https://www.python.org/downloads/>) >= v3.6.9
* [pip](https://pypi.org/project/pip/#history) >= v19.0
* Essential utilities
  * [git](https://git-scm.com/downloads), [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/), [wget](https://www.gnu.org/software/wget/faq.html#download)

**Optional Packages**
* Containerized build
  * [Docker](https://docs.docker.com/install/) >= 19.03
  * [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
* Toolchains and SDKs
  * (Cross compilation for Jetson platform) [NVIDIA JetPack](https://developer.nvidia.com/embedded/jetpack) >= 5.0 (current support only for TensorRT 8.4.0)
  * (For Windows builds) [Visual Studio](https://visualstudio.microsoft.com/vs/older-downloads/) 2017 Community or Enterprise edition
  * (Cross compilation for QNX platform) [QNX Toolchain](https://blackberry.qnx.com/en)
* PyPI packages (for demo applications/tests)
  * [onnx](https://pypi.org/project/onnx/) 1.9.0
  * [onnxruntime](https://pypi.org/project/onnxruntime/) 1.8.0
  * [tensorflow-gpu](https://pypi.org/project/tensorflow/) >= 2.5.1
  * [Pillow](https://pypi.org/project/Pillow/) >= 9.0.1
  * [pycuda](https://pypi.org/project/pycuda/) < 2021.1
  * [numpy](https://pypi.org/project/numpy/)
  * [pytest](https://pypi.org/project/pytest/)
* Code formatting tools (for contributors)
  * [Clang-format](https://clang.llvm.org/docs/ClangFormat.html)
  * [Git-clang-format](https://github.com/llvm-mirror/clang/blob/master/tools/clang-format/git-clang-format)

  > NOTE: [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt), [cub](http://nvlabs.github.io/cub/), and [protobuf](https://github.com/protocolbuffers/protobuf.git) packages are downloaded along with TensorRT OSS, and not required to be installed.

## Downloading TensorRT Build

1. #### Download TensorRT OSS
	```bash
	git clone -b master https://github.com/nvidia/TensorRT TensorRT
	cd TensorRT
	git submodule update --init --recursive
	```

2. #### (Optional - if not using TensorRT container) Specify the TensorRT GA release build

    If using the TensorRT OSS build container, TensorRT libraries are preinstalled under `/usr/lib/x86_64-linux-gnu` and you may skip this step.

    Else download and extract the TensorRT GA build from [NVIDIA Developer Zone](https://developer.nvidia.com/nvidia-tensorrt-download).

    **Example: Ubuntu 20.04 on x86-64 with cuda-11.6.2**

    ```bash
    cd ~/Downloads
    tar -xvzf TensorRT-8.4.1.5.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz
    export TRT_LIBPATH=`pwd`/TensorRT-8.4.1.5
    ```

    **Example: Windows on x86-64 with cuda-11.4**

    ```powershell
    cd ~\Downloads
    Expand-Archive .\TensorRT-8.4.1.5.Windows10.x86_64.cuda-11.6.cudnn8.4.zip
    $Env:TRT_LIBPATH = '$(Get-Location)\TensorRT-8.4.1.5'
    $Env:PATH += 'C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\'
    ```


3. #### (Optional - for Jetson builds only) Download the JetPack SDK
    1. Download and launch the JetPack SDK manager. Login with your NVIDIA developer account.
    2. Select the  platform and target OS  (example: Jetson AGX Xavier, `Linux Jetpack 5.0`), and click Continue.
    3. Under `Download & Install Options` change the download folder and select `Download now, Install later`. Agree to the license terms and click Continue.
    4. Move the extracted files into the `<TensorRT-OSS>/docker/jetpack_files` folder.


## Setting Up The Build Environment

For Linux platforms, we recommend that you generate a docker container for building TensorRT OSS as described below. For native builds, on Windows for example, please install the [prerequisite](#prerequisites) *System Packages*.

1. #### Generate the TensorRT-OSS build container.
    The TensorRT-OSS build container can be generated using the supplied Dockerfiles and build script. The build container is configured for building TensorRT OSS out-of-the-box.

    **Example: Ubuntu 20.04 on x86-64 with cuda-11.6.2 (default)**
    ```bash
    ./docker/build.sh --file docker/ubuntu-20.04.Dockerfile --tag tensorrt-ubuntu20.04-cuda11.6
    ```
    **Example: CentOS/RedHat 7 on x86-64 with cuda-10.2**
    ```bash
    ./docker/build.sh --file docker/centos-7.Dockerfile --tag tensorrt-centos7-cuda10.2 --cuda 10.2
    ```
    **Example: Ubuntu 20.04 cross-compile for Jetson (aarch64) with cuda-11.4.2 (JetPack SDK)**
    ```bash
    ./docker/build.sh --file docker/ubuntu-cross-aarch64.Dockerfile --tag tensorrt-jetpack-cuda11.4
    ```
    **Example: Ubuntu 20.04 on aarch64 with cuda-11.4.2**
    ```bash
    ./docker/build.sh --file docker/ubuntu-20.04-aarch64.Dockerfile --tag tensorrt-aarch64-ubuntu20.04-cuda11.4
    ```

2. #### Launch the TensorRT-OSS build container.
    **Example: Ubuntu 20.04 build container**
	```bash
	./docker/launch.sh --tag tensorrt-ubuntu20.04-cuda11.6 --gpus all
	```
	> NOTE:
	1. Use the `--tag` corresponding to build container generated in Step 1.
	2. [NVIDIA Container Toolkit](#prerequisites) is required for GPU access (running TensorRT applications) inside the build container.
	3. `sudo` password for Ubuntu build containers is 'nvidia'.
	4. Specify port number using `--jupyter <port>` for launching Jupyter notebooks.

## Building TensorRT-OSS
* Generate Makefiles or VS project (Windows) and build.

    **Example: Linux (x86-64) build with default cuda-11.6.2**
	```bash
	cd $TRT_OSSPATH
	mkdir -p build && cd build
	cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out
	make -j$(nproc)
	```

    > NOTE: On CentOS7, the default g++ version does not support C++14. For native builds (not using the CentOS7 build container), first install devtoolset-8 to obtain the updated g++ toolchain as follows:
    ```bash
    yum -y install centos-release-scl
    yum-config-manager --enable rhel-server-rhscl-7-rpms
    yum -y install devtoolset-8
    export PATH="/opt/rh/devtoolset-8/root/bin:${PATH}
    ```

    **Example: Linux (aarch64) build with default cuda-11.6.2**
	```bash
	cd $TRT_OSSPATH
	mkdir -p build && cd build
	cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64-native.toolchain
	make -j$(nproc)
	```

    **Example: Native build on Jetson (aarch64) with cuda-11.4**
    ```bash
    cd $TRT_OSSPATH
    mkdir -p build && cd build
    cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DTRT_PLATFORM_ID=aarch64 -DCUDA_VERSION=11.4
    CC=/usr/bin/gcc make -j$(nproc)
    ```
    > NOTE: C compiler must be explicitly specified via `CC=` for native `aarch64` builds of protobuf.

    **Example: Ubuntu 20.04 Cross-Compile for Jetson (aarch64) with cuda-11.4 (JetPack)**
	```bash
	cd $TRT_OSSPATH
	mkdir -p build && cd build
	cmake .. -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64.toolchain -DCUDA_VERSION=11.4 -DCUDNN_LIB=/pdk_files/cudnn/usr/lib/aarch64-linux-gnu/libcudnn.so -DCUBLAS_LIB=/usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/libcublas.so -DCUBLASLT_LIB=/usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/libcublasLt.so

	make -j$(nproc)
	```
    > NOTE: The latest JetPack SDK v5.0 only supports TensorRT 8.4.0.

    **Example: Windows (x86-64) build in Powershell**
	```powershell
	cd $Env:TRT_OSSPATH
	mkdir -p build ; cd build
	cmake .. -DTRT_LIB_DIR=$Env:TRT_LIBPATH -DTRT_OUT_DIR='$(Get-Location)\out' -DCMAKE_TOOLCHAIN_FILE=..\cmake\toolchains\cmake_x64_win.toolchain
	msbuild ALL_BUILD.vcxproj
	```
	> NOTE:
	1. The default CUDA version used by CMake is 11.6.2. To override this, for example to 10.2, append `-DCUDA_VERSION=10.2` to the cmake command.
	2. If samples fail to link on CentOS7, create this symbolic link: `ln -s $TRT_OUT_DIR/libnvinfer_plugin.so $TRT_OUT_DIR/libnvinfer_plugin.so.8`
* Required CMake build arguments are:
	- `TRT_LIB_DIR`: Path to the TensorRT installation directory containing libraries.
	- `TRT_OUT_DIR`: Output directory where generated build artifacts will be copied.
* Optional CMake build arguments:
	- `CMAKE_BUILD_TYPE`: Specify if binaries generated are for release or debug (contain debug symbols). Values consists of [`Release`] | `Debug`
	- `CUDA_VERISON`: The version of CUDA to target, for example [`11.6.2`].
	- `CUDNN_VERSION`: The version of cuDNN to target, for example [`8.4`].
	- `PROTOBUF_VERSION`:  The version of Protobuf to use, for example [`3.0.0`]. Note: Changing this will not configure CMake to use a system version of Protobuf, it will configure CMake to download and try building that version.
	- `CMAKE_TOOLCHAIN_FILE`: The path to a toolchain file for cross compilation.
	- `BUILD_PARSERS`: Specify if the parsers should be built, for example [`ON`] | `OFF`.  If turned OFF, CMake will try to find precompiled versions of the parser libraries to use in compiling samples. First in `${TRT_LIB_DIR}`, then on the system. If the build type is Debug, then it will prefer debug builds of the libraries before release versions if available.
	- `BUILD_PLUGINS`: Specify if the plugins should be built, for example [`ON`] | `OFF`. If turned OFF, CMake will try to find a precompiled version of the plugin library to use in compiling samples. First in `${TRT_LIB_DIR}`, then on the system. If the build type is Debug, then it will prefer debug builds of the libraries before release versions if available.
	- `BUILD_SAMPLES`: Specify if the samples should be built, for example [`ON`] | `OFF`.
	- `GPU_ARCHS`: GPU (SM) architectures to target. By default we generate CUDA code for all major SMs. Specific SM versions can be specified here as a quoted space-separated list to reduce compilation time and binary size. Table of compute capabilities of NVIDIA GPUs can be found [here](https://developer.nvidia.com/cuda-gpus). Examples:
        - NVidia A100: `-DGPU_ARCHS="80"`
        - Tesla T4, GeForce RTX 2080: `-DGPU_ARCHS="75"`
        - Titan V, Tesla V100: `-DGPU_ARCHS="70"`
        - Multiple SMs: `-DGPU_ARCHS="80 75"`
	- `TRT_PLATFORM_ID`: Bare-metal build (unlike containerized cross-compilation) on non Linux/x86 platforms must explicitly specify the target platform. Currently supported options: `x86_64` (default), `aarch64`

# References

## TensorRT Resources

* [TensorRT Developer Home](https://developer.nvidia.com/tensorrt)
* [TensorRT QuickStart Guide](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html)
* [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
* [TensorRT Sample Support Guide](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html)
* [TensorRT ONNX Tools](https://docs.nvidia.com/deeplearning/tensorrt/index.html#tools)
* [TensorRT Discussion Forums](https://devtalk.nvidia.com/default/board/304/tensorrt/)
* [TensorRT Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)

## Known Issues

* Please refer to [TensorRT 8.4 Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/tensorrt-8.html#tensorrt-8)
