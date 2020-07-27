[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Documentation](https://img.shields.io/badge/TensorRT-documentation-brightgreen.svg)](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)



# TensorRT Open Source Software

This repository contains the Open Source Software (OSS) components of NVIDIA TensorRT. Included are the sources for TensorRT plugins and parsers (Caffe and ONNX), as well as sample applications demonstrating usage and capabilities of the TensorRT platform.


## Prerequisites

To build the TensorRT OSS components, ensure you meet the following package requirements:

**System Packages**

* [CUDA](https://developer.nvidia.com/cuda-toolkit)
  * Recommended versions:
  * cuda-11.0 + cuDNN-8.0
  * cuda-10.2 + cuDNN-8.0

* [GNU Make](https://ftp.gnu.org/gnu/make/) >= v4.1

* [CMake](https://github.com/Kitware/CMake/releases) >= v3.13

* [Python](<https://www.python.org/downloads/>) >= v3.6.5

* [PIP](https://pypi.org/project/pip/#history) >= v19.0
  * PyPI packages
  * [numpy](https://pypi.org/project/numpy/)
  * [onnx](https://pypi.org/project/onnx/1.6.0/) 1.6.0
  * [onnxruntime](https://pypi.org/project/onnxruntime/) >= 1.3.0
  * [pytest](https://pypi.org/project/pytest/)

* Essential libraries and utilities
  * [Git](https://git-scm.com/downloads), [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/), [Wget](https://www.gnu.org/software/wget/faq.html#download), [Zlib](https://zlib.net/)

* Cross compilation for Jetson platforms requires JetPack's host component installation
  * [JetPack](https://developer.nvidia.com/embedded/jetpack) >= 4.4
 
* Windows requires Visual Studio 2017 either Community or Enterprise Version
  * [Visual Studio](https://visualstudio.microsoft.com/vs/older-downloads/)

* Cross compilation for QNX requires the qnx developer toolchain
  * [QNX](https://blackberry.qnx.com/en)

**Optional Packages**

* Containerized builds
  * [Docker](https://docs.docker.com/install/) >= 19.03
  * [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) >= 2.0 or `nvidia-container-toolkit`

* Code formatting tools
  * [Clang-format](https://clang.llvm.org/docs/ClangFormat.html)
  * [Git-clang-format](https://github.com/llvm-mirror/clang/blob/master/tools/clang-format/git-clang-format)

* Required PyPI packages for Demos
  * [Tensorflow-gpu](https://pypi.org/project/tensorflow/1.14.0/) == 1.15.0

**TensorRT Release**

* [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download) v7.1


NOTE: Along with the TensorRT OSS components, the following source packages will also be downloaded, and they are not required to be installed on the system.

- [ONNX-TensorRT](https://github.com/onnx/onnx-tensorrt) v7.1
- [CUB](http://nvlabs.github.io/cub/) v1.8.0
- [Protobuf](https://github.com/protocolbuffers/protobuf.git) v3.8.x


## Downloading The TensorRT Components

1. #### Download TensorRT OSS sources.

	**Example: Bash**

	```bash
	git clone -b master https://github.com/nvidia/TensorRT TensorRT
	cd TensorRT
	git submodule update --init --recursive
	export TRT_SOURCE=`pwd`
	```

	**Example: Powershell**

	```powershell
	git clone -b master https://github.com/nvidia/TensorRT TensorRT
	cd TensorRT
	git submodule update --init --recursive
	$Env:TRT_RELEASE_PATH = $(Get-Location)
	```

2. #### Download the TensorRT binary release.

	To build the TensorRT OSS, obtain the corresponding TensorRT 7.1 binary release from [NVidia Developer Zone](https://developer.nvidia.com/nvidia-tensorrt-7x-download). For a list of key features, known and fixed issues, refer to the [TensorRT 7.1 Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/tensorrt-7.html#rel_7-1-0).

	**Example: Ubuntu 18.04 with cuda-11.0**

	Download and extract the latest *TensorRT 7.1 GA package for Ubuntu 18.04 and CUDA 11.0*
	```bash
	cd ~/Downloads
	tar -xvzf TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.0.cudnn8.0.tar.gz
	export TRT_RELEASE=`pwd`/TensorRT-7.1.3.4
	```

	**Example: Ubuntu 18.04 with cuda-11.0 on PowerPC**

	Download and extract the latest *TensorRT 7.1 GA package for Ubuntu 18.04 and CUDA 11.0*
	```bash
	cd ~/Downloads
	# Download TensorRT-7.1.3.4.Ubuntu-18.04.powerpc64le-gnu.cuda-11.0.cudnn8.0.tar.gz
	tar -xvzf TensorRT-7.1.3.4.Ubuntu-18.04.powerpc64le-gnu.cuda-11.0.cudnn8.0.tar.gz
	export TRT_RELEASE=`pwd`/TensorRT-7.1.3.4
	```

	**Example: CentOS/RedHat 7 with cuda-10.2**

	Download and extract the *TensorRT 7.1 GA for CentOS/RedHat 7 and CUDA 10.2 tar package*
	```bash
	cd ~/Downloads
	tar -xvzf TensorRT-7.1.3.4.CentOS-8.0.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz
	export TRT_RELEASE=`pwd`/TensorRT-7.1.3.4
	```

	**Example: Ubuntu 16.04 with cuda-11.0**

	Download and extract the *TensorRT 7.1 GA for Ubuntu 16.04 and CUDA 11.0 tar package*
	```bash
	cd ~/Downloads
	tar -xvzf TensorRT-7.1.3.4.Ubuntu-16.04.x86_64-gnu.cuda-11.0.cudnn8.0.tar.gz
	export TRT_RELEASE=`pwd`/TensorRT-7.1.3.4
	```

	**Example: Ubuntu18.04 cross compile QNX with cuda-10.2**

	Download and extract the *TensorRT 7.1 GA for QNX and CUDA 10.2 tar package*
	```bash
	cd ~/Downloads
	tar -xvzf TensorRT-7.1.3.4.Ubuntu-18.04.aarch64-qnx.cuda-10.2.cudnn7.6.tar.gz
	export TRT_RELEASE=`pwd`/TensorRT-7.1.3.4
	export QNX_HOST=/path/to/qnx/toolchain/host/linux/x86_64
	export QNX_TARGET=/path/to/qnx/toolchain/target/qnx7
	```
	**Example: Windows with cuda-11.0**

	Download and extract the *TensorRT 7.1 GA for Windows and CUDA 11.0 zip package* and add *msbuild* to *PATH*
	```powershell
	cd ~\Downloads
	Expand-Archive .\TensorRT-7.1.3.4.Windows10.x86_64.cuda-11.0.cudnn8.0.zip
	$Env:TRT_RELEASE_PATH = '$(Get-Location)\TensorRT-7.1.3.4'
	$Env:PATH += 'C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\'
	```

3. #### Download JetPack toolchain for cross-compilation.[OPTIONAL]

    **JetPack example**

    Using the SDK manager, download the host componets of the PDK version or Jetpack specified in the name of the Dockerfile. To do this:
    1. [**SDK Manager Step 01**] Log into the SDK manager
    2. [**SDK Manager Step 01**] Select the correct platform and Target OS System  (should be corresponding to the name of the Dockerfile you are building (e.g. Jetson AGX Xavier, `Linux Jetpack 4.4`), then click `Continue`
    3. [**SDK Manager Step 02**] Under `Download & Install Options` make note of or change the download folder **and Select Download now. Install later.** then agree to the license terms and click `Continue`
    You should now have all expected files to build the container. Move these into the `docker/jetpack_files` folder.

## Setting Up The Build Environment

* Install the *System Packages* list of components in the *Prerequisites* section.

* Alternatively, use the build containers as described below:

1. #### Generate the TensorRT build container.

    The docker container can be built using the included Dockerfiles and build script. The build container is configured with the environment and packages required for building TensorRT OSS.

    **Example: Ubuntu 18.04 with cuda-11.0**

    ```bash
    ./docker/build.sh --file docker/ubuntu.Dockerfile --tag tensorrt-ubuntu --os 18.04 --cuda 11.0
    ```

    **Example: Ubuntu 16.04 with cuda-11.0**

    ```bash
    ./docker/build.sh --file docker/ubuntu.Dockerfile --tag tensorrt-ubuntu1604 --os 16.04 --cuda 11.0
    ```

    **Example: CentOS/RedHat 7 with cuda-10.2**

    ```bash
    ./docker/build.sh --file docker/centos.Dockerfile --tag tensorrt-centos --os 7 --cuda 10.2
    ```

    **Example: Cross compile for JetPack 4.4 with cuda-10.2**
    ```bash
    ./docker/build.sh --file docker/ubuntu-cross-aarch64.Dockerfile --tag tensorrt-ubuntu-jetpack --os 18.04 --cuda 10.2
    ```

    **Example: Cross compile for PowerPC with cuda-11.0**
    ```bash
    ./docker/build.sh --file docker/ubuntu-cross-ppc64le.Dockerfile --tag tensorrt-ubuntu-ppc --os 18.04 --cuda 11.0
    ```

2. #### Launch the TensorRT build container.

	```bash
	./docker/launch.sh --tag tensorrt-ubuntu --gpus all --release $TRT_RELEASE --source $TRT_SOURCE
	```

	> NOTE: To run TensorRT/CUDA programs in the build container, install [NVIDIA Docker support](#prerequisites). Docker versions < 19.03 require `nvidia-docker2` and `--runtime=nvidia` flag for docker run commands. On versions >= 19.03, you need the `nvidia-container-toolkit` package and `--gpus <NUM_GPUS>` flag.


## Building The TensorRT OSS Components

* Generate Makefiles and build.

   **Example: Linux**

	```bash
	cd $TRT_SOURCE
	mkdir -p build && cd build
	cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_OUT_DIR=`pwd`/out
	make -j$(nproc)
	```
	
    **Example: Bare-metal build on Jetson (ARM64) with cuda-10.2**

    ```bash
    cd $TRT_SOURCE
    mkdir -p build && cd build
    cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_OUT_DIR=`pwd`/out -DTRT_PLATFORM_ID=aarch64 -DCUDA_VERSION=10.2
    make -j$(nproc)
    ```

    **Example: Cross compile for QNX with cuda-10.2**

	```bash
	cd $TRT_SOURCE
	mkdir -p build && cd build
	cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_OUT_DIR=`pwd`/out -DCMAKE_TOOLCHAIN_FILE=$TRT_SOURCE/cmake/toolchains/cmake_qnx.toolchain
	make -j$(nproc)
	```

    **Example: Powershell**

	```powershell
	cd $Env:TRT_SOURCE
	mkdir -p build ; cd build
	cmake .. -DTRT_LIB_DIR=$Env:TRT_RELEASE\lib -DTRT_OUT_DIR='$(Get-Location)\out' -DCMAKE_TOOLCHAIN_FILE=..\cmake\toolchains\cmake_x64_win.toolchain
	msbuild ALL_BUILD.vcxproj
	```

	> NOTE:
	> 1. The default CUDA version used by CMake is 11.0. To override this, for example to 10.2, append `-DCUDA_VERSION=10.2` to the cmake command.
	> 2. Samples may fail to link on CentOS7. To work around this create the following symbolic link:
	> `ln -s $TRT_OUT_DIR/libnvinfer_plugin.so $TRT_OUT_DIR/libnvinfer_plugin.so.7`

	The required CMake arguments are:

	- `TRT_LIB_DIR`: Path to the TensorRT installation directory containing libraries.

	- `TRT_OUT_DIR`: Output directory where generated build artifacts will be copied.

	The following CMake build parameters are *optional*:

	- `CMAKE_BUILD_TYPE`: Specify if binaries generated are for release or debug (contain debug symbols). Values consists of [`Release`] | `Debug`

	- `CUDA_VERISON`: The version of CUDA to target, for example [`11.0`].

	- `CUDNN_VERSION`: The version of cuDNN to target, for example [`8.0`].

	- `NVCR_SUFFIX`: Optional nvcr/cuda image suffix. Set to "-rc" for CUDA11 RC builds until general availability. Blank by default.

	- `PROTOBUF_VERSION`:  The version of Protobuf to use, for example [`3.8.x`]. Note: Changing this will not configure CMake to use a system version of Protobuf, it will configure CMake to download and try building that version.

	- `CMAKE_TOOLCHAIN_FILE`: The path to a toolchain file for cross compilation.

	- `BUILD_PARSERS`: Specify if the parsers should be built, for example [`ON`] | `OFF`.  If turned OFF, CMake will try to find precompiled versions of the parser libraries to use in compiling samples. First in `${TRT_LIB_DIR}`, then on the system. If the build type is Debug, then it will prefer debug builds of the libraries before release versions if available.

	- `BUILD_PLUGINS`: Specify if the plugins should be built, for example [`ON`] | `OFF`. If turned OFF, CMake will try to find a precompiled version of the plugin library to use in compiling samples. First in `${TRT_LIB_DIR}`, then on the system. If the build type is Debug, then it will prefer debug builds of the libraries before release versions if available.

	- `BUILD_SAMPLES`: Specify if the samples should be built, for example [`ON`] | `OFF`.

	Other build options with limited applicability:

	- `CUB_VERSION`: The version of CUB to use, for example [`1.8.0`].

	- `GPU_ARCHS`: GPU (SM) architectures to target. By default we generate CUDA code for all major SMs. Specific SM versions can be specified here as a quoted space-separated list to reduce compilation time and binary size. Table of compute capabilities of NVIDIA GPUs can be found [here](https://developer.nvidia.com/cuda-gpus). Examples:
        - NVidia A100: `-DGPU_ARCHS="80"`
        - Tesla T4, GeForce RTX 2080: `-DGPU_ARCHS="75"`
        - Titan V, Tesla V100: `-DGPU_ARCHS="70"`
        - Multiple SMs: `-DGPU_ARCHS="80 75"`

    - `TRT_PLATFORM_ID`: Bare-metal build (unlike containerized cross-compilation) on non Linux/x86 platforms must explicitly specify the target platform. Currently supported options: `x86_64` (default), `aarch64`


## Useful Resources

#### TensorRT

* [TensorRT Homepage](https://developer.nvidia.com/tensorrt)
* [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
* [TensorRT Sample Support Guide](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html)
* [TensorRT Discussion Forums](https://devtalk.nvidia.com/default/board/304/tensorrt/)


## Known Issues

#### TensorRT 7.1
* [demo/BERT](demo/BERT) has a known accuracy regression for Volta GPUs; F1 score dropped (from 90 in TensorRT 7.0) to 85. A fix is underway.
* See [Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/tensorrt-7.html#rel_7-1-3).
