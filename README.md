[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Documentation](https://img.shields.io/badge/TensorRT-documentation-brightgreen.svg)](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)



# TensorRT Open Source Software

This repository contains the Open Source Software (OSS) components of NVIDIA TensorRT. Included are the sources for TensorRT plugins and parsers (Caffe and ONNX), as well as sample applications demonstrating usage and capabilities of the TensorRT platform.


## Prerequisites

To build the TensorRT OSS components, ensure you meet the following package requirements:

**System Packages**

* [CUDA](https://developer.nvidia.com/cuda-toolkit)
  * Recommended versions:
  * [cuda-10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) + cuDNN-7.6
  * [cuda-10.0](https://developer.nvidia.com/cuda-10.0-download-archive) + cuDNN-7.6
  * [cuda-9.0](https://developer.nvidia.com/cuda-90-download-archive) + cuDNN 7.6

* [GNU Make](https://ftp.gnu.org/gnu/make/) >= v4.1

* [CMake](https://github.com/Kitware/CMake/releases) >= v3.13

* [Python](<https://www.python.org/downloads/>)
  * Recommended versions:
  * [Python2](https://www.python.org/downloads/release/python-2715/) >= v2.7.15
  * [Python3](https://www.python.org/downloads/release/python-365/) >= v3.6.5

* [PIP](https://pypi.org/project/pip/#history) >= v19.0

* Essential libraries and utilities
  * [Git](https://git-scm.com/downloads), [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/), [Wget](https://www.gnu.org/software/wget/faq.html#download), [Zlib](https://zlib.net/)

**Optional Packages**

* Containerized builds
  * [Docker](https://docs.docker.com/install/) >= 1.12
  * [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) >= 2.0

* Code formatting tools
  * [Clang-format](https://clang.llvm.org/docs/ClangFormat.html)
  * [Git-clang-format](https://github.com/llvm-mirror/clang/blob/master/tools/clang-format/git-clang-format)

**TensorRT Release**

* [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download) v6.0.1

NOTE: Along with the TensorRT OSS components, the following source packages will also be downloaded, and they are not required to be installed on the system.

- [ONNX-TensorRT](https://github.com/onnx/onnx-tensorrt) v6.0
- [CUB](http://nvlabs.github.io/cub/) v1.8.0
- [Protobuf](https://github.com/protocolbuffers/protobuf.git) v3.8.x


## Downloading The TensorRT Components

1. #### Download TensorRT OSS sources.

	```bash
	git clone -b master https://github.com/nvidia/TensorRT TensorRT
	cd TensorRT
	git submodule update --init --recursive
	export TRT_SOURCE=`pwd`
	```

2. #### Download the TensorRT binary release.

	To build the TensorRT OSS, obtain the corresponding TensorRT 6.0.1 binary release from [NVidia Developer Zone](https://developer.nvidia.com/nvidia-tensorrt-download). For a list of key features, known and fixed issues, see the [TensorRT 6.0.1 Release Notes](https://docs.nvidia.com/deeplearning/sdk/tensorrt-release-notes/index.html).

	**Example: Ubuntu 18.04 with cuda-10.1**

	Download and extract the *TensorRT 6.0.1.5 GA for Ubuntu 18.04 and CUDA 10.1 tar package*
	```bash
	cd ~/Downloads
	# Download TensorRT-6.0.1.5.Ubuntu-18.04.x86_64-gnu.cuda-10.1.cudnn7.6.tar.gz
	tar -xvzf TensorRT-6.0.1.5.Ubuntu-18.04.x86_64-gnu.cuda-10.1.cudnn7.6.tar.gz
	export TRT_RELEASE=`pwd`/TensorRT-6.0.1.5
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_RELEASE/lib
	```

	**Example: CentOS/RedHat 7 with cuda-9.0**

	Download and extract the *TensorRT 6.0.1.5 GA for CentOS/RedHat 7 and CUDA 9.0 tar package*
	```bash
	cd ~/Downloads
	# Download TensorRT-6.0.1.5.Red-Hat.x86_64-gnu.cuda-9.0.cudnn7.6.tar.gz
	tar -xvzf TensorRT-6.0.1.5.Red-Hat.x86_64-gnu.cuda-9.0.cudnn7.6.tar.gz
	export TRT_RELEASE=`pwd`/TensorRT-6.0.1.5
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_RELEASE/lib
	```

## Setting Up The Build Environment

* Install the *System Packages* list of components in the *Prerequisites* section.

* Alternatively, use the build containers as described below:

1. #### Generate the TensorRT build container.

	The docker container can be built using the included Dockerfile. The build container is configured with the environment and packages required for building TensorRT OSS.

	**Example: Ubuntu 18.04 with cuda-10.1**

	```bash
	docker build -f docker/ubuntu-18.04.Dockerfile --build-arg CUDA_VERSION=10.1 --tag=tensorrt .
	```

	**Example: CentOS/RedHat 7 with cuda-9.0**

	```bash
	docker build -f docker/centos-7.Dockerfile --build-arg CUDA_VERSION=9.0 --tag=tensorrt .
	```

2. #### Launch the TensorRT build container.

	```bash
	docker run -v $TRT_RELEASE:/tensorrt -v $TRT_SOURCE:/workspace/TensorRT -it tensorrt:latest
	```

	> NOTE: To run TensorRT/CUDA programs within the build container, install [nvidia-docker](#prerequisites). Replace the `docker run` command with `nvidia-docker run` or `docker run --runtime=nvidia`.


## Building The TensorRT OSS Components

* Generate Makefiles and build.

	```bash
	cd $TRT_SOURCE
	mkdir -p build && cd build 
	cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_BIN_DIR=`pwd`/out
	make -j$(nproc)
	```

	> NOTE:
	> 1. The default CUDA version used by CMake is 10.1. To override this, for example to 9.0, append `-DCUDA_VERSION=9.0` to the cmake command.
	> 2. Samples may fail to link on CentOS7. To work around this create the following symbolic link:
	> ```bash
	> ln -s $TRT_BIN_DIR/libnvinfer_plugin.so $TRT_BIN_DIR/libnvinfer_plugin.so.6
	> ```

	The required CMake arguments are:

	- `TRT_LIB_DIR`: Path to the TensorRT installation directory containing libraries.

	- `TRT_BIN_DIR`: Output directory where generated build artifacts will be copied.

	The following CMake build parameters are *optional*:

	- `CMAKE_BUILD_TYPE`: Specify if binaries generated are for release or debug (contain debug symbols). Values consists of [`Release`] | `Debug`

	- `CUDA_VERISON`: The version of CUDA to target, for example [`10.1`].

	- `CUDNN_VERSION`: The version of cuDNN to target, for example [`7.5`].

	- `PROTOBUF_VERSION`:  The version of Protobuf to use, for example [`3.8.x`]. Note: Changing this will not configure CMake to use a system version of Protobuf, it will configure CMake to download and try building that version.

	- `CMAKE_TOOLCHAIN_FILE`: The path to a toolchain file for cross compilation.

	- `BUILD_PARSERS`: Specify if the parsers should be built, for example [`ON`] | `OFF`.  If turned OFF, CMake will try to find precompiled versions of the parser libraries to use in compiling samples. First in `${TRT_LIB_DIR}`, then on the system. If the build type is Debug, then it will prefer debug builds of the libraries before release versions if available.

	- `BUILD_PLUGINS`: Specify if the plugins should be built, for example [`ON`] | `OFF`. If turned OFF, CMake will try to find a precompiled version of the plugin library to use in compiling samples. First in `${TRT_LIB_DIR}`, then on the system. If the build type is Debug, then it will prefer debug builds of the libraries before release versions if available.

	- `BUILD_SAMPLES`: Specify if the samples should be built, for example [`ON`] | `OFF`.

	Other build options with limited applicability:

	- `NVINTERNAL`: Used by TensorRT team for internal builds. Values consists of [`OFF`] | `ON`.

	- `PROTOBUF_INTERNAL_VERSION`: The version of protobuf to use, for example [`10.0`].  Only applicable if `NVINTERNAL` is also enabled.

	- `NVPARTNER`: For use by NVIDIA partners with exclusive source access.  Values consists of [`OFF`] | `ON`.

	- `CUB_VERSION`: The version of CUB to use, for example [`1.8.0`].

	- `GPU_ARCHS`: GPU (SM) architectures to target. By default we generate CUDA code for the latest SM version. If lower SM versions are desired, they can be specified here as a comma separated list. Table of compute capabilities of NVIDIA GPUs can be found [here](https://developer.nvidia.com/cuda-gpus). Examples:
	  - Titan V: `-DGPU_ARCHS="70"`
	  - Tesla V100: `-DGPU_ARCHS="70"`
	  - GeForce RTX 2080: `-DGPU_ARCHS="75"`
	  - Tesla T4: `-DGPU_ARCHS="75"`

## Install the TensorRT OSS Components [Optional]

* Copy the build artifacts into the TensorRT installation directory, updating the installation.
  * TensorRT installation directory is determined as `$TRT_LIB_DIR/..`
  * Installation might require superuser privileges depending on the path and permissions of files being replaced.
  * Installation is not supported in cross compilation scenario. Please copy the result files from `build/out` folder into the target device.

	```bash
	sudo make install
	```
* Verify the TensorRT samples have been installed correctly.

	```bash
	cd $TRT_LIB_DIR/../bin/
	./sample_googlenet
	```

	If the sample was installed correctly, the following information will be printed out in the terminal.

	```bash
	[08/23/2019-22:08:57] [I] Building and running a GPU inference engine for GoogleNet
	[08/23/2019-22:08:59] [I] [TRT] Some tactics do not have sufficient workspace memory to run. Increasing workspace size may increase performance, please check verbose output.
	[08/23/2019-22:09:05] [I] [TRT] Detected 1 inputs and 1 output network tensors.
	[08/23/2019-22:09:05] [I] Ran /tensorrt/bin/sample_googlenet with: 
	[08/23/2019-22:09:05] [I] Input(s): data 
	[08/23/2019-22:09:05] [I] Output(s): prob 
	&&&& PASSED TensorRT.sample_googlenet # /tensorrt/bin/sample_googlenet
	```

## Useful Resources

#### TensorRT

* [TensorRT Homepage](https://developer.nvidia.com/tensorrt)
* [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)
* [TensorRT Sample Support Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html)
* [TensorRT Discussion Forums](https://devtalk.nvidia.com/default/board/304/tensorrt/)


## Known Issues

#### TensorRT 6.0.1
* See [Release Notes](https://docs.nvidia.com/deeplearning/sdk/tensorrt-release-notes/index.html).
